# Copyright 2025 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import asyncio
import json
import logging
import os
from enum import Enum
from typing import Any, Optional
from uuid import uuid4

import torch
from PIL import Image

from verl.experimental.agent_loop.agent_loop import (
    AgentLoopBase,
    AgentLoopOutput,
    ToolListWrap,
    register,
)
from verl.experimental.agent_loop.tool_parser import FunctionCall, ToolParser
from verl.experimental.agent_loop.utils import build_gpt_oss_tool_response_text
from verl.tools.function_tool import FunctionTool, normalize_function_tool_return
from verl.tools.schemas import ToolResponse
from verl.utils.profiler import simple_timer
from verl.utils.rollout_perf import is_current_trace_enabled, push_trace_context, start_span
from verl.utils.rollout_trace import rollout_trace_op
from verl.workers.rollout.replica import TokenOutput

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))

SPEC_DECODE_EXTRA_KEYS = (
    "spec_num_draft_tokens",
    "spec_num_accepted_tokens",
    "spec_num_verify_steps",
)


class AgentState(Enum):
    PENDING = "pending"
    GENERATING = "generating"
    PROCESSING_TOOLS = "processing_tools"
    TERMINATED = "terminated"


class AgentData:
    """Encapsulates all state variables for the agent loop. AgentData is passed to tool calling in case that
    tool may need to access full history state. User can store any tool session data in `extra_fields`."""

    def __init__(
        self,
        messages: list[dict[str, Any]],
        image_data: list[Image.Image],
        video_data: list[tuple[torch.Tensor, dict[str, Any]]],
        audio_data: Optional[list[Any]],
        mm_processor_kwargs: Optional[dict[str, Any]],
        metrics: dict[str, Any],
        request_id: str,
        tools_kwargs: dict[str, Any],
    ):
        self.messages = messages
        self.image_data = image_data
        self.video_data = video_data
        self.audio_data = audio_data
        self.mm_processor_kwargs = mm_processor_kwargs or {}
        self.metrics = metrics
        self.request_id = request_id
        self.tools_kwargs = tools_kwargs

        # State variables
        self.prompt_ids: list[int] = []
        self.response_ids: list[int] = []
        self.response_mask: list[int] = []
        self.response_logprobs: list[float] = []
        self.turn_scores: list[float] = []
        self.tool_rewards: list[float] = []
        self.user_turns = 0
        self.assistant_turns = 0

        # Temporary state for tool calls
        self.tool_calls: list[FunctionCall] = []

        self.routed_experts = None

        # Extra fields for dynamic addition, e.g., tool session data
        self.extra_fields: dict[str, Any] = {}


@register("tool_agent")
class ToolAgentLoop(AgentLoopBase):
    def __init__(self, *args, tools: Optional[ToolListWrap] = None, **kwargs):
        """Initialize the tool agent loop.

        Args:
            tools: Tools to use for the tool agent loop.
        """
        super().__init__(*args, **kwargs)

        self.max_user_turns = self.rollout_config.multi_turn.max_user_turns
        self.max_assistant_turns = self.rollout_config.multi_turn.max_assistant_turns
        self.max_parallel_calls = self.rollout_config.multi_turn.max_parallel_calls
        self.max_tool_response_length = self.rollout_config.multi_turn.max_tool_response_length
        self.tool_response_truncate_side = self.rollout_config.multi_turn.tool_response_truncate_side

        tool_list = tools.tools if tools else []
        self.tools = {tool.name: tool for tool in tool_list}
        self.tool_schemas = [tool.tool_schema.model_dump(exclude_unset=True, exclude_none=True) for tool in tool_list]
        self.tool_parser = ToolParser.get_tool_parser(self.rollout_config.multi_turn.format, self.tokenizer)
        self.tool_parser_name = self.rollout_config.multi_turn.format

        self.prompt_length = self.rollout_config.prompt_length
        self.response_length = self.rollout_config.response_length

    def _safe_text_token_count(self, text: Optional[str]) -> int:
        if not text or not is_current_trace_enabled():
            return 0
        try:
            return len(self.tokenizer.encode(text, add_special_tokens=False))
        except Exception:
            return 0

    @rollout_trace_op
    async def run(self, sampling_params: dict[str, Any], **kwargs) -> AgentLoopOutput:
        messages = list(kwargs["raw_prompt"])

        # extract multimodal inputs from messages
        multi_modal_data = await self.process_multi_modal_info(messages)
        images = multi_modal_data.get("images")
        videos = multi_modal_data.get("videos")
        audios = multi_modal_data.get("audios")
        mm_processor_kwargs = self._get_mm_processor_kwargs(audios)

        metrics = {}
        request_id = uuid4().hex
        tools_kwargs = kwargs.get("tools_kwargs", {})

        agent_data = AgentData(
            messages=messages,
            image_data=images,
            video_data=videos,
            audio_data=audios,
            mm_processor_kwargs=mm_processor_kwargs,
            metrics=metrics,
            request_id=request_id,
            tools_kwargs=tools_kwargs,
        )
        agent_data.extra_fields.update(
            {
                "tool_call_counts": 0,
                "llm_generated_token_counts": 0,
                "tool_response_token_counts": 0,
            }
        )

        # Per-sample tool selection: filter global tools by extra_info.tool_selection
        extra_info = kwargs.get("extra_info", {}) or {}
        tool_selection = extra_info.get("tool_selection")
        if tool_selection and self.tools:
            selected = {name: self.tools[name] for name in tool_selection if name in self.tools}
            agent_data._active_tools = selected
            agent_data._active_tool_schemas = [
                t.tool_schema.model_dump(exclude_unset=True, exclude_none=True) for t in selected.values()
            ]
        else:
            agent_data._active_tools = self.tools
            agent_data._active_tool_schemas = self.tool_schemas

        # State machine loop
        with push_trace_context(logical_request_id=request_id, agent_loop="tool_agent"):
            state = AgentState.PENDING
            while state != AgentState.TERMINATED:
                if state == AgentState.PENDING:
                    state = await self._handle_pending_state(agent_data, sampling_params)
                elif state == AgentState.GENERATING:
                    state = await self._handle_generating_state(agent_data, sampling_params)
                elif state == AgentState.PROCESSING_TOOLS:
                    state = await self._handle_processing_tools_state(agent_data)
                else:
                    logger.error(f"Invalid state: {state}")
                    state = AgentState.TERMINATED

        # Finalize output
        response_ids = agent_data.prompt_ids[-len(agent_data.response_mask) :]
        prompt_ids = agent_data.prompt_ids[: len(agent_data.prompt_ids) - len(agent_data.response_mask)]
        multi_modal_data = {}
        if agent_data.image_data is not None:
            multi_modal_data["images"] = agent_data.image_data
        if agent_data.video_data is not None:
            multi_modal_data["videos"] = agent_data.video_data
        if agent_data.audio_data is not None:
            multi_modal_data["audios"] = agent_data.audio_data

        output: AgentLoopOutput = AgentLoopOutput(
            prompt_ids=prompt_ids,
            response_ids=response_ids[: self.response_length],
            response_mask=agent_data.response_mask[: self.response_length],
            multi_modal_data=multi_modal_data,
            mm_processor_kwargs=agent_data.mm_processor_kwargs,
            response_logprobs=agent_data.response_logprobs[: self.response_length]
            if agent_data.response_logprobs
            else None,
            num_turns=agent_data.user_turns + agent_data.assistant_turns + 1,
            metrics=agent_data.metrics,
            routed_experts=(
                agent_data.routed_experts[: len(prompt_ids) + self.response_length]
                if agent_data.routed_experts is not None
                else None
            ),
            extra_fields=agent_data.extra_fields,
        )
        output.extra_fields.update({"turn_scores": agent_data.turn_scores, "tool_rewards": agent_data.tool_rewards})
        return output

    async def _handle_pending_state(self, agent_data: AgentData, sampling_params: dict[str, Any]) -> AgentState:
        """Handle the pending state: prepare the prompt and start generation."""
        schemas = getattr(agent_data, "_active_tool_schemas", self.tool_schemas)
        prompt_ids = await self.apply_chat_template(
            agent_data.messages,
            tools=schemas,
            images=agent_data.image_data,
            videos=agent_data.video_data,
            audios=agent_data.audio_data,
            mm_processor_kwargs=agent_data.mm_processor_kwargs,
        )
        agent_data.prompt_ids = prompt_ids
        return AgentState.GENERATING

    async def _handle_generating_state(
        self, agent_data: AgentData, sampling_params: dict[str, Any], ignore_termination: bool = False
    ) -> AgentState:
        """Handle the generating state: generate model response and check for tool calls."""
        # Inject tool parser stop tokens so generation halts after each tool call
        if self.tool_parser.stop_token_ids:
            stop_token_ids = list(set((sampling_params.get("stop_token_ids") or []) + self.tool_parser.stop_token_ids))
            sampling_params = {**sampling_params, "stop_token_ids": stop_token_ids}

        turn_index = agent_data.assistant_turns
        with push_trace_context(turn_index=turn_index, turn_kind="assistant"):
            llm_turn_span = start_span(
                "llm_turn",
                {
                    "input_token_count": len(agent_data.prompt_ids),
                    "requested_max_tokens": sampling_params.get("max_tokens")
                    or sampling_params.get("max_new_tokens")
                    or self.response_length,
                },
            )
            try:
                with simple_timer("generate_sequences", agent_data.metrics):
                    output: TokenOutput = await self.server_manager.generate(
                        request_id=agent_data.request_id,
                        prompt_ids=agent_data.prompt_ids,
                        sampling_params=sampling_params,
                        image_data=agent_data.image_data,
                        video_data=agent_data.video_data,
                        audio_data=agent_data.audio_data,
                        mm_processor_kwargs=agent_data.mm_processor_kwargs,
                    )
                llm_turn_span.finish(
                    {
                        "output_token_count": len(output.token_ids),
                        "stop_reason": output.stop_reason,
                        "num_preempted": output.num_preempted,
                    }
                )
            except Exception as e:
                llm_turn_span.finish(status="error", error=e)
                raise
        # first time to set num_preempted
        if agent_data.metrics.get("num_preempted") is None:
            agent_data.metrics["num_preempted"] = output.num_preempted if output.num_preempted is not None else -1
        # then add num_preempted to the metrics
        else:
            agent_data.metrics["num_preempted"] += output.num_preempted if output.num_preempted is not None else 0

        if agent_data.assistant_turns == 0:
            # Preserve perf counters while keeping the original first-turn engine extra fields.
            perf_counters = {
                "tool_call_counts": agent_data.extra_fields.get("tool_call_counts", 0),
                "llm_generated_token_counts": agent_data.extra_fields.get("llm_generated_token_counts", 0),
                "tool_response_token_counts": agent_data.extra_fields.get("tool_response_token_counts", 0),
            }
            agent_data.extra_fields.update(output.extra_fields)
            agent_data.extra_fields.update(perf_counters)
        else:
            # Multi-round calls, only update the maximum max_global_steps.
            max_global_steps = output.extra_fields.get("max_global_steps", None)
            if max_global_steps:
                agent_data.extra_fields["max_global_steps"] = max_global_steps
            for key in SPEC_DECODE_EXTRA_KEYS:
                if key in output.extra_fields and key in agent_data.extra_fields:
                    agent_data.extra_fields[key] = int(agent_data.extra_fields[key]) + int(output.extra_fields[key])

        agent_data.assistant_turns += 1
        agent_data.response_ids = output.token_ids
        agent_data.prompt_ids += agent_data.response_ids
        agent_data.response_mask += [1] * len(agent_data.response_ids)
        agent_data.extra_fields["llm_generated_token_counts"] += len(agent_data.response_ids)
        if output.log_probs:
            agent_data.response_logprobs += output.log_probs

        if output.routed_experts is not None:
            agent_data.routed_experts = output.routed_experts

        # Check termination conditions
        if not ignore_termination and len(agent_data.response_mask) >= self.response_length:
            return AgentState.TERMINATED
        if self.max_assistant_turns and agent_data.assistant_turns >= self.max_assistant_turns:
            return AgentState.TERMINATED
        if self.max_user_turns and agent_data.user_turns >= self.max_user_turns:
            return AgentState.TERMINATED

        # Extract tool calls (use per-sample tools if routed)
        active_tools = getattr(agent_data, "_active_tools", self.tools)
        tools = [tool.tool_schema for tool in active_tools.values()]
        _, agent_data.tool_calls = await self.tool_parser.extract_tool_calls(agent_data.response_ids, tools)

        if agent_data.tool_calls:
            return AgentState.PROCESSING_TOOLS
        else:
            return AgentState.TERMINATED

    async def _handle_processing_tools_state(self, agent_data: AgentData) -> AgentState:
        """Handle the processing tools state: execute tool calls and prepare tool responses."""
        add_messages: list[dict[str, Any]] = []
        new_images_this_turn: list[Any] = []  # Local variable instead of agent_data attribute

        tasks = []
        tool_call_names = []
        for tool_call in agent_data.tool_calls[: self.max_parallel_calls]:
            tasks.append(self._call_tool(tool_call, agent_data.tools_kwargs, agent_data))
            tool_call_names.append(tool_call.name)
        agent_data.extra_fields["tool_call_counts"] += len(tasks)

        tool_turn_span = None
        with push_trace_context(turn_index=agent_data.user_turns, turn_kind="tool"):
            tool_turn_span = start_span("tool_turn", {"tool_call_count": len(tasks)})
            try:
                with simple_timer("tool_calls", agent_data.metrics):
                    responses = await asyncio.gather(*tasks)
            except Exception as e:
                tool_turn_span.finish(status="error", error=e)
                raise

        # Process tool responses and update multi_modal_data
        # Removed: agent_data.new_images_this_turn = []
        for tool_response, tool_reward, _ in responses:
            # Create message from tool response
            if tool_response.image or tool_response.video:
                # Multi-modal content with structured format
                if not getattr(self.processor, "image_processor", None):
                    raise ValueError(
                        "Multimedia data can only be processed by `processor`, but the processor is None. "
                        "This error is often caused if you are using a LLM model but your tool returns multimodal "
                        "data. Plase use a vlm as the base model."
                    )
                content = []
                if tool_response.image:
                    content.append({"type": "image"})
                if tool_response.video:
                    content.append({"type": "video"})
                if tool_response.text:
                    content.append({"type": "text", "text": tool_response.text})
                message = {"role": "tool", "content": content}
            else:
                # Text-only content
                message = {"role": "tool", "content": tool_response.text or ""}

            add_messages.append(message)

            # Handle image data
            if tool_response.image:
                # Add new image data
                if isinstance(tool_response.image, list):
                    # Ensure all elements in the list are valid image objects
                    for img in tool_response.image:
                        if img is not None:  # Add a check to ensure the image is not None
                            new_images_this_turn.append(img)  # Using local variable
                else:
                    # Ensure the image is not None
                    if tool_response.image is not None:
                        new_images_this_turn.append(tool_response.image)  # Using local variable

            # Handle video data
            if tool_response.video:
                # Currently not supported, raise informative error
                logger.warning("Multimedia type 'video' is not currently supported. Only 'image' is supported.")
                raise NotImplementedError(
                    "Multimedia type 'video' is not currently supported. Only 'image' is supported."
                )

            if tool_reward is not None:
                agent_data.tool_rewards.append(tool_reward)

        agent_data.messages.extend(add_messages)

        if self.tool_parser_name == "gpt-oss":
            logger.info("manually format tool responses for gpt-oss")
            tool_response_text = build_gpt_oss_tool_response_text(add_messages, tool_call_names)
            response_ids = await self.loop.run_in_executor(
                None, lambda: self.tokenizer.encode(tool_response_text, add_special_tokens=False)
            )
        elif self.tool_parser_name == "gemma4":
            # Gemma4's chat template drops tool responses when passed without the preceding
            # assistant tool_call message. Manually format the response tokens.
            # Format: <|tool_response>response:func_name{value:<|"|>content<|"|>}<tool_response|>
            parts = []
            for msg, name in zip(add_messages, tool_call_names, strict=True):
                content = msg.get("content", "")
                if isinstance(content, list):
                    content = "".join([item.get("text", "") for item in content if item.get("type") == "text"])
                if isinstance(content, list):
                    content = "".join([item.get("text", "") for item in content if item.get("type") == "text"])
                parts.append(f'<|tool_response>response:{name}{{value:<|"|>{content}<|"|>}}<tool_response|>')
            tool_response_text = "".join(parts)
            response_ids = await self.loop.run_in_executor(
                None, lambda: self.tokenizer.encode(tool_response_text, add_special_tokens=False)
            )
        else:
            # Note that we have to pass None to the images and videos if there are no new images / videos
            # to stay compatible with downstream image processing logic!
            images = new_images_this_turn if new_images_this_turn else None
            videos = None
            response_ids = await self.apply_chat_template(
                add_messages,
                images=images,
                videos=videos,
                remove_system_prompt=True,
            )

        if len(agent_data.response_mask) + len(response_ids) >= self.response_length:
            if tool_turn_span is not None:
                tool_turn_span.finish(
                    {
                        "tool_response_token_count": len(response_ids),
                        "terminated_by_response_length": True,
                    }
                )
            return AgentState.TERMINATED
        # Update prompt_ids and response_mask

        if new_images_this_turn:
            if agent_data.image_data is None:
                agent_data.image_data = []
            elif not isinstance(agent_data.image_data, list):
                agent_data.image_data = [agent_data.image_data]
            for img in new_images_this_turn:
                agent_data.image_data.append(img)

        agent_data.prompt_ids += response_ids
        agent_data.response_mask += [0] * len(response_ids)
        agent_data.extra_fields["tool_response_token_counts"] += len(response_ids)
        if agent_data.response_logprobs:
            agent_data.response_logprobs += [0.0] * len(response_ids)
        agent_data.user_turns += 1
        if tool_turn_span is not None:
            tool_turn_span.finish(
                {
                    "tool_response_token_count": len(response_ids),
                    "terminated_by_response_length": False,
                }
            )
        return AgentState.GENERATING

    async def _call_tool(
        self, tool_call: FunctionCall, tools_kwargs: dict[str, Any], agent_data: AgentData
    ) -> tuple[ToolResponse, float, dict]:
        """Call tool and return tool response.

        Dispatches between two contracts:
        - ``FunctionTool``: stateless function-based tool. Invoked directly with
          parsed arguments; no lifecycle.
        - ``BaseTool`` subclass: stateful tool with full lifecycle.
        """
        active_tools = getattr(agent_data, "_active_tools", self.tools)
        tool_name = tool_call.name
        tool_call_id = uuid4().hex
        arguments_text = tool_call.arguments or ""
        tool_response_text = ""
        tool_response_kwargs = None
        tool_reward = 0.0
        res = {}
        status = "ok"
        error = None

        with push_trace_context(tool_call_id=tool_call_id, tool_name=tool_name):
            span = start_span(
                "tool_call",
                {
                    "tool_name": tool_name,
                    "argument_byte_count": len(arguments_text.encode("utf-8")),
                    "argument_token_count": self._safe_text_token_count(arguments_text),
                },
            )
            tool, instance_id = None, None
            try:
                # Validate tool name
                if tool_name not in active_tools:
                    available = list(active_tools.keys())
                    msg = f"Unknown function '{tool_name}'. Available tools: {available}"
                    logger.warning(msg)
                    status = "unknown_tool"
                    tool_response_text = msg
                    span.finish(
                        {
                            "output_byte_count": len(tool_response_text.encode("utf-8")),
                            "output_token_count": self._safe_text_token_count(tool_response_text),
                            "truncated": False,
                            "has_image": False,
                            "has_video": False,
                        },
                        status=status,
                    )
                    return ToolResponse(text=msg), 0.0, {}

                # Validate tool arguments
                try:
                    tool_args = json.loads(arguments_text)
                except (json.JSONDecodeError, TypeError) as e:
                    msg = f"Invalid JSON in arguments for '{tool_name}': {e}"
                    logger.warning(msg)
                    status = "invalid_arguments"
                    error = e
                    tool_response_text = msg
                    span.finish(
                        {
                            "output_byte_count": len(tool_response_text.encode("utf-8")),
                            "output_token_count": self._safe_text_token_count(tool_response_text),
                            "truncated": False,
                            "has_image": False,
                            "has_video": False,
                        },
                        status=status,
                        error=error,
                    )
                    return ToolResponse(text=msg), 0.0, {}

                # Execute tool
                tool = active_tools[tool_name]

                if isinstance(tool, FunctionTool):
                    # Function-based tools have no lifecycle; call directly.
                    # Note: tools_kwargs (create_kwargs / release_kwargs) is intentionally
                    # ignored here. Function tools are stateless and per-trajectory state
                    # injection is not supported by design; use a BaseTool subclass instead.
                    raw = await tool.call(tool_args)
                    tool_execution_response, tool_reward, res = normalize_function_tool_return(raw)
                else:
                    # BaseTool subclass
                    kwargs = tools_kwargs.get(tool_name, {})
                    instance_id, _ = await tool.create(create_kwargs=kwargs.get("create_kwargs", {}))
                    tool_execution_response, tool_reward, res = await tool.execute(
                        instance_id, tool_args, agent_data=agent_data
                    )

                tool_response_text = tool_execution_response.text
                truncated = False
                if tool_response_text and len(tool_response_text) > self.max_tool_response_length:
                    truncated = True
                    if self.tool_response_truncate_side == "left":
                        tool_response_text = "(truncated)..." + tool_response_text[-self.max_tool_response_length :]
                    elif self.tool_response_truncate_side == "right":
                        tool_response_text = tool_response_text[: self.max_tool_response_length] + "...(truncated)"
                    else:
                        length = self.max_tool_response_length // 2
                        tool_response_text = (
                            tool_response_text[:length] + "...(truncated)..." + tool_response_text[-length:]
                        )

                # Create ToolResponse from tool execution result
                tool_response_kwargs = {"text": tool_response_text}

                # Add multimedia data if present
                for attr_name in ["image", "video"]:
                    if hasattr(tool_execution_response, attr_name):
                        attr_value = getattr(tool_execution_response, attr_name)
                        if attr_value is not None:
                            tool_response_kwargs[attr_name] = attr_value

                span.finish(
                    {
                        "output_byte_count": len((tool_response_text or "").encode("utf-8")),
                        "output_token_count": self._safe_text_token_count(tool_response_text),
                        "truncated": truncated,
                        "has_image": bool(tool_response_kwargs.get("image")),
                        "has_video": bool(tool_response_kwargs.get("video")),
                    },
                    status=status,
                    error=error,
                )
                return ToolResponse(**tool_response_kwargs), tool_reward, res
            except Exception as e:
                status = "error"
                error = e
                logger.warning(f"Error executing tool '{tool_name}': {e}")
                tool_response_text = f"Error executing tool '{tool_name}': {e}"
                span.finish(
                    {
                        "output_byte_count": len(tool_response_text.encode("utf-8")),
                        "output_token_count": self._safe_text_token_count(tool_response_text),
                        "truncated": False,
                        "has_image": False,
                        "has_video": False,
                    },
                    status=status,
                    error=error,
                )
                return ToolResponse(text=tool_response_text), 0.0, {}
            finally:
                # Only BaseTool instances need release (function tools never set instance_id).
                if tool and instance_id and not isinstance(tool, FunctionTool):
                    await tool.release(instance_id)
