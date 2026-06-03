# Copyright 2026 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
from __future__ import annotations

import asyncio
import os
from pathlib import Path
from typing import Any
from uuid import uuid4

try:
    from vllm import LLM, SamplingParams
    from vllm.v1.metrics.loggers import StatLoggerManager
    try:
        from vllm.inputs import TokensPrompt
    except Exception:
        TokensPrompt = None
except Exception:  # pragma: no cover - vLLM is only required for --engine vllm.
    LLM = None
    SamplingParams = None
    StatLoggerManager = None
    TokensPrompt = None

try:
    from .base import EngineResponse, ReplayEngine
except ImportError:
    from base import EngineResponse, ReplayEngine

try:
    from ..schema import ReplayLLMRequest
except ImportError:
    from schema import ReplayLLMRequest

try:
    from verl.utils.rollout_perf import init_rollout_perf
    from verl.utils.rollout_perf.vllm_metrics import RolloutPerfRequestKVMetricsSampler, RolloutPerfVLLMStatLogger
except Exception:  # pragma: no cover - allows importing the module without full verl runtime deps.
    init_rollout_perf = None
    RolloutPerfRequestKVMetricsSampler = None
    RolloutPerfVLLMStatLogger = None


class VLLMReplayEngine(ReplayEngine):
    """vLLM-backed replay engine with a small async batching queue."""

    backend_name = "vllm"

    def __init__(
        self,
        *,
        model: str,
        output_dir: str | Path,
        tensor_parallel_size: int = 1,
        max_model_len: int | None = None,
        gpu_memory_utilization: float = 0.90,
        batch_wait_ms: float = 5.0,
        max_batch_size: int = 32,
        sample_interval_ms: int = 100,
        trust_remote_code: bool = True,
        enforce_eager: bool = False,
    ):
        if LLM is None or SamplingParams is None or StatLoggerManager is None:
            raise RuntimeError("vLLM is required for --engine vllm")
        self.model = model
        self.output_dir = Path(output_dir).expanduser().resolve()
        self.tensor_parallel_size = max(1, int(tensor_parallel_size))
        self.max_model_len = max_model_len
        self.batch_wait_s = max(0.0, float(batch_wait_ms)) / 1000.0
        self.max_batch_size = max(1, int(max_batch_size))
        self.sample_interval_ms = max(1, int(sample_interval_ms))
        self._queue: asyncio.Queue[tuple[ReplayLLMRequest, asyncio.Future[EngineResponse]]] = asyncio.Queue()
        self._batch_task: asyncio.Task[Any] | None = None
        self._closed = False

        tmpdir = os.environ.setdefault("TMPDIR", f"/tmp/rollout_perf_replay_vllm_{os.getpid()}")
        os.environ.setdefault("VLLM_RPC_BASE_PATH", os.path.join(tmpdir, "rpc"))
        Path(os.environ["TMPDIR"]).mkdir(parents=True, exist_ok=True)
        Path(os.environ["VLLM_RPC_BASE_PATH"]).mkdir(parents=True, exist_ok=True)
        os.environ["VERL_ROLLOUT_PERF_OUTPUT_DIR"] = str(self.output_dir)
        os.environ["VERL_ROLLOUT_PERF_ENGINE_INTERNAL_SAMPLE_INTERVAL_MS"] = str(self.sample_interval_ms)
        os.environ["VERL_ROLLOUT_PERF_FORCE_ENGINE_INTERNAL_ACTIVE"] = "1"
        os.environ.setdefault("VERL_ROLLOUT_PERF_FLUSH_INTERVAL_S", "1.0")

        if (
            init_rollout_perf is None
            or RolloutPerfRequestKVMetricsSampler is None
            or RolloutPerfVLLMStatLogger is None
        ):
            raise RuntimeError(
                "verl rollout perf vLLM metrics helpers are required for --engine vllm; "
                "run replay from the repository root or install verl on PYTHONPATH."
            )
        init_rollout_perf(
            {
                "enable": True,
                "output_dir": str(self.output_dir),
                "format": "jsonl",
                "capture_content": False,
                "flush_interval_s": 1.0,
            },
            role="vllm_server",
        )

        llm_kwargs: dict[str, Any] = {
            "model": model,
            "tensor_parallel_size": self.tensor_parallel_size,
            "gpu_memory_utilization": gpu_memory_utilization,
            "trust_remote_code": trust_remote_code,
            "enforce_eager": enforce_eager,
            "disable_log_stats": False,
        }
        if max_model_len is not None:
            llm_kwargs["max_model_len"] = max_model_len
        self.llm = LLM(**llm_kwargs)
        self._install_stat_logger()
        self.token_id = self._choose_prompt_token_id()
        self.kv_sampler = RolloutPerfRequestKVMetricsSampler(
            context={
                "engine_backend": "vllm",
                "engine_index": 0,
                "replay": True,
            },
            sample_interval_ms=self.sample_interval_ms,
        )

    def _install_stat_logger(self) -> None:
        llm_engine = getattr(self.llm, "llm_engine", None)
        vllm_config = getattr(llm_engine, "vllm_config", None)
        if llm_engine is None or vllm_config is None:
            raise RuntimeError("Unable to install rollout perf vLLM stat logger: missing llm_engine.vllm_config")
        llm_engine.logger_manager = StatLoggerManager(
            vllm_config=vllm_config,
            custom_stat_loggers=[RolloutPerfVLLMStatLogger],
            enable_default_loggers=False,
        )
        llm_engine.logger_manager.log_engine_initialized()

    def _choose_prompt_token_id(self) -> int:
        try:
            tokenizer = self.llm.get_tokenizer()
            ids = tokenizer.encode("hello", add_special_tokens=False)
            if ids:
                return int(ids[0])
        except Exception:
            pass
        return 100

    async def generate(self, request: ReplayLLMRequest) -> EngineResponse:
        if request.output_token_count <= 0:
            return EngineResponse(output_token_count=0, stop_reason=request.stop_reason)
        if self._closed:
            raise RuntimeError("VLLMReplayEngine is closed")
        if self._batch_task is None:
            self._batch_task = asyncio.create_task(self._batch_loop())
        loop = asyncio.get_running_loop()
        future: asyncio.Future[EngineResponse] = loop.create_future()
        await self._queue.put((request, future))
        return await future

    async def _batch_loop(self) -> None:
        while True:
            request, future = await self._queue.get()
            batch = [(request, future)]
            if self.batch_wait_s > 0:
                await asyncio.sleep(self.batch_wait_s)
            while len(batch) < self.max_batch_size:
                try:
                    batch.append(self._queue.get_nowait())
                except asyncio.QueueEmpty:
                    break
            try:
                responses = await asyncio.to_thread(self._run_batch, [item[0] for item in batch])
                for (_, item_future), response in zip(batch, responses):
                    if not item_future.done():
                        item_future.set_result(response)
            except Exception as exc:
                for _, item_future in batch:
                    if not item_future.done():
                        item_future.set_exception(exc)

    def _run_batch(self, requests: list[ReplayLLMRequest]) -> list[EngineResponse]:
        active_request_ids: list[str] = []
        prompts = []
        sampling_params = []
        for request in requests:
            request_id = request.engine_request_id or request.logical_request_id or uuid4().hex
            active_request_ids.append(str(request_id))
            prompt_len = max(1, int(request.prompt_token_count))
            output_len = max(1, int(request.output_token_count))
            if self.max_model_len is not None and prompt_len + output_len > self.max_model_len:
                raise ValueError(
                    f"Replay request exceeds max_model_len: prompt={prompt_len}, output={output_len}, "
                    f"max_model_len={self.max_model_len}"
                )
            if self.kv_sampler is not None:
                self.kv_sampler.start_request(str(request_id), prompt_len)
            prompt_ids = [self.token_id] * prompt_len
            prompts.append(TokensPrompt(prompt_token_ids=prompt_ids) if TokensPrompt is not None else prompt_ids)
            sampling_params.append(
                SamplingParams(
                    max_tokens=output_len,
                    min_tokens=output_len,
                    ignore_eos=True,
                    temperature=0.0,
                    detokenize=False,
                )
            )
        try:
            outputs = self.llm.generate(prompts, sampling_params, use_tqdm=False)
            responses = []
            for request_id, request, output in zip(active_request_ids, requests, outputs):
                token_ids = output.outputs[0].token_ids if output.outputs else []
                output_len = len(token_ids)
                if self.kv_sampler is not None:
                    self.kv_sampler.update_request(request_id, output_len)
                ttft_ns = request.original_ttft_ns
                responses.append(
                    EngineResponse(
                        output_token_count=output_len,
                        stop_reason=output.outputs[0].finish_reason if output.outputs else request.stop_reason,
                        ttft_ns=ttft_ns,
                    )
                )
            return responses
        finally:
            for request_id in active_request_ids:
                if self.kv_sampler is not None:
                    self.kv_sampler.finish_request(request_id)

    async def close(self) -> None:
        self._closed = True
        if self._batch_task is not None:
            self._batch_task.cancel()
            try:
                await self._batch_task
            except asyncio.CancelledError:
                pass
        if self.kv_sampler is not None:
            self.kv_sampler.close()
        shutdown = getattr(self.llm, "shutdown", None)
        if callable(shutdown):
            shutdown()
