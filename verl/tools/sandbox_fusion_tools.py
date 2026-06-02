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

"""Sandbox-Fusion native tool compatibility layer.

Some recipes still import ``verl.tools.sandbox_fusion_tools.SandboxFusionTool``.
The current tool stack only provides the generic ``BaseTool`` contract, so this
module keeps the old recipe contract while returning the current ``ToolResponse``
objects expected by ``ToolAgentLoop``.
"""

from __future__ import annotations

import asyncio
import json
from typing import Any, Callable

from verl.tools.base_tool import BaseTool
from verl.tools.schemas import OpenAIFunctionToolSchema, ToolResponse
from verl.utils.reward_score.sandbox_fusion.utils import call_sandbox_api


class _ExecuteProxy:
    async def remote(self, fn: Callable, *args, **kwargs):
        return await asyncio.to_thread(fn, *args, **kwargs)


class _LocalExecutionPool:
    def __init__(self):
        self.execute = _ExecuteProxy()


class SandboxFusionTool(BaseTool):
    """Native code-execution tool backed by a Sandbox-Fusion HTTP service."""

    def __init__(self, config: dict, tool_schema: OpenAIFunctionToolSchema):
        super().__init__(config, tool_schema)
        self.sandbox_fusion_url = config["sandbox_fusion_url"]
        self.default_timeout = int(config.get("default_timeout", 30))
        self.default_language = config.get("default_language", "python")
        self.memory_limit_mb = int(config.get("memory_limit_mb", 1024))
        self.execution_pool = _LocalExecutionPool()

    def execute_code(self, instance_id: str, code: str, timeout: int | None = None, language: str | None = None):
        del instance_id
        timeout = int(timeout or self.default_timeout)
        language = language or self.default_language

        response, error = call_sandbox_api(
            sandbox_fusion_url=self.sandbox_fusion_url,
            code=code,
            stdin=None,
            compile_timeout=timeout,
            run_timeout=timeout,
            memory_limit_mb=self.memory_limit_mb,
            language=language,
        )
        if error:
            return ToolResponse(text=error)

        run_result = (response or {}).get("run_result") or {}
        compile_result = (response or {}).get("compile_result") or {}
        stdout = run_result.get("stdout") or ""
        stderr = run_result.get("stderr") or ""
        compile_stderr = compile_result.get("stderr") or ""
        status = (response or {}).get("status")
        run_status = run_result.get("status")
        return_code = run_result.get("return_code")

        if stdout:
            return ToolResponse(text=stdout.rstrip("\n"))

        details = {
            "status": status,
            "run_status": run_status,
            "return_code": return_code,
            "stderr": stderr,
            "compile_stderr": compile_stderr,
        }
        return ToolResponse(text=json.dumps(details, ensure_ascii=False))
