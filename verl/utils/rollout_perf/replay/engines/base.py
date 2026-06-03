# Copyright 2026 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
from __future__ import annotations

from dataclasses import dataclass

try:
    from ..schema import ReplayLLMRequest
except ImportError:  # Allows importing engines.mock from direct runner.py execution.
    from schema import ReplayLLMRequest


@dataclass
class EngineResponse:
    output_token_count: int
    stop_reason: str | None = None
    ttft_ns: int | None = None


class ReplayEngine:
    async def generate(self, request: ReplayLLMRequest) -> EngineResponse:
        raise NotImplementedError
