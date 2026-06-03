# Copyright 2026 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
from __future__ import annotations

import asyncio

try:
    from .base import EngineResponse, ReplayEngine
except ImportError:
    from base import EngineResponse, ReplayEngine

try:
    from ..schema import ReplayLLMRequest
except ImportError:
    from schema import ReplayLLMRequest


class MockReplayEngine(ReplayEngine):
    def __init__(self, *, time_scale: float = 0.0, fixed_latency_ms: float | None = None):
        self.time_scale = max(0.0, float(time_scale))
        self.fixed_latency_ms = fixed_latency_ms

    async def generate(self, request: ReplayLLMRequest) -> EngineResponse:
        if self.fixed_latency_ms is not None:
            sleep_s = max(0.0, self.fixed_latency_ms) / 1000.0
        else:
            sleep_s = ((request.original_engine_duration_ns or request.original_llm_duration_ns or 0) / 1_000_000_000)
            sleep_s *= self.time_scale
        if sleep_s > 0:
            await asyncio.sleep(sleep_s)
        return EngineResponse(
            output_token_count=request.output_token_count,
            stop_reason=request.stop_reason,
            ttft_ns=request.original_ttft_ns,
        )
