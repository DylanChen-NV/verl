# Copyright 2026 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
from __future__ import annotations

import asyncio

try:
    from .schema import ReplayToolCall
except ImportError:  # Allows direct file execution via runner.py on login nodes.
    from schema import ReplayToolCall


class ToolSleepMock:
    def __init__(self, *, time_scale: float):
        self.time_scale = max(0.0, float(time_scale))

    async def run(self, tool_call: ReplayToolCall) -> dict[str, int | str | None]:
        sleep_s = (tool_call.sleep_ns / 1_000_000_000) * self.time_scale
        if sleep_s > 0:
            await asyncio.sleep(sleep_s)
        return {
            "tool_call_id": tool_call.tool_call_id,
            "tool_name": tool_call.tool_name,
            "output_token_count": tool_call.output_token_count,
        }
