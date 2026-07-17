#!/usr/bin/env python3
from pathlib import Path


path = Path(__file__).resolve().parents[3] / "dynamic_recompute_trajectory_flexkv_followup_plan.md"
text = path.read_text(encoding="utf-8")
old = """1. **已完成**：创建 Mooncake-first 隔离代码和 runner，完成首轮 A/B/C，并定位严格证据缺口。
2. 验证 final metadata 进入现有 finished-only step；明确不新增 Mooncake worker API/class 或 EngineCore dispatch 接口。
3. 以现有 `pause_generation()` delayed-free completion 作为 aggregate barrier，并补 finished-only dispatch integration test。
4. 申请并持续保留 DFW `2×8 H100` persistent debug 资源，完成 non-sleep 单请求 smoke。
5. 完成 sleep-enabled、strict-cross-node smoke，达到同 request partial-prefix 覆盖率验收。
6. 用最终代码重跑 A、B、C，并持续更新运行进度。
7. 生成 Grafana trajectory、A/B/C 指标和最终报告，固化复现环境并更新项目进展。
"""
new = """1. **已完成**：创建 Mooncake-first 隔离代码和 runner，完成首轮 A/B/C，并定位严格证据缺口。
2. **已完成**：验证 final metadata 进入现有 finished-only step；未新增 Mooncake worker API/class 或 EngineCore dispatch 接口。
3. **已完成**：以现有 `pause_generation()` delayed-free completion 作为 aggregate barrier，并通过 5 个容器测试覆盖 final-only dispatch。
4. **已完成**：申请并持续保留 DFW `2×8 H100` persistent debug 资源；会话为 `dyn-moon-final-dispatch-20260717`。
5. **已完成**：`mcpre1` 完成 sleep-enabled strict-cross-node smoke；原请求在 `10.65.4.75` SEND 8,288 tokens，retry 在 `10.65.4.69` RECV 8,288 tokens，首次仅计算 1 token。完整记录见 `next_tests/dynamic_recompute_trajectory_mooncake/MOONCAKE_CROSS_NODE_PRECHECK_20260717.md`。
6. **进行中**：用最终代码和相同 target cohort 正式重跑 A、B、C，并持续更新运行进度。
7. 待完成：生成 Grafana trajectory、A/B/C 指标和最终报告，固化复现环境并更新项目进展。
"""
if new in text:
    print("plan_progress_already_updated=true")
elif old in text:
    path.write_text(text.replace(old, new, 1), encoding="utf-8")
    print("plan_progress_updated=true")
else:
    raise SystemExit("plan progress block not found")
