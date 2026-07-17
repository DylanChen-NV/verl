#!/usr/bin/env bash
set -euo pipefail
root=/lustre/fs1/portfolios/coreai/projects/coreai_devtech_all/users/ziqingc/05_claude_ws/dynamic_mooncake_abort_dev/scripts
for name in \
  apply_vllm_mooncake_abort_reuse.py \
  patch_prepared_repo_abort_barrier.py.sh \
  run_persistent_moon_c_cross_node_precheck.sh
do
  gzip -dc "${root}/${name}.gz" > "${root}/${name}"
done
chmod +x "${root}"/*.sh "${root}"/*.py
python3 -S - <<'PY'
from pathlib import Path

path = Path("/lustre/fs1/portfolios/coreai/projects/coreai_devtech_all/users/ziqingc/05_claude_ws/dynamic_mooncake_abort_dev/scripts/dfw_h100_dynamic_recompute_30b_r8k_req32_mooncake_abc.sh")
text = path.read_text()
replacements = {
    "export TRAINER_PPO_MICRO_BSZ=1 TRAINER_SAVE_FREQ=-1 TRAIN_PROMPT_MINI_BSZ=8 REQUIRE_BATCHES=2\nexport PLANNED_TRAINER_STEPS=5 TOTAL_ROLLOUT_STEPS=80": "export TRAINER_PPO_MICRO_BSZ=${TRAINER_PPO_MICRO_BSZ:-1} TRAINER_SAVE_FREQ=${TRAINER_SAVE_FREQ:--1}\nexport TRAIN_PROMPT_MINI_BSZ=${TRAIN_PROMPT_MINI_BSZ:-8} REQUIRE_BATCHES=${REQUIRE_BATCHES:-2}\nexport PLANNED_TRAINER_STEPS=${PLANNED_TRAINER_STEPS:-5} TOTAL_ROLLOUT_STEPS=${TOTAL_ROLLOUT_STEPS:-80}",
    "export N_RESP_PER_PROMPT=2 CONCURRENT_SAMPLES_PER_REPLICA=32 ROLLOUT_MAX_NUM_SEQS=32 ROLLOUT_TP=2": "export N_RESP_PER_PROMPT=${N_RESP_PER_PROMPT:-2}\nexport CONCURRENT_SAMPLES_PER_REPLICA=${CONCURRENT_SAMPLES_PER_REPLICA:-32}\nexport ROLLOUT_MAX_NUM_SEQS=${ROLLOUT_MAX_NUM_SEQS:-32} ROLLOUT_TP=${ROLLOUT_TP:-2}",
    "export DYNAMIC_DEACTIVATE_RATIO=0.25 VERL_RECOMPUTE_DEACTIVATE_GRACE_S=0": "export DYNAMIC_DEACTIVATE_RATIO=${DYNAMIC_DEACTIVATE_RATIO:-0.25}\nexport VERL_RECOMPUTE_DEACTIVATE_GRACE_S=${VERL_RECOMPUTE_DEACTIVATE_GRACE_S:-0}",
}
changed = False
for old, new in replacements.items():
    if old in text:
        text = text.replace(old, new, 1)
        changed = True
    elif new not in text:
        raise SystemExit(f"runner override target not found: {old!r}")
if changed:
    path.write_text(text)
print(f"runner_override_patch={changed}")
PY
"${root}/patch_prepared_repo_abort_barrier.py.sh"
