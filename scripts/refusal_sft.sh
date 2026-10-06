#!/usr/bin/env bash
# Refusal SFT on an RL checkpoint, on one node (sft/recipes/refusal/). The paper's numbers are
# measured after this step.
#
# Usage:
#   CONDA_ENV=<sft env> DATA_ROOT=<data root> scripts/refusal_sft.sh <rl_checkpoint> <output_dir>
#     rl_checkpoint  e.g. rl/output/opsd+grpo/<run>/checkpoint-200
#     output_dir     new directory; must not contain checkpoint-*
#
# 75-step schedule, a checkpoint every 25 steps; the paper uses checkpoint-50 (the D2RL row used
# MAX_STEPS=125). Env: GPUS (8), MAX_STEPS (75), SAVE_STEPS (25), META
# (sft/recipes/refusal/meta_refusal.json), and everything sft/recipes/refusal/sft_refusal.sh reads.
set -euo pipefail
source "$(dirname "$0")/_common.sh"

[ "$#" -eq 2 ] || { sed -n '2,13p' "$0"; exit 2; }
CKPT="$(abspath "$1")"; OUT="$(abspath "$2")"
need CONDA_ENV "<prefix of the sft env, see sft/requirements.lock.txt>"
[ -f "$CKPT/config.json" ] || die "not a checkpoint directory: $CKPT"

# starting directory: symlinks to the RL checkpoint; the name must contain "qwen3" and "vl"
BASE="${OUT%/}.base/Qwen3-VL-8B-base"
[ -e "$BASE" ] || bash "$REPO/tools/make_sft_base.sh" "$CKPT" "$BASE"

mkdir -p "$OUT"
render_meta "${META:-$REPO/sft/recipes/refusal/meta_refusal.json}" "$OUT/meta.rendered.json"
export PATH="$CONDA_ENV/bin:$PATH" PYTHONNOUSERSITE=1
cd "$REPO/sft"
GPUS="${GPUS:-8}" bash recipes/refusal/sft_refusal.sh "$OUT/meta.rendered.json" "$OUT" "$BASE"
