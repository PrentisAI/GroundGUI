#!/usr/bin/env bash
# Refusal SFT on an RL checkpoint, on one node (sft/scripts/sft_post_rl.sh). The paper's numbers are
# measured after this step.
#
# Usage:
#   CONDA_ENV=<sft env> DATA_ROOT=<data root> scripts/refusal_sft.sh <rl_checkpoint> <output_dir>
#     rl_checkpoint  e.g. rl/output/opsd_rloo/<run>/checkpoint-N
#     output_dir     new directory; must not contain checkpoint-*
#
# 75-step schedule, a checkpoint every 25 steps; the paper uses checkpoint-50 (the D2RL row used
# MAX_STEPS=125). Env: GPUS (8), MAX_STEPS (75), SAVE_STEPS (25), META ($DATA_ROOT/meta/meta_refusal.json
# when DATA_ROOT is ScreenRef-Annotations, else sft/configs/train_data_post_rl.json),
# SCREENREF_PARQUET (as in scripts/sft.sh), and everything sft/scripts/sft_post_rl.sh reads.
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
screenref_unpack
if [ -z "${META:-}" ]; then
  if [ -f "${DATA_ROOT:-/nonexistent}/meta/meta_refusal.json" ]; then META="$(abspath "$DATA_ROOT")/meta/meta_refusal.json"
  else META="$REPO/sft/configs/train_data_post_rl.json"; fi
fi
render_meta "$META" "$OUT/meta.rendered.json"
export PATH="$CONDA_ENV/bin:$PATH" PYTHONNOUSERSITE=1
cd "$REPO/sft"
GPUS="${GPUS:-8}" bash scripts/sft_post_rl.sh "$OUT/meta.rendered.json" "$OUT" "$BASE"
