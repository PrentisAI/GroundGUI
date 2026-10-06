#!/usr/bin/env bash
# Supervised fine-tuning on one node (sft/).
#
# Usage:
#   CONDA_ENV=<sft env> DATA_ROOT=<data root> scripts/sft.sh <output_dir> <base_model> [meta.json]
#     output_dir  where checkpoints are written (an existing checkpoint-* is resumed)
#     base_model  model to fine-tune, e.g. a Qwen3-VL-8B-Instruct directory
#     meta.json   data meta (default: sft/configs/train_data_sft.json); ${DATA_ROOT} in it is
#                 substituted into <output_dir>/meta.rendered.json first
#
# Env: GPUS (8), and everything sft/scripts/sft_local.sh reads.
# Output: <output_dir>/checkpoint-*  (transformers-5 format; scripts/rloo.sh, opsd.sh and
# eval.sh accept it directly and convert it themselves).
set -euo pipefail
source "$(dirname "$0")/_common.sh"

[ "$#" -ge 2 ] || { sed -n '2,13p' "$0"; exit 2; }
OUT="$(abspath "$1")"; BASE="$(abspath "$2")"; META="$(abspath "${3:-$REPO/sft/configs/train_data_sft.json}")"
need CONDA_ENV "<prefix of the sft env, see sft/requirements.lock.txt>"
[ -d "$BASE" ] || die "base model not found: $BASE"

mkdir -p "$OUT"
render_meta "$META" "$OUT/meta.rendered.json"
export PATH="$CONDA_ENV/bin:$PATH" PYTHONNOUSERSITE=1
cd "$REPO/sft"
GPUS="${GPUS:-8}" bash scripts/sft_local.sh "$OUT/meta.rendered.json" "$OUT" "$BASE"
