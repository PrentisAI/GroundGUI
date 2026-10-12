#!/usr/bin/env bash
# Supervised fine-tuning on one node (sft/).
#
# Usage:
#   CONDA_ENV=<sft env> DATA_ROOT=<data root> scripts/sft.sh <output_dir> <base_model> [meta.json]
#     output_dir  where checkpoints are written (an existing checkpoint-* is resumed)
#     base_model  model to fine-tune, e.g. a Qwen3-VL-8B-Instruct directory
#     meta.json   data meta (default: $DATA_ROOT/meta/meta_non_refusal.json when DATA_ROOT is
#                 ScreenRef-Annotations, else sft/configs/train_data_sft.json); ${DATA_ROOT} in it is
#                 substituted, and relative paths are taken under DATA_ROOT, into
#                 <output_dir>/meta.rendered.json first
#
# ScreenRef: DATA_ROOT=<ScreenRef-Annotations dir>, SCREENREF_PARQUET=<downloaded ScreenRef dir>;
# the screenshots are unpacked once into $DATA_ROOT/images/ before training (~330 GB).
#
# Env: GPUS (8), and everything sft/scripts/sft_local.sh reads.
# Output: <output_dir>/checkpoint-*  (transformers-5 format; scripts/rloo.sh, opsd.sh and
# eval.sh accept it directly and convert it themselves).
set -euo pipefail
source "$(dirname "$0")/_common.sh"

[ "$#" -ge 2 ] || { sed -n '2,18p' "$0"; exit 2; }
OUT="$(abspath "$1")"; BASE="$(abspath "$2")"
if [ -n "${3:-}" ]; then META="$(abspath "$3")"
elif [ -f "${DATA_ROOT:-/nonexistent}/meta/meta_non_refusal.json" ]; then META="$(abspath "$DATA_ROOT")/meta/meta_non_refusal.json"
else META="$REPO/sft/configs/train_data_sft.json"; fi
need CONDA_ENV "<prefix of the sft env, see sft/requirements.lock.txt>"
[ -d "$BASE" ] || die "base model not found: $BASE"

mkdir -p "$OUT"
screenref_unpack
render_meta "$META" "$OUT/meta.rendered.json"
export PATH="$CONDA_ENV/bin:$PATH" PYTHONNOUSERSITE=1
cd "$REPO/sft"
GPUS="${GPUS:-8}" bash scripts/sft_local.sh "$OUT/meta.rendered.json" "$OUT" "$BASE"
