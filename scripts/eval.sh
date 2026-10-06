#!/usr/bin/env bash
# Evaluate one checkpoint on the five grounding benchmarks (eval/), one after another.
#
# Usage:
#   ENV_PREFIX=<eval env> DATA_ROOT=<benchmark root> scripts/eval.sh <checkpoint> [output_dir]
#     checkpoint  an RL checkpoint (used as is) or an SFT / refusal-SFT checkpoint saved by
#                 transformers 5 (converted into <output_dir>/serve first by
#                 tools/build_serve_dir.sh; needs INSTRUCT_DIR=<Qwen3-VL-8B-Instruct dir>)
#     output_dir  default <checkpoint>-eval; results go to <output_dir>/<benchmark>/
#
# Env: BENCHMARKS ("screenspot-v2 screenspot-pro mmbench-gui ui-vision osworld-g"),
#      PROMPT (scalecua_toolcall), OSWG_PROMPT (scalecua_toolcall_abstain_schema), TP (2), and
#      everything eval/run_eval.sh reads. Scoring is GROUNDING_OFFICIAL_SCORING=0 unless set.
# DATA_ROOT is only needed the first time, to render eval/dataset_info.json from its template.
set -euo pipefail
source "$(dirname "$0")/_common.sh"

[ "$#" -ge 1 ] || { sed -n '2,15p' "$0"; exit 2; }
CKPT="$(abspath "$1")"; OUT="$(abspath "${2:-${CKPT%/}-eval}")"
need ENV_PREFIX "<prefix of the eval env, see eval/requirements.lock.txt>"
[ -f "$CKPT/config.json" ] || die "not a checkpoint directory: $CKPT"

if [ ! -f "$REPO/eval/dataset_info.json" ]; then
  need DATA_ROOT "<directory that holds the benchmarks>"
  render_meta "$REPO/eval/dataset_info.template.json" "$REPO/eval/dataset_info.json"
fi

MODEL_DIR="$CKPT"
if [ "$(ckpt_format "$CKPT")" = tf5 ]; then
  need INSTRUCT_DIR "<Qwen3-VL-8B-Instruct dir> (tokenizer template for the serve dir)"
  MODEL_DIR="$OUT/serve"
  [ -f "$MODEL_DIR/preprocessor_config.json" ] || bash "$REPO/tools/build_serve_dir.sh" "$MODEL_DIR" "$CKPT" "$INSTRUCT_DIR"
fi

cd "$REPO/eval"
for b in ${BENCHMARKS:-screenspot-v2 screenspot-pro mmbench-gui ui-vision osworld-g}; do
  p="${PROMPT:-scalecua_toolcall}"
  case "$b" in osworld-g*) p="${OSWG_PROMPT:-scalecua_toolcall_abstain_schema}" ;; esac
  mkdir -p "$OUT/$b"
  echo "=== $b  (prompt $p) ==="
  TP="${TP:-2}" BENCHMARK="$b" OUTPUT_DIR="$OUT/$b" bash run_eval.sh "$MODEL_DIR" "$p"
done
echo "results: $OUT/<benchmark>/"
