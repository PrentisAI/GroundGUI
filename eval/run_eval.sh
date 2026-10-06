#!/usr/bin/env bash
# Single-node grounding evaluation driver (vLLM engine) for any benchmark in
# dataset_info.json (default: OSWorld-G).
#
# Usage:
#   ENV_PREFIX=/path/to/eval_env bash run_eval.sh <model_path> [prompt]
#   ENV_PREFIX=... TP=2 bash run_eval.sh /path/to/model scalecua_toolcall_abstain_schema
#   ENV_PREFIX=... BENCHMARK=osworld-g-refined bash run_eval.sh /path/to/model guiowl
#
# Env vars: ENV_PREFIX (required), BENCHMARK, TP, BATCH_SIZE, CACHE_ROOT, OUTPUT_DIR, DEBUG,
#   IMAGE_FACTOR, MAX_IMAGE_PIXELS, COORD_SPACE, GROUNDING_OFFICIAL_SCORING.
#
# On OSWorld-G the prompt decides whether the 54 refusal samples (out of 564) can be
# answered correctly at all:
#   scalecua_toolcall                 no abstain action -> ceiling 510/564 = 90.4%
#   scalecua_toolcall_abstain_schema  for checkpoints trained with the abstain action
#   scalecua_toolcall_refusal_schema  for checkpoints trained to refuse via terminate(status="failure")
#   guiowl / qwen3vl / qwen3vl_point  external Qwen3-VL models
set -euo pipefail

ENV_PREFIX="${ENV_PREFIX:?set ENV_PREFIX=<prefix of the eval env, see eval/requirements.lock.txt>}"
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

MODEL_PATH="${1:?usage: run_eval.sh <model_path> [prompt]}"
PROMPT="${2:-scalecua_toolcall}"
BENCHMARK="${BENCHMARK:-osworld-g}"   # e.g. osworld-g-refined
TP="${TP:-1}"
BATCH_SIZE="${BATCH_SIZE:-8}"

if [ ! -e "$MODEL_PATH" ]; then
    echo "Error: model not found at $MODEL_PATH" >&2; exit 1
fi

# ---- Cache redirection ------------------------------------------------------
# Every cache is redirected explicitly; FlashInfer only honours FLASHINFER_WORKSPACE_BASE,
# not XDG_CACHE_HOME.
CACHE_ROOT="${CACHE_ROOT:-${XDG_CACHE_HOME:-$HOME/.cache}/gcua_eval}"
export XDG_CACHE_HOME="${CACHE_ROOT}/xdg"
export HF_HOME="${CACHE_ROOT}/hf"
export HUGGINGFACE_HUB_CACHE="${HF_HOME}/hub"
export TRITON_CACHE_DIR="${CACHE_ROOT}/triton"
export MPLCONFIGDIR="${CACHE_ROOT}/mpl"
export VLLM_CACHE_ROOT="${CACHE_ROOT}/vllm"
export TORCHINDUCTOR_CACHE_DIR="${CACHE_ROOT}/inductor"
export FLASHINFER_WORKSPACE_BASE="${CACHE_ROOT}/flashinfer"
mkdir -p "$XDG_CACHE_HOME" "$HF_HOME" "$TRITON_CACHE_DIR" "$MPLCONFIGDIR" \
         "$VLLM_CACHE_ROOT" "$TORCHINDUCTOR_CACHE_DIR" "$FLASHINFER_WORKSPACE_BASE"

# Keep user site-packages (~/.local) from shadowing the env's packages.
export PYTHONNOUSERSITE=1

# ---- Coordinate space / scoring ---------------------------------------------
# smart_resize factor. Qwen3-VL: patch 16 * merge 2 = 32; use 28 for Qwen2.5-VL models.
export IMAGE_FACTOR="${IMAGE_FACTOR:-32}"
export MAX_IMAGE_PIXELS="${MAX_IMAGE_PIXELS:-16777216}"
# The scalecua_toolcall / guiowl prompts already map 0-1000 outputs back to pixels in
# calculate_metrics; setting COORD_SPACE=norm1000 on top rescales twice and accuracy drops
# to ~0. Leave it empty for these prompts.
export COORD_SPACE="${COORD_SPACE:-}"
# 0 = exact containment in pixel space
# 1 = official-style: GT box scaled to 0-1000 space and rounded outward (looser; matches the
#     numbers reported by GUI-Owl / UI-Venus). The mode is NOT recorded in result files;
#     use rescore.py to re-score existing predictions under the other mode.
export GROUNDING_OFFICIAL_SCORING="${GROUNDING_OFFICIAL_SCORING:-0}"

# vLLM backend: prebuilt flash-attn kernels instead of FlashInfer JIT.
export VLLM_ATTN_BACKEND="${VLLM_ATTN_BACKEND:-FLASH_ATTN}"
export VLLM_USE_FLASHINFER_SAMPLER="${VLLM_USE_FLASHINFER_SAMPLER:-0}"

echo "=========================================="
echo "OSWorld-G evaluation"
echo "  model      : $MODEL_PATH"
echo "  prompt     : $PROMPT"
echo "  TP         : $TP   batch: $BATCH_SIZE"
echo "  env        : $ENV_PREFIX"
echo "  IMAGE_FACTOR=$IMAGE_FACTOR  MAX_IMAGE_PIXELS=$MAX_IMAGE_PIXELS"
echo "  COORD_SPACE='${COORD_SPACE}'  OFFICIAL_SCORING=$GROUNDING_OFFICIAL_SCORING"
echo "  CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"
echo "=========================================="

cd "$SCRIPT_DIR"

# OUTPUT_DIR: write results elsewhere (e.g. when ./output is not writable by the job user).
# Empty = eval.py default ./output/<model>/<benchmark>/.
OUT_ARGS=()
if [ -n "${OUTPUT_DIR:-}" ]; then
    OUT_ARGS=(--output-dir "$OUTPUT_DIR")
    echo "  output-dir : $OUTPUT_DIR"
fi

"${ENV_PREFIX}/bin/python" eval.py \
    "$MODEL_PATH" \
    "${OUT_ARGS[@]}" \
    --benchmark "$BENCHMARK" \
    --prompt "$PROMPT" \
    --engine vllm \
    --tensor-parallel "$TP" \
    --batch-size "$BATCH_SIZE" \
    --temperature 0.0 \
    --debug-mode "${DEBUG:-0}" \
    --no-cache

echo "Results: ${SCRIPT_DIR}/output/<model_name>/<benchmark>/ (or \$OUTPUT_DIR)"
