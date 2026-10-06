#!/usr/bin/env bash
# Single-node (non-Slurm) launcher for the refusal-SFT recipe on Qwen3-VL-8B.
# Same training path as scripts/sft_local.sh (torchrun qwenvl/train/train_qwen.py,
# --training_mode sft), with a short fixed-step schedule and stricter argument handling:
# all three positional arguments are required, and an output_dir that already contains
# checkpoint-* is rejected. Slurm wrapper: recipes/refusal/sft_refusal.sbatch.
#
# Usage (run from sft/, with the SFT env's bin/ first on PATH):
#   GPUS=8 bash recipes/refusal/sft_refusal.sh <meta.json> <output_dir> <base_model>
#     <meta.json>   refusal meta file (e.g. recipes/refusal/meta_refusal.json)
#     <output_dir>  new directory; must not contain checkpoint-*
#     <base_model>  model dir to fine-tune; its path must contain "qwen3" and "vl"
#
# Requires transformers>=5 (Qwen3-VL and its native packed-sequence support);
# see sft/requirements.lock.txt for the tested stack.
#
# Optional env vars (default):
#   MAX_STEPS (75), SAVE_STEPS (25), LOGGING_STEPS (25), RUN_NAME (q3vl8b-refusal)
#   GPUS (4), NPROC_PER_NODE (=GPUS), NNODES (1), NODE_RANK (0),
#   MASTER_ADDR (127.0.0.1), MASTER_PORT (31520)
#   CUDA_HOME (/usr/local/cuda-13.2), PYTORCH_CUDA_ALLOC_CONF (expandable_segments:True)
#   MAX_GRAD_NORM (5), GRAD_NORM_PROBE (1), GRAD_NORM_SKIP_MULT (0), GRAD_NORM_SKIP_WARMUP (50)
#   PACK_CARRY (0), PACK_CARRY_LIMIT (64)

set -x

cleanup() {
  pkill -P $$
}
for sig in INT QUIT HUP TERM; do
  trap "cleanup; trap - \$sig EXIT; kill -s \$sig \"$$\"" "$sig"
done
trap cleanup EXIT

export TRITON_CACHE_DIR="/tmp/triton/"
# Blackwell (sm_103a): the ptxas bundled with CUDA 13 torch wheels already supports sm_103a,
# so TRITON_PTXAS_PATH is not forced. If Triton fails with a ptxas / sm_103a error, point
# TRITON_PTXAS_PATH at a CUDA>=12.9 ptxas.
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}

# Local single-node defaults; override via env vars.
# ZeRO-3 shards optimizer state per rank: resuming from a checkpoint requires the same
# number of GPUs it was saved with.
GPUS=${GPUS:-4}
NPROC_PER_NODE=${NPROC_PER_NODE:-${GPUS}}
NNODES=${NNODES:-1}
NODE_RANK=${NODE_RANK:-0}
MASTER_ADDR=${MASTER_ADDR:-127.0.0.1}
MASTER_PORT=${MASTER_PORT:-31520}

# Training meta. For data whose actions are Qwen <tool_call> JSON with coordinates in
# thousandths ([0,1000]), keep --coord_norm True.
# All three arguments are required; there are no fallback defaults.
datasets=${1:?ERROR: argument 1 (meta json) is required; refusing to fall back to a default dataset.}
output_dir=${2:?ERROR: argument 2 (output dir) is required; refusing to fall back to a default path.}
# The model passed here is loaded as weights only (no optimizer / LR schedule / data position).
pretrained=${3:?ERROR: argument 3 (base weights) is required; refusing to fall back to the untrained Instruct base.}

# Refuse an output_dir that already has checkpoints: train_qwen.py silently resumes from
# any checkpoint-* it finds there, which would continue someone else's run and overwrite
# its outputs instead of starting a fresh one.
if compgen -G "${output_dir}/checkpoint-*" > /dev/null 2>&1; then
  echo "ERROR: ${output_dir} already contains checkpoint-*; refusing to start (prevents overwriting / silently resuming a run)."
  exit 1
fi
mkdir -p "${output_dir}"

export CUDA_HOME=${CUDA_HOME:-/usr/local/cuda-13.2}
export PATH="${CUDA_HOME}/bin:${PATH}"
export PYTHONPATH="${PYTHONPATH}:$(pwd)"

# Sanity checks.
if [ ! -s "${datasets}" ]; then
  echo "ERROR: ${datasets} is missing or empty."
  exit 1
fi
if [ ! -d "${pretrained}" ]; then
  echo "ERROR: pretrained model dir '${pretrained}' not found. Download Qwen3-VL-8B first."
  exit 1
fi

# DeepSpeed configuration
deepspeed=./scripts/zero3.json

# Training hyperparameters.
#
# Packing (--data_flatten True): each per-device batch is packed into one sequence of at
# most --model_max_length tokens. If a batch overflows, FlattenedDataCollator in
# qwenvl/data/data_qwen.py drops whole trailing samples, not just extra tokens
# (PACK_CARRY=0), or defers them to the next batch (PACK_CARRY=1); see PACK_CARRY below.
# Keep batch_size x typical sample length well under model_max_length so overflow stays rare.
# Screenshots are not downsampled (no --max_pixels; the DataArguments default is 16777216),
# so single samples with large images can be several thousand tokens long.
#
# --group_sampling groups batches by modality (multimodal vs. text-only), not by length.
# On an all-multimodal dataset it reduces to i.i.d. random sampling.
#
# dataloader_num_workers 4 / prefetch_factor 1 bound the host RAM held by in-flight
# decoded images (pixel_values of a batch of large screenshots can reach GBs per rank).
#
# Global batch = batch_size * grad_accum_steps * GPUS.
lr=1e-6
# Fixed-step schedule: the run length is set by max_steps, not epochs.
max_steps="${MAX_STEPS:-75}"
save_steps="${SAVE_STEPS:-25}"
logging_steps="${LOGGING_STEPS:-25}"
echo "max_steps=${max_steps} save_steps=${save_steps}"
# GPU memory is dominated by the LM-head loss, not activations: ForCausalLMLoss upcasts the
# logits to fp32 unconditionally (about 2 x tokens x 151,936 x 4 bytes per rank). To raise
# batch_size, switch to a chunked / fused cross-entropy (e.g. Liger) first.
batch_size=8
grad_accum_steps=1
echo "grad_accum_steps=${grad_accum_steps}"

# Clipping only has an effect when the threshold lies inside the grad_norm distribution.
# If every step is clipped, each step is rescaled to the same norm; Adam's update is
# invariant to a global gradient scale, so the threshold value no longer matters and
# spikes are not singled out. Early SFT steps typically show grad_norm around 40-55, so
# the default of 5 clips every step. grad_norm is typically a warmup transient (large early,
# much smaller later), so do not calibrate on early-step percentiles; choose a value above
# the early peak. Use GRAD_NORM_PROBE=1 to see the actual distribution.
max_grad_norm="${MAX_GRAD_NORM:-5}"
echo "max_grad_norm=${max_grad_norm}"

# Gradient-norm diagnostics / step skipping, implemented in qwenvl/train/trainer.py (the
# data-source field comes from qwenvl/data/data_qwen.py). Both read environment variables,
# so they must be exported to reach the processes torchrun spawns. Override from Slurm with
#   sbatch --export=ALL,GRAD_NORM_SKIP_MULT=5 scripts/sft_8b_slurm.sbatch
# (--export must include ALL, otherwise the job environment is replaced and PATH is lost).
#
# GRAD_NORM_PROBE=1 only logs, one line per optimizer step:
#   [gn-probe] step=.. grad_norm=.. maxlen=.. tokens=.. spread=.. nsample=.. src=..
# It costs one extra all_gather_object per step (tens of ms). grad_norm is the norm over
# trainable parameters only, so freezing or unfreezing a tower shifts the distribution;
# re-check MAX_GRAD_NORM against the probe output after such changes.
GRAD_NORM_PROBE="${GRAD_NORM_PROBE:-1}"

# GRAD_NORM_SKIP_MULT>0 skips any step whose grad_norm exceeds (median of the last 200
# steps) x MULT, via DeepSpeed's overflow path (optimizer.step() is not called). 0 = off.
# Leave it off until the probe shows spikes are random: if they come from a particular data
# subset, skipping systematically drops the hardest samples.
# GRAD_NORM_SKIP_WARMUP = number of steps of history collected before skipping can trigger.
GRAD_NORM_SKIP_MULT="${GRAD_NORM_SKIP_MULT:-0}"
GRAD_NORM_SKIP_WARMUP="${GRAD_NORM_SKIP_WARMUP:-50}"
export GRAD_NORM_PROBE GRAD_NORM_SKIP_MULT GRAD_NORM_SKIP_WARMUP
echo "GRAD_NORM_PROBE=${GRAD_NORM_PROBE} GRAD_NORM_SKIP_MULT=${GRAD_NORM_SKIP_MULT} GRAD_NORM_SKIP_WARMUP=${GRAD_NORM_SKIP_WARMUP}"

# Packing overflow: 1 = defer overflowing samples to the next batch; 0 = drop them.
# Read at import time by FlattenedDataCollator in qwenvl/data/data_qwen.py, so it must be
# exported. Log lines: "[pack] ... deferring N to the next batch" (1) or
# "Token indices sequence length is longer than ... Truncating to ... with N samples" (0).
# The default here is 0 with a batch_size small enough that overflow is rare; grep the log
# for "Truncating" to check the actual drop rate, and lower batch_size or set PACK_CARRY=1
# if it is high. Deferred samples are queued per dataloader worker and workers are rebuilt
# each epoch, so up to dataloader_num_workers samples can still be lost per epoch.
# PACK_CARRY_LIMIT caps the queue length (a safeguard against a cap far below sample length).
PACK_CARRY="${PACK_CARRY:-0}"
PACK_CARRY_LIMIT="${PACK_CARRY_LIMIT:-64}"
export PACK_CARRY PACK_CARRY_LIMIT
echo "PACK_CARRY=${PACK_CARRY} PACK_CARRY_LIMIT=${PACK_CARRY_LIMIT}"

# Training entry point
entry_file=qwenvl/train/train_qwen.py

# Output configuration
run_name="${RUN_NAME:-q3vl8b-refusal}"

echo "Launching torchrun: GPUS=${GPUS}, NNODES=${NNODES}, NPROC_PER_NODE=${NPROC_PER_NODE}, MASTER_ADDR=${MASTER_ADDR}, MASTER_PORT=${MASTER_PORT}, META=${datasets}, OUTPUT=${output_dir}, PRETRAINED=${pretrained}"

# Training arguments. Notes:
#   --training_mode sft: loss only on the assistant answer. The user turn and all visual
#     tokens are always masked; --system_loss_ratio 0.0 also masks the system header.
#   --average_tokens_across_devices True: normalise the loss by the global token count so
#     every token has equal weight. With False each rank averages over its own tokens and
#     ranks are then weighted equally, which up-weights tokens on ranks that received fewer
#     tokens (per-rank token counts differ under i.i.d. sampling; the probe's spread= field
#     is the max/min ratio across ranks). The choice changes the absolute loss value.
#   --tune_mm_vision False / --tune_mm_mlp False: only the language model is trained.
#   --coord_norm True: coordinates are thousandths in [0,1000].
#   --save_only_model True: checkpoints hold weights only (no optimizer / scheduler state),
#     so they are small but cannot be used to resume this run.
args="
    --deepspeed ${deepspeed} \
    --model_name_or_path ${pretrained} \
    --data_path ${datasets} \
    --data_flatten True \
    --training_mode sft \
    --system_loss_ratio 0.0 \
    --group_sampling True \
    --tune_mm_vision False \
    --tune_mm_mlp False \
    --tune_mm_llm True \
    --bf16 \
    --coord_norm True \
    --output_dir ${output_dir} \
    --max_steps ${max_steps} \
    --per_device_train_batch_size ${batch_size} \
    --per_device_eval_batch_size $((batch_size*2)) \
    --gradient_accumulation_steps ${grad_accum_steps} \
    --eval_strategy no \
    --save_strategy steps \
    --save_steps ${save_steps} \
    --save_only_model True \
    --save_total_limit 10 \
    --learning_rate ${lr} \
    --weight_decay 0.001 \
    --warmup_steps 5 \
    --max_grad_norm ${max_grad_norm} \
    --lr_scheduler_type cosine \
    --logging_steps ${logging_steps} \
    --model_max_length 73728 \
    --gradient_checkpointing True \
    --dataloader_num_workers 4 \
    --dataloader_persistent_workers False \
    --dataloader_prefetch_factor 1 \
    --average_tokens_across_devices True \
    --run_name ${run_name} \
    --report_to none"

# Launch training
torchrun --nproc_per_node=${NPROC_PER_NODE} \
         --master_addr=${MASTER_ADDR} \
         --master_port=${MASTER_PORT} \
         --node_rank ${NODE_RANK} \
         --nnodes ${NNODES} \
         ${entry_file} ${args} \
         2>&1 | tee "${output_dir}/training_log.txt"
