#!/bin/bash
# Grounding RL launcher: GRPO / RLOO, optionally with the OPSD+GRPO branch.
#
# Chain:  submit.sh <arm> -> train_opsd_grpo.sbatch -> train_opsd_grpo.sh -> train_grpo.sh (this)
#
# Topology-neutral core. On one node it starts a 1-GPU vLLM rollout server (`swift rollout`,
# TP=1) and a DDP + ZeRO-2 trainer (`swift rlhf --rlhf_type grpo`) on the remaining GPUs.
# Generation is a small fraction of a step, so one rollout GPU is enough. Normally called by
# train_opsd_grpo.sh; to run it directly, set the GPU split yourself:
#
#   ENV_BIN=<rl env>/bin GPU_ROLLOUT=7 GPUS_TRAINER=0,1,2,3,4,5,6 \
#       bash Grounding_scripts/train_grpo.sh
#
# Required: ENV_BIN (bin/ of the RL env, see rl/requirements.lock.txt).
# Optional (defaults below): MODEL DATASET OUT_DIR ADD_VERSION RUN_TAG LOGDIR, GPU_ROLLOUT
#   GPUS_TRAINER PORT MASTER_PORT DL_WORKERS, NUM_GEN BS GAS LR EPOCHS TEMPERATURE BETA
#   MAX_GRAD_NORM GRAD_CKPT, SCALE_REWARDS (group = GRPO, none = RLOO-style), REWARD_FUNCS
#   REWARD_WEIGHTS DYNAMIC_SAMPLE, MEASURE_MODE MAX_STEPS SAVE_STEPS SAVE_ONLY_MODEL RESUME_FROM,
#   REPORT_TO SWANLAB_MODE_ARG, USE_OPSD_GRPO + NZG_*/TEACHER_*/OPSD_*.
#
# CORRECTNESS -- each of these fails silently if wrong:
# 1. The rollout server MUST run with --vllm_enforce_eager true. It is a separate process, so
#    the flag is not in the trainer's args.json, and it changes the answers, not just the
#    speed: outside vLLM CompilationMode 1 the Dynamo guards are dropped, which freezes
#    Qwen3-VL's DeepStack branch off and drops grounding accuracy from 60.0% to 7.85%. Every
#    reward would be computed on that broken model and nearly every advantage would be ~0.
#    Check rewards/GroundAcc/mean at step 1: it must be near the init's accuracy, not ~0.1.
# 2. The weight-sync fence in swift/rlhf_trainers/utils.py must be present (asserted below);
#    without it the rollout server serves a partly updated model.
# 3. The rewards read the dataset's `solution` and `additional_paras` columns, passed through
#    RowPreprocessor.rows_to_batched (_init_grpo forces remove_unused_columns=False).
#
# REWARDS (swift/cus_rewards/):
#   ground-acc   binary   1.0 iff the predicted point is inside the GT bbox
#   g2acc        dense    GUI-G^2 gaussian point reward, sigma = 0.5 * bbox side
#   bbox-reward  mixed    GUI-G1 style R_hit + 0.25*R_dist + 0.125*R_box
#   ground-format          1.0 iff a parseable <tool_call> was emitted
# Default here is `ground-acc g2acc` at 1.0 / 0.5 (the OPSD+GRPO wrapper uses binary only).
# Binary alone gives zero variance (no gradient) for groups that all land in-box; the gaussian
# still separates those. It does not rescue far misses (most sit >4 sigma out, score <1e-3).
# Every reward is logged per sample into completions.jsonl.
#
# Metrics to watch:
#   frac_reward_zero_std    fraction of groups with zero variance of the weighted reward.
#   rewards/GroundAcc/mean  in-box rate of the policy. At step 1 it should be close to the
#                           init's greedy accuracy on the corpus (sampled rollouts at T=1.0
#                           start a bit lower). ~0.10 means item 1 above; ~0.02 with
#                           GroundFormat at 1.0 means the WRONG COORDINATE SPACE -- run
#                           Grounding_scripts/preflight.py.
set -euo pipefail

REPO="$(cd "$(dirname "$(readlink -f "$0")")/.." && pwd)"
cd "$REPO"

run_name="groundcua-grpo"
# MODEL: the Qwen3-VL-8B SFT checkpoint as prepared by Grounding_scripts/prepare_ckpt.py.
# Do not use the raw SFT dir: its transformers-5 config.json makes the trainer die under
# transformers 4.57.6 ('NoneType' object has no attribute 'get' in Qwen3VLTextRotaryEmbedding)
# while vLLM loads it fine, so only the trainer side fails.
MODEL="${MODEL:?set MODEL=<SFT checkpoint in transformers-4.57 format, see Grounding_scripts/prepare_ckpt.py>}"
ENV="${ENV_BIN:?set ENV_BIN=<rl env>/bin, see rl/requirements.lock.txt}"
# Log file names are keyed on run_name + RUN_TAG; give concurrent runs distinct RUN_TAGs.
LOGDIR="${LOGDIR:-$REPO/logs}"
mkdir -p "$LOGDIR"

# ---- environment ---------------------------------------------------------------
# PYTHONNOUSERSITE: a stale ~/.local transformers can shadow the env (the banner prints the
# resolved path). VLLM_USE_DEEP_GEMM=0: vLLM 0.22.1 enables it by default and an outdated
# deep_gemm fails engine startup; models are bf16. PATH: vLLM's cudagraph profiling needs `ninja`.
export PATH="$ENV:$PATH"
export PYTHONNOUSERSITE=1
export VLLM_USE_DEEP_GEMM=0

# ---- make THIS repo's swift the one Python imports ----------------------------
# An editable ms-swift install of another checkout would win over ./swift: the `swift` console
# script puts bin/ (not cwd) on sys.path[0]. PYTHONPATH wins because the editable finder is
# appended after PathFinder, and it is inherited by every subprocess and torchrun rank.
# Without it the run silently uses the other trainer, so the check below is fatal.
export PYTHONPATH="$REPO${PYTHONPATH:+:$PYTHONPATH}"
# Probe from an empty temp dir, not `python -c` in $REPO (which would find ./swift via cwd
# and pass even without PYTHONPATH); this matches the console script's import context.
_PROBE_DIR="$(mktemp -d)"
trap 'rm -rf "$_PROBE_DIR"' EXIT
cat > "$_PROBE_DIR/_swift_probe.py" <<'PYCHK'
import os, sys
repo = sys.argv[1]
import swift
got, want = os.path.realpath(swift.__path__[0]), os.path.realpath(os.path.join(repo, "swift"))
print(f"swift import resolves to: {got}")
sys.exit(0 if got == want else 1)
PYCHK
$ENV/python "$_PROBE_DIR/_swift_probe.py" "$REPO" || {
    echo "ERROR: the imported swift is NOT this repo's copy -- PYTHONPATH is not taking" >&2
    echo "       effect, so this run would silently use another swift install's trainer. Refusing." >&2
    exit 1; }
rm -rf "$_PROBE_DIR"; trap - EXIT

# DATASET: GroundCUA-style grounding rows. They must carry the internvl2_5_desktop_grounding_v1
# system prompt the init was SFT'd under, which asks for per-mille (0-1000) coordinates -- the
# unit the reward scores. Pass an ABSOLUTE path; relative paths resolve against cwd.
DATASET="${DATASET:?set DATASET=<absolute path of the RL corpus .jsonl>}"
OUT_DIR="${OUT_DIR:-output/${run_name}}"
ADD_VERSION="${ADD_VERSION:-true}"
RUN_TAG="${RUN_TAG:-}"
LOGSUF="${RUN_TAG:+-$RUN_TAG}"

# ---- MEASURE_MODE ----------------------------------------------------------
# true  -> bounded probe: MAX_STEPS steps, no checkpoints (to measure frac_reward_zero_std).
# false -> real training run.
MEASURE_MODE="${MEASURE_MODE:-false}"

# ---- RESUME ---------------------------------------------------------------------
# SAVE_ONLY_MODEL=true (default) saves weights only: checkpoints are NOT resumable.
# false also saves optimizer, scheduler and RNG state (global_step<N>/, ~+105 GB per 8B
# checkpoint). RESUME_FROM=<checkpoint dir>|auto (auto = highest-step full checkpoint under
# OUT_DIR) continues such a run with the same OUT_DIR and config. Weights-only checkpoints are
# refused: resuming from one would silently restart Adam and the LR schedule.
SAVE_ONLY_MODEL="${SAVE_ONLY_MODEL:-true}"
RESUME_FROM="${RESUME_FROM:-}"
if [ "$RESUME_FROM" = "auto" ]; then
    RESUME_FROM=$(for d in "$OUT_DIR"/checkpoint-*/; do
                      d="${d%/}"; ls -d "$d"/global_step* >/dev/null 2>&1 && echo "${d##*checkpoint-} $d"
                  done | sort -n | tail -1 | cut -d' ' -f2-)
    [ -n "$RESUME_FROM" ] || { echo "ERROR: RESUME_FROM=auto but no full checkpoint (global_step*/) under $OUT_DIR" >&2; exit 1; }
fi
if [ -n "$RESUME_FROM" ]; then
    [ -s "$RESUME_FROM/trainer_state.json" ] || { echo "ERROR: RESUME_FROM=$RESUME_FROM has no trainer_state.json" >&2; exit 1; }
    ls -d "$RESUME_FROM"/global_step* >/dev/null 2>&1 || { echo "ERROR: RESUME_FROM=$RESUME_FROM is weights-only (no global_step*/); cannot resume optimizer state" >&2; exit 1; }
    [ "$SAVE_ONLY_MODEL" = "false" ] || echo "WARN: resuming but SAVE_ONLY_MODEL=$SAVE_ONLY_MODEL -- new checkpoints will NOT be resumable" >&2
fi
MAX_STEPS="${MAX_STEPS:-60}"

# ---- reward -----------------------------------------------------------------
REWARD_FUNCS="${REWARD_FUNCS:-ground-acc g2acc}"
REWARD_WEIGHTS="${REWARD_WEIGHTS:-1.0 0.5}"

# ---- GRPO knobs -------------------------------------------------------------
# generation_batch_size = BS * n_trainer_gpus * GAS  (steps_per_generation defaults
# to GAS), and generation_batch_size / NUM_GEN = UNIQUE PROMPTS per optimizer step,
# which sets the epoch length (steps per epoch = corpus rows / prompts per step):
#
#   node   trainer GPUs   BS  GAS   completions   prompts/step
#      8              7    2    8           112             14   <- default
#      7              6    2    8            96             12
#      6              5    2    8            80             10
#
# Per-rank work is the same at every width, so extra GPUs buy shorter epochs, not faster steps.
NUM_GEN="${NUM_GEN:-8}"
BS="${BS:-2}"
GAS="${GAS:-8}"
# Gradient checkpointing trades trainer time for memory; GRAD_CKPT=false is faster but needs more.
GRAD_CKPT="${GRAD_CKPT:-true}"
LR="${LR:-1e-6}"
# Not a step-size knob under Adam (m/sqrt(v) is invariant to a uniform gradient rescale);
# raising it only damps spiky steps less. The step-size knob is LR.
MAX_GRAD_NORM="${MAX_GRAD_NORM:-1.0}"
EPOCHS="${EPOCHS:-1}"
TEMPERATURE="${TEMPERATURE:-1.0}"
# beta = KL to the reference policy. 0 builds NO reference model (DAPO/Dr.GRPO default),
# saving a second 8B model in memory.
BETA="${BETA:-0.0}"
# group = GRPO; none = no per-group std division (RLOO-style advantage).
SCALE_REWARDS="${SCALE_REWARDS:-group}"
# DAPO dynamic sampling: resample groups whose reward std is 0. Keep it OFF while
# measuring the zero-variance rate -- it would hide exactly that number.
DYNAMIC_SAMPLE="${DYNAMIC_SAMPLE:-false}"

# Dataloader workers PER RANK; they share CPUs with the rollout server's image decoding.
DL_WORKERS="${DL_WORKERS:-8}"

# Topology. Inside a Slurm step the device cgroup renumbers the job's GPUs 0..N-1, so these
# are CGROUP-LOCAL indices, not the physical ones scontrol prints. Launch inside the step.
GPU_ROLLOUT="${GPU_ROLLOUT:-7}"
GPUS_TRAINER="${GPUS_TRAINER:-0,1,2,3,4,5,6}"
NPROC=$(awk -F',' '{print NF}' <<< "$GPUS_TRAINER")
PORT="${PORT:-8199}"
MASTER_PORT="${MASTER_PORT:-29517}"

# ~17 GiB per checkpoint. The last step is always saved.
SAVE_STEPS="${SAVE_STEPS:-125}"

# tensorboard by default. swanlab in cloud mode needs a login and a TTY, and aborts a
# non-interactive job; add it with REPORT_TO="tensorboard swanlab" (local mode, see below).
REPORT_TO="${REPORT_TO:-tensorboard}"

# ---- OPSD+GRPO (RSTG, arXiv:2608.00782) -----------------------------------
# Trainer: swift/rlhf_trainers/opsd_grpo_trainer.py. USE_OPSD_GRPO=false (default) leaves
# OPSD_ARGS empty, i.e. plain GRPO. rlhf_type stays 'grpo': --use_opsd_grpo only selects the
# trainer class, so all GRPO setup is identical to a plain run.
#   NZG_BETA_PEAK/VALLEY: RSTG anneals 5e-3 -> 1e-3; its Table 2 shows a large or
#   constant beta degrades everything. Do not raise it.
#   REWARD_FUNCS must contain ground-acc: "all rollouts fail" is read off that binary column.
#   DYNAMIC_SAMPLE must stay false (discarding zero-variance groups is the competing fix).
USE_OPSD_GRPO="${USE_OPSD_GRPO:-false}"
NZG_BETA_PEAK="${NZG_BETA_PEAK:-0.005}"
NZG_BETA_VALLEY="${NZG_BETA_VALLEY:-0.001}"
NZG_WARMUP="${NZG_WARMUP:-0}"
NZG_DECAY="${NZG_DECAY:-0}"
NZG_ALL_ROLLOUTS="${NZG_ALL_ROLLOUTS:-false}"
NZG_ALLOW_MIXED="${NZG_ALLOW_MIXED:-false}"
NZG_LOG_ROUTING="${NZG_LOG_ROUTING:-true}"
NZG_OPD_BETA_PEAK="${NZG_OPD_BETA_PEAK:-0}"      # >0 turns the A_OPD ablation on
NZG_OPD_BETA_VALLEY="${NZG_OPD_BETA_VALLEY:-0}"
NZG_OPD_WARMUP="${NZG_OPD_WARMUP:-0}"
NZG_OPD_DECAY="${NZG_OPD_DECAY:-0}"
NZG_OPD_ALL_ROLLOUTS="${NZG_OPD_ALL_ROLLOUTS:-false}"
NZG_ROUTE_ALL="${NZG_ROUTE_ALL:-false}"          # pure-OPSD ablation: route EVERY group to the teacher
ALLOW_SINGLE_GEN="${ALLOW_SINGLE_GEN:-false}"    # K=1 pure-OPSD: lift the num_generations>=2 check (default false = unchanged)
TEACHER_DEEPSPEED="${TEACHER_DEEPSPEED:-zero2}"  # NOT zero3: a ZeRO-3 teacher forward all-gathers, and the OPD forward is rank-conditional -> hang
OPSD_MASK_MODE="${OPSD_MASK_MODE:-gaussian}"
# `hint`, not `hint_minimal`: hint_minimal stops marker words ("green rectangle") leaking into
# generated descriptions, but the target here is a bare coordinate tool call, and it would weaken
# the teacher signal. If the policy drifts into prose, watch nzg/format_match / nzg/unparsed.
OPSD_HINT_MODE="${OPSD_HINT_MODE:-hint}"
# OPD token weighting: position (default; coordinate digits weighted OPSD_POS_BETA * k_t,
# k_t = units 1 / tens 2 / hundreds 3) | uniform (all tokens 1.0).
OPSD_TOKEN_WEIGHT_MODE="${OPSD_TOKEN_WEIGHT_MODE:-position}"
OPSD_POS_BETA="${OPSD_POS_BETA:-1.0}"

# Per-run mask cache: with the shared default `cache/opd_cache`, concurrent runs would silently
# hand each other's masks to the teacher.
OPSD_MASK_DIR="${OPSD_MASK_DIR:-train_cache/opd-${RUN_TAG:-untagged}}"

# ---- swanlab mode --------------------------------------------------------------
# If swanlab is in REPORT_TO, force local mode: --swanlab_mode overrides SWANLAB_MODE, and cloud
# mode without a TTY dies at on_train_begin. Never export SWANLAB_PROJECT (swanlab 0.9.2 fails
# parsing it); the --swanlab_project arg is fine.
SWANLAB_MODE_ARG="${SWANLAB_MODE_ARG:-local}"
SWANLAB_ARGS=()
case " $REPORT_TO " in
    *" swanlab "*) SWANLAB_ARGS=(--swanlab_mode "$SWANLAB_MODE_ARG") ;;
esac
OPSD_ARGS=()
if [ "$USE_OPSD_GRPO" = "true" ]; then
    OPSD_ARGS=(--use_opsd_grpo true
               --nzg_reward_name GroundAcc
               --nzg_sft_beta_peak "$NZG_BETA_PEAK"
               --nzg_sft_beta_valley "$NZG_BETA_VALLEY"
               --nzg_sft_warmup_steps "$NZG_WARMUP"
               --nzg_sft_decay_steps "$NZG_DECAY"
               --nzg_sft_all_rollouts "$NZG_ALL_ROLLOUTS"
               --nzg_allow_mixed_reward "$NZG_ALLOW_MIXED"
               --nzg_log_routing "$NZG_LOG_ROUTING")
    # ---- A_OPD branch (stage 2), an ablation; OFF unless NZG_OPD_BETA_PEAK > 0.
    if [ "$(awk -v x="$NZG_OPD_BETA_PEAK" 'BEGIN{print (x>0)?1:0}')" = "1" ]; then
        [ -n "${TEACHER_MODEL:-}" ] || { echo "ERROR: NZG_OPD_BETA_PEAK>0 needs TEACHER_MODEL." >&2; exit 1; }
        OPSD_ARGS+=(--teacher_model "$TEACHER_MODEL"
                    --teacher_model_type "qwen3_vl"
                    --teacher_deepspeed "$TEACHER_DEEPSPEED"
                    --opsd_mask_mode "$OPSD_MASK_MODE"
                    --opsd_mask_dir "$OPSD_MASK_DIR"
                    --opsd_hint_mode "$OPSD_HINT_MODE"
                    --nzg_opd_beta_peak "$NZG_OPD_BETA_PEAK"
                    --nzg_opd_beta_valley "$NZG_OPD_BETA_VALLEY"
                    --nzg_opd_warmup_steps "$NZG_OPD_WARMUP"
                    --nzg_opd_decay_steps "$NZG_OPD_DECAY"
                    --nzg_opd_all_rollouts "$NZG_OPD_ALL_ROLLOUTS"
                    --nzg_route_all "$NZG_ROUTE_ALL"
                    --allow_single_generation "$ALLOW_SINGLE_GEN")
        OPSD_ARGS+=(--nzg_opd_weight_mode "$OPSD_TOKEN_WEIGHT_MODE"
                    --nzg_opd_pos_beta "$OPSD_POS_BETA")
        echo " branch:   + A_OPD ablation  teacher=${TEACHER_MODEL##*/}  hint=${OPSD_HINT_MODE}" \
             "mask=${OPSD_MASK_MODE}  opd_beta ${NZG_OPD_BETA_PEAK}->${NZG_OPD_BETA_VALLEY}"
        echo "           mask_dir=${OPSD_MASK_DIR}  teacher_ds=${TEACHER_DEEPSPEED}" \
             "all_rollouts=${NZG_OPD_ALL_ROLLOUTS}  route_all=${NZG_ROUTE_ALL}  single_gen=${ALLOW_SINGLE_GEN}"
        # A zero2 teacher is fully REPLICATED (~16.3 GiB per rank; prepare_deepspeed maps
        # non-3 stages to 0). Intended: a ZeRO-3 teacher all-gathers inside its forward, and
        # the OPD forward is rank-conditional, so ranks with no routed row never join and
        # the step hangs. Do not switch to zero3.
    fi
    # A second non-zero-weight continuous reward breaks the routing: an all-miss group can
    # still have variance, get a non-zero A_GRPO, and you get GRPO + beta*SFT, not RSTG.
    if [ "$REWARD_FUNCS" != "ground-acc" ] && [ "$NZG_ALLOW_MIXED" != "true" ]; then
        echo "ERROR: USE_OPSD_GRPO=true wants REWARD_FUNCS='ground-acc' REWARD_WEIGHTS='1.0'" >&2
        echo "       (got REWARD_FUNCS='${REWARD_FUNCS}' REWARD_WEIGHTS='${REWARD_WEIGHTS}')." >&2
        echo "       Set NZG_ALLOW_MIXED=true only if the GRPO+SFT hybrid on a continuous" >&2
        echo "       reward is deliberately what you want -- it is NOT the RSTG method." >&2
        exit 1
    fi
    [ "$DYNAMIC_SAMPLE" = "false" ] || {
        echo "ERROR: USE_OPSD_GRPO=true is incompatible with DYNAMIC_SAMPLE=${DYNAMIC_SAMPLE}." >&2; exit 1; }
    grep -q "class OPSDGRPOTrainer" swift/rlhf_trainers/opsd_grpo_trainer.py \
        || { echo "ERROR: swift/rlhf_trainers/opsd_grpo_trainer.py is missing. Refusing to start." >&2; exit 1; }
fi

if [ "$MEASURE_MODE" = "true" ]; then
    STEP_ARGS=(--max_steps "$MAX_STEPS" --save_strategy no)
else
    STEP_ARGS=(--num_train_epochs "$EPOCHS" --save_steps "$SAVE_STEPS" --save_total_limit 100
               --save_only_model "$SAVE_ONLY_MODEL")
    [ -n "$RESUME_FROM" ] && STEP_ARGS+=(--resume_from_checkpoint "$RESUME_FROM")
fi

echo "=============================================="
echo " method:   GRPO   reward='${REWARD_FUNCS}'  weights='${REWARD_WEIGHTS}'"
if [ "$USE_OPSD_GRPO" = "true" ]; then
echo " branch:   OPSD+GRPO  NZG route on 'GroundAcc'  sft_beta ${NZG_BETA_PEAK}->${NZG_BETA_VALLEY}"
echo "           (warmup ${NZG_WARMUP} decay ${NZG_DECAY}, all_rollouts=${NZG_ALL_ROLLOUTS})"
fi
echo " group:    G=${NUM_GEN}  bs=${BS} gas=${GAS} -> $((BS * NPROC * GAS)) completions"
echo "           = $((BS * NPROC * GAS / NUM_GEN)) unique prompts per optimizer step"
echo " policy:   lr=${LR} beta=${BETA} T=${TEMPERATURE} scale_rewards=${SCALE_REWARDS}"
echo "           dynamic_sample=${DYNAMIC_SAMPLE}"
echo " schedule: measure_mode=${MEASURE_MODE} $([ "$MEASURE_MODE" = true ] && echo "max_steps=${MAX_STEPS} (no checkpoints)" || echo "epochs=${EPOCHS}")"
echo " ckpt:     save_only_model=${SAVE_ONLY_MODEL}  resume_from=${RESUME_FROM:-<none>}"
echo " rollout:  GPU ${GPU_ROLLOUT} (enforce_eager)   port ${PORT}"
echo " trainer:  GPUs ${GPUS_TRAINER}  (nproc=${NPROC})   master_port ${MASTER_PORT}"
echo " memory:   gradient_checkpointing=${GRAD_CKPT}  BS=${BS} GAS=${GAS}"
echo " data:     ${DATASET}     output: ${OUT_DIR}/ (add_version=${ADD_VERSION})"
echo " report:   ${REPORT_TO}   dl_workers/rank=${DL_WORKERS} (x${NPROC} = $((NPROC * DL_WORKERS)))"
echo " host:     $(hostname)   GPUs visible here: $(nvidia-smi -L | wc -l)   step GPUs: ${SLURM_STEP_GPUS:-unset}"
echo " python:   $($ENV/python -c 'import sys; print(sys.executable)')"
echo " transformers: $($ENV/python -c 'import transformers; print(transformers.__version__, transformers.__file__)')"
echo "=============================================="

# Otherwise check_num_generations aborts ~2 min in, after the rollout server has started.
GEN_BS=$((BS * NPROC * GAS))
if [ $((GEN_BS % NUM_GEN)) -ne 0 ]; then
    echo "ERROR: generation_batch_size ${GEN_BS} (=BS ${BS} x NPROC ${NPROC} x GAS ${GAS})" >&2
    echo "       is not divisible by NUM_GEN ${NUM_GEN}. Adjust BS or GAS." >&2
    exit 1
fi

# A GPU cannot be both the rollout server and a trainer rank (silent OOM or worse).
for g in ${GPUS_TRAINER//,/ }; do
    [ "$g" = "$GPU_ROLLOUT" ] && {
        echo "ERROR: GPU ${g} is listed as BOTH the rollout GPU and a trainer rank." >&2; exit 1; }
done

grep -q "FIX (2026-08-12): fence the async ncclBroadcast" swift/rlhf_trainers/utils.py \
    || { echo "ERROR: weight-sync fence missing from swift/rlhf_trainers/utils.py. Refusing to start." >&2; exit 1; }

command -v ninja >/dev/null \
    || { echo "ERROR: 'ninja' is not on PATH; vLLM's cudagraph profiling needs it. Refusing to start." >&2; exit 1; }

for g in ${GPU_ROLLOUT} ${GPUS_TRAINER//,/ }; do
    used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i "$g")
    if [ "$used" -gt 1024 ]; then
        echo "ERROR: GPU $g already has ${used} MiB resident. Refusing to start." >&2
        exit 1
    fi
done

# ---- rollout server: 1 GPU ------------------------------------------------
# --vllm_enforce_eager true is REQUIRED (header, CORRECTNESS item 1).
CUDA_VISIBLE_DEVICES="${GPU_ROLLOUT}" \
IMAGE_MAX_TOKEN_NUM=10000 \
$ENV/swift rollout \
    --model_type "qwen3_vl" \
    --model "$MODEL" \
    --vllm_gpu_memory_utilization 0.8 \
    --vllm_max_model_len 20000 \
    --vllm_tensor_parallel_size 1 \
    --vllm_data_parallel_size 1 \
    --vllm_enforce_eager true \
    --host 127.0.0.1 \
    --port $PORT > "$LOGDIR/rollout-$run_name$LOGSUF.log" 2>&1 &
ROLLOUT_PID=$!
# Kill by captured PID, never by pattern (a pattern kill can hit other users' processes);
# remove only the view dir this job created.
trap 'kill $ROLLOUT_PID 2>/dev/null || true' EXIT
echo "rollout PID $ROLLOUT_PID -- to abort cleanly: scancel the srun step, or kill $ROLLOUT_PID plus the torchrun group. Never pkill."

echo "waiting for rollout server on :$PORT ..."
for i in $(seq 1 60); do
    if curl -s -m 3 "http://127.0.0.1:$PORT/health/" >/dev/null 2>&1; then
        echo "rollout up after ~$((i * 20))s: $(curl -s http://127.0.0.1:$PORT/get_world_size/)"
        break
    fi
    if ! kill -0 $ROLLOUT_PID 2>/dev/null; then
        echo "rollout server died, see $LOGDIR/rollout-$run_name$LOGSUF.log" >&2
        exit 1
    fi
    sleep 20
done
curl -s -m 3 "http://127.0.0.1:$PORT/health/" >/dev/null || { echo "rollout never came up" >&2; exit 1; }

# ---- trainer: remaining GPUs, DDP + ZeRO-2 ---------------------------------
NNODES=1 \
NODE_RANK=0 \
MASTER_ADDR=127.0.0.1 \
MASTER_PORT=$MASTER_PORT \
NPROC_PER_NODE=$NPROC \
IMAGE_MAX_TOKEN_NUM=10000 \
PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True' \
CUDA_VISIBLE_DEVICES="${GPUS_TRAINER}" \
$ENV/swift rlhf \
    --rlhf_type grpo \
    --model "$MODEL" \
    --model_type "qwen3_vl" \
    --train_type full \
    --dataset "$DATASET" \
    --torch_dtype bfloat16 \
    --reward_funcs $REWARD_FUNCS \
    --reward_weights $REWARD_WEIGHTS \
    --num_generations $NUM_GEN \
    --temperature $TEMPERATURE \
    --top_p 1.0 \
    --beta $BETA \
    --scale_rewards $SCALE_REWARDS \
    --advantage_estimator grpo \
    --loss_type grpo \
    --num_iterations 1 \
    --dynamic_sample $DYNAMIC_SAMPLE \
    --overlong_filter true \
    --truncation_strategy delete \
    --per_device_train_batch_size $BS \
    --gradient_accumulation_steps $GAS \
    --learning_rate $LR \
    --warmup_ratio 0.03 \
    --max_grad_norm $MAX_GRAD_NORM \
    --max_length 20000 \
    --max_completion_length 128 \
    "${STEP_ARGS[@]}" \
    "${OPSD_ARGS[@]}" \
    "${SWANLAB_ARGS[@]}" \
    --logging_steps 1 \
    --log_completions true \
    --report_to $REPORT_TO \
    --output_dir "$OUT_DIR" \
    --add_version $ADD_VERSION \
    --dataloader_num_workers $DL_WORKERS \
    --dataset_num_proc 4 \
    --deepspeed zero2 \
    --attn_impl flash_attn \
    --gradient_checkpointing "$GRAD_CKPT" \
    --use_vllm true \
    --vllm_mode server \
    --vllm_server_host 127.0.0.1 \
    --vllm_server_port $PORT 2>&1 | tee "$LOGDIR/train-$run_name$LOGSUF.log"
# Default SAVE_ONLY_MODEL=true: checkpoints are weights only and cannot be resumed.
