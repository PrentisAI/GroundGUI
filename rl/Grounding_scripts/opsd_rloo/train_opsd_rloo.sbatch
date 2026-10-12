#!/bin/bash
#SBATCH --job-name=groundgui-rl
# One arm of the grounding RL runs in ONE allocation: preflight -> train. Evaluate the saved
# checkpoints afterwards with scripts/eval.sh.
#
# Chain:  submit.sh <arm> -> train_opsd_rloo.sbatch (this) -> train_opsd_rloo.sh -> train_rl.sh
#
# Normally submitted by submit.sh, which passes CFG and the arm's env via --export. Direct use:
#   sbatch --export=ALL,ENV_BIN=<rl env>/bin,CFG=<name> Grounding_scripts/opsd_rloo/train_opsd_rloo.sbatch
# Run from rl/ (REPO defaults to the submit dir).
#
# Env read here: ENV_BIN (required), CFG, REPO, BASE_MODEL, DATASET, REPORT_TO,
#   BS/GAS/NUM_GEN (divisibility check only), SKIP_PREFLIGHT,
#   PREFLIGHT_REPLAY_ROWS; everything else is consumed by
#   train_opsd_rloo.sh /
#   train_rl.sh. Give each pipeline its own job name so a name-based scancel cannot hit
#   another run.
#SBATCH --gres=gpu:8
#SBATCH --gres-flags=disable-binding
#SBATCH --cpus-per-task=128
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --time=16:00:00
# 16 h leaves headroom over training.
# --output: point it at a log directory that exists before submission; Slurm cannot create it
# and the job then fails without leaving a log.
#SBATCH --output=slurm-groundgui-rl-%j.log
# --cpus-per-task = 16 per GPU. The topology below is DERIVED from the allocation, so running
# on fewer GPUs is one flag:
#   sbatch --gres=gpu:6 --cpus-per-task=96 Grounding_scripts/opsd_rloo/train_opsd_rloo.sbatch
# or `GPUS=6 bash Grounding_scripts/opsd_rloo/submit.sh <arm>`.
set -euo pipefail

REPO="${REPO:-${SLURM_SUBMIT_DIR:-$PWD}}"   # submit.sh cds to rl/ before sbatch
ENV="${ENV_BIN:?set ENV_BIN=<rl env>/bin, see rl/requirements.lock.txt}"
cd "$REPO"

# PYTHONNOUSERSITE: a stale ~/.local transformers can shadow the env. VLLM_USE_DEEP_GEMM=0: an
# outdated deep_gemm build fails vLLM startup. PATH must be exported explicitly because sbatch
# does not run a login shell (vLLM's cudagraph profiling shells out to `ninja`).
export PYTHONNOUSERSITE=1
export VLLM_USE_DEEP_GEMM=0
export PATH="$ENV:$PATH"

# Make Python import THIS repo's swift, not an editable install elsewhere in the env (see
# train_rl.sh, which also asserts the imported path). Set here too so the CPU-only stages
# (routing audit, analysis) resolve the same tree.
export PYTHONPATH="$REPO${PYTHONPATH:+:$PYTHONPATH}"

# ---- swanlab -------------------------------------------------------------------
# LOCAL mode: needs no login and no TTY; cloud mode aborts a non-interactive job. The real switch
# is the `--swanlab_mode local` arg added by train_rl.sh (it overrides SWANLAB_MODE).
# SWANLAB_PROJECT must stay UNSET: swanlab 0.9.2 parses it through pydantic_settings and dies.
export SWANLAB_MODE=local
unset SWANLAB_PROJECT SWANLAB_LOG_DIR || true
export REPORT_TO="${REPORT_TO:-tensorboard swanlab}"

# ---- run identity --------------------------------------------------------------
# Config-named dir with --add_version false, so checkpoints land at a fixed path.
CFG="${CFG:-gcua-run}"
export OUT_DIR="output/opsd_rloo/${CFG}"
export ADD_VERSION=false
export RUN_TAG="$CFG"
VDIR="$OUT_DIR"

# Must point at the same model as MODEL whenever MODEL is overridden: train_rl.sh reads
# MODEL for trainer/rollout, while BASE_MODEL drives the preflight gate here. A mismatch
# would silently preflight the wrong model.
BASE_MODEL="${BASE_MODEL:?set BASE_MODEL=<the checkpoint the run starts from (same as MODEL)>}"

echo "host=$(hostname)  job=${SLURM_JOB_ID:-none}  step GPUs=${SLURM_STEP_GPUS:-unset}"
echo "cfg=$CFG  vdir=$VDIR  report_to=$REPORT_TO"
echo "dataset=${DATASET:?set DATASET=<absolute path of the RL corpus .jsonl>}"
echo "arm:  sft_beta=${NZG_BETA_PEAK:-<default 0.005>}  opd_beta=${NZG_OPD_BETA_PEAK:-0}" \
     "$([ "$(awk -v x="${NZG_OPD_BETA_PEAK:-0}" 'BEGIN{print (x>0)?1:0}')" = 1 ] \
        && echo "teacher=${OPSD_HINT_MODE:-hint}/${OPSD_MASK_MODE:-gaussian}" \
        || echo '(A_OPD off)')"
nvidia-smi -L

# ---- topology, DERIVED from the allocation -------------------------------------------------
# Inside the job's device cgroup the GPUs are renumbered 0..N-1, so these are cgroup-local
# indices. The rollout server always gets exactly 1 GPU (generation is a small fraction of a
# step, vLLM data-parallel is unavailable on 0.22.1 for a dense model, and TP=1 is enough);
# every other GPU is a trainer rank (DDP + ZeRO-2).
NGPU=$(nvidia-smi -L | wc -l)
[ "${NGPU:-0}" -ge 2 ] || { echo "ERROR: need >=2 GPUs, saw ${NGPU}" >&2; exit 1; }
export GPU_ROLLOUT=$((NGPU - 1))
export GPUS_TRAINER=$(seq -s, 0 $((NGPU - 2)))
NPROC=$((NGPU - 1))
# Pre-allocation divisibility check (arguments.py would abort only after the rollout server
# started). The defaults must mirror train_opsd_rloo.sh, which runs later under srun: GAS=24.
# The authoritative numbers are
# train_rl.sh's own " group:" banner line in the log.
_NG=${NUM_GEN:-8}
_GA=${GAS:-24}
GEN_BS=$(( ${BS:-2} * NPROC * _GA ))
if [ $(( GEN_BS % _NG )) -ne 0 ]; then
    echo "ERROR: generation_batch_size ${GEN_BS} (BS x ${NPROC} trainers x GAS ${_GA}) is not" >&2
    echo "       divisible by NUM_GEN ${_NG}. Adjust BS or GAS for this GPU count." >&2
    exit 1
fi
ALL_GPUS=$(seq -s, 0 $((NGPU - 1)))
echo "topology: ${NGPU} GPUs = 1 rollout (cgroup ${GPU_ROLLOUT}) + ${NPROC} trainers (${GPUS_TRAINER})"
echo "          ${GEN_BS} completions/step -> $(( GEN_BS / _NG )) unique prompts/step" \
     "(predicted; train_rl.sh's ' group:' line is authoritative)"

# ================================ 0. PREFLIGHT =================================
# CPU-only, ~30 s. Gates on the imported swift tree, the transformers-4 config repair of the
# checkpoint, and the corpus columns + prompt md5 + GT-replay reward -- all otherwise silent.
# SKIP_PREFLIGHT=1 only to debug the preflight itself.
if [ "${SKIP_PREFLIGHT:-0}" != "1" ]; then
    $ENV/python Grounding_scripts/preflight.py \
        --model "$BASE_MODEL" \
        --dataset "$DATASET" \
        --replay-rows "${PREFLIGHT_REPLAY_ROWS:-0}" \
        --skip-bench \
        || { echo "ERROR: preflight failed. Refusing to burn the allocation." >&2; exit 1; }
fi

# ================================ 1. TRAIN =====================================
${SLURM_JOB_ID:+srun --unbuffered} bash 'Grounding_scripts/opsd_rloo/train_opsd_rloo.sh'
echo "=== training returned 0 ==="

# Weight-sync health: if the fence failed, the run trained on garbage rollouts.
$ENV/python Grounding_scripts/rollout_health.py "$VDIR" 2>&1 | tail -8 || true

echo "=== training finished; checkpoints are in $VDIR (evaluate them with scripts/eval.sh) ==="
