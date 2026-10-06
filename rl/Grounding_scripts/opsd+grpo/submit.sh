#!/bin/bash
# Submitter for the RL runs reported in the paper. Each arm is a set of env vars on top of the
# SAME chain, so the arms cannot drift apart in the parts that should be identical:
#   submit.sh <arm> (this) -> train_opsd_grpo.sbatch -> train_opsd_grpo.sh -> train_grpo.sh
#
#   bash Grounding_scripts/opsd+grpo/submit.sh <arm> [CFG]        (run from rl/)
#
#   rloo      RLOO (SCALE_REWARDS=none), no OPSD term.
#   opsd      on-policy self-distillation: every group routed to the privileged teacher,
#             reward weight 0.
#   D2RL      run opsd, then rloo with MODEL=BASE_MODEL=<opsd run>/checkpoint-200.
#
# CFG (default per arm) names the run dir output/opsd+grpo/<CFG>/; a non-empty existing dir is
# refused unless RESUME_FROM is set. Env read here: GPUS (default 8 = 1 rollout + 7 trainers),
# JOB_NAME, WALL_TIME, RESUME_FROM, and the beta overrides noted per arm. The whole environment
# is forwarded to the job (--export=ALL), so set ENV_BIN (required) before calling.
#
# Every run evaluates a checkpoint curve on the five _v3 benchmarks and builds the prompt-matched
# zero point once.
#
set -euo pipefail
cd "$(dirname "$(readlink -f "$0")")/../.."

ARM="${1:-}"
case "$ARM" in
    # RLOO. scale_rewards=none stops dividing by the per-group std, so A_i = R_i - mean(R). Literal
    # RLOO is (k/(k-1)) * (R_i - mean), i.e. this times a global 8/7, which Adam ignores, so no
    # --advantage_estimator change is needed (`--advantage_estimator rloo` WITHOUT
    # scale_rewards=none would be a no-op: LOO baseline / std == 8/7 * the std-normalised advantage).
    rloo)    CFG="${2:-gcua-rloo}"
             ENVS="USE_OPSD_GRPO=false,SCALE_REWARDS=none" ;;

    # OPSD: every group routed to the privileged teacher (NZG_ROUTE_ALL) and reward weight 0,
    # so the RL advantage is identically zero and the only gradient is the teacher reverse KL.
    # Beta 1 because Adam normalises the scale anyway; SCALE_REWARDS=none avoids the sigma=0
    # division on an all-zero reward.
    opsd)    CFG="${2:-gcua-opsd}"
             ENVS="OPSD_TOKEN_WEIGHT_MODE=${OPSD_TOKEN_WEIGHT_MODE:-uniform},REWARD_WEIGHTS=0,SCALE_REWARDS=none,NZG_ROUTE_ALL=true,NZG_BETA_PEAK=0,NZG_BETA_VALLEY=0,NZG_OPD_BETA_PEAK=${NZG_OPD_BETA_PEAK:-1},NZG_OPD_BETA_VALLEY=${NZG_OPD_BETA_VALLEY:-1}" ;;

    *) sed -n '2,20p' "$0" >&2; exit 1 ;;
esac

# Refuse to write into a directory that already holds a run: two runs sharing a vdir would
# interleave nzg_routing.jsonl / completions.jsonl and overwrite each other's checkpoints.
VDIR="output/opsd+grpo/${CFG}"
# Exception: RESUME_FROM=<ckpt>|auto (see train_grpo.sh) continues the run in this vdir.
if [ -n "${RESUME_FROM:-}" ]; then
    [ -d "$VDIR" ] || { echo "ERROR: RESUME_FROM set but $VDIR does not exist." >&2; exit 1; }
    echo "resume: RESUME_FROM=${RESUME_FROM} -> continuing the run in $VDIR"
elif [ -e "$VDIR" ] && [ -n "$(ls -A "$VDIR" 2>/dev/null)" ]; then
    echo "ERROR: $VDIR already exists and is not empty." >&2
    echo "       Pick another CFG, or move the old one aside. Refusing to mix two runs." >&2
    exit 1
fi

# Fail in one second rather than 30 minutes into an allocation.
bash -n Grounding_scripts/opsd+grpo/train_opsd_grpo.sbatch

# GPUS=N resizes the job; the sbatch derives 1 rollout + (N-1) trainers from the allocation.
# More GPUs shorten the epoch, not the step, and a whole node may queue much longer than 6 GPUs,
# so fewer GPUs can finish sooner. Check the expected start time first, e.g.
#   sbatch --test-only --gres=gpu:8 --cpus-per-task=128 --time=24:00:00 --wrap=true
GPUS="${GPUS:-8}"
CPUS=$(( GPUS * 16 ))          # 16 CPUs per GPU
if [ "$GPUS" -lt 2 ]; then echo "ERROR: GPUS must be >= 2 (1 rollout + >=1 trainer)." >&2; exit 1; fi

echo "arm=$ARM  cfg=$CFG  vdir=$VDIR"
echo "env: $ENVS"
echo "ckpt: save_only_model=${SAVE_ONLY_MODEL:-true (weights only, not resumable)}"
echo "gpus: $GPUS (1 rollout + $((GPUS-1)) trainers)  cpus: $CPUS"
echo "dataset: ${DATASET:-<unset; required>}"
# JOB_NAME overrides the sbatch's #SBATCH --job-name. On a shared account, give each run a
# distinctive name so a name-based scancel cannot hit someone else's job. Unset => the sbatch's
# own name stands.
echo "job-name: ${JOB_NAME:-<sbatch default: groundgui-gcua-opsdgrpo>}"
# WALL_TIME overrides the sbatch's 16 h --time, e.g. for a short MEASURE_MODE probe. Do not
# shorten it for a real run: training plus the eval ladder needs most of it.
echo "wall-time: ${WALL_TIME:-<sbatch default: 16:00:00>}"
# LOCAL=1: run on this machine instead of submitting -- same environment, same sbatch file.
if [ "${LOCAL:-0}" = 1 ]; then
    IFS=',' read -r -a _kv <<< "CFG=${CFG},${ENVS}"
    export "${_kv[@]}"
    exec bash Grounding_scripts/opsd+grpo/train_opsd_grpo.sbatch
fi
sbatch --gres="gpu:${GPUS}" --cpus-per-task="${CPUS}" ${JOB_NAME:+--job-name="${JOB_NAME}"} \
       ${WALL_TIME:+--time="${WALL_TIME}"} \
       --export="ALL,CFG=${CFG},${ENVS}" Grounding_scripts/opsd+grpo/train_opsd_grpo.sbatch
