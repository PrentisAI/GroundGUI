#!/bin/bash
#SBATCH --job-name=groundgui-gcua-opsdgrpo
# One arm of the grounding RL study, end to end in ONE allocation: preflight -> train ->
# routing audit -> checkpoint-curve eval + zero point -> curve summary -> optional endpoint.
#
# Chain:  submit.sh <arm> -> train_opsd_grpo.sbatch (this) -> train_opsd_grpo.sh -> train_grpo.sh
#
# Normally submitted by submit.sh, which passes CFG and the arm's env via --export. Direct use:
#   sbatch --export=ALL,ENV_BIN=<rl env>/bin,CFG=<name> Grounding_scripts/opsd+grpo/train_opsd_grpo.sbatch
# Run from rl/ (REPO defaults to the submit dir).
#
# Env read here: ENV_BIN (required), CFG, REPO, BASE_MODEL, DATASET, BENCH_SUFFIX, REPORT_TO,
#   BS/GAS/NUM_GEN (divisibility check only), SKIP_PREFLIGHT,
#   PREFLIGHT_REPLAY_ROWS, SKIP_TRAIN, EVAL_STEPS, BASE_EXP; everything else is consumed by
#   train_opsd_grpo.sh /
#   train_grpo.sh. Give each pipeline its own job name so a name-based scancel cannot hit
#   another run.
#SBATCH --gres=gpu:8
#SBATCH --gres-flags=disable-binding
#SBATCH --cpus-per-task=128
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --time=16:00:00
# 16 h leaves headroom over training plus the eval ladder. If training overruns, the
# checkpoints survive and SKIP_TRAIN=1 re-enters at the eval.
# --output: point it at a log directory that exists before submission; Slurm cannot create it
# and the job then fails without leaving a log.
#SBATCH --output=slurm-gcua-opsdgrpo-%j.log
# --cpus-per-task = 16 per GPU. The topology below is DERIVED from the allocation, so running
# on fewer GPUs is one flag:
#   sbatch --gres=gpu:6 --cpus-per-task=96 Grounding_scripts/opsd+grpo/train_opsd_grpo.sbatch
# or `GPUS=6 bash Grounding_scripts/opsd+grpo/submit.sh <arm>`.
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
# train_grpo.sh, which also asserts the imported path). Set here too so the CPU-only stages
# (routing audit, analysis) resolve the same tree.
export PYTHONPATH="$REPO${PYTHONPATH:+:$PYTHONPATH}"

# ---- swanlab -------------------------------------------------------------------
# LOCAL mode: needs no login and no TTY; cloud mode aborts a non-interactive job. The real switch
# is the `--swanlab_mode local` arg added by train_grpo.sh (it overrides SWANLAB_MODE).
# SWANLAB_PROJECT must stay UNSET: swanlab 0.9.2 parses it through pydantic_settings and dies.
export SWANLAB_MODE=local
unset SWANLAB_PROJECT SWANLAB_LOG_DIR || true
export REPORT_TO="${REPORT_TO:-tensorboard swanlab}"

# ---- run identity --------------------------------------------------------------
# Config-named dir with --add_version false, so the eval below knows the path without
# discovering a timestamp.
CFG="${CFG:-gcua-run}"
export OUT_DIR="output/opsd+grpo/${CFG}"
export ADD_VERSION=false
export RUN_TAG="$CFG"
VDIR="$OUT_DIR"

# _v3 = benchmark files using the internvl2_5_desktop_grounding_v1 system prompt (md5
# bc0d22f21d3e1b5c451e7bf03ce49cf6), the SAME prompt as the GroundCUA training rows and the SFT
# init. The unsuffixed ground_benchmark files use a different system prompt and would measure
# the model off-distribution. Training corpus and BENCH_SUFFIX go together. Results
# land in <ckpt>/infer_result<BENCH_SUFFIX>/, hence RESULT_DIRNAME in the symlink below.
BENCH_SUFFIX="${BENCH_SUFFIX:-_v3}"
RESULT_DIRNAME="infer_result${BENCH_SUFFIX}"
# Must point at the same model as MODEL whenever MODEL is overridden: train_grpo.sh reads
# MODEL for trainer/rollout, while BASE_MODEL drives the preflight gate and the zero point
# here. A mismatch would silently preflight and baseline the wrong model.
BASE_MODEL="${BASE_MODEL:?set BASE_MODEL=<the checkpoint the run starts from (same as MODEL)>}"

echo "host=$(hostname)  job=${SLURM_JOB_ID:-none}  step GPUs=${SLURM_STEP_GPUS:-unset}"
echo "cfg=$CFG  vdir=$VDIR  report_to=$REPORT_TO  bench_suffix=$BENCH_SUFFIX"
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
# started). The defaults must mirror train_opsd_grpo.sh, which runs later under srun: GAS=24.
# The authoritative numbers are
# train_grpo.sh's own " group:" banner line in the log.
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
     "(predicted; train_grpo.sh's ' group:' line is authoritative)"

# ================================ 0. PREFLIGHT =================================
# CPU-only, ~30 s. Gates on the imported swift tree, the transformers-4 config repair of the
# checkpoint, the corpus columns + prompt md5 + GT-replay reward, and the benchmark prompt
# match -- all otherwise silent. SKIP_PREFLIGHT=1 only to debug the preflight itself.
if [ "${SKIP_PREFLIGHT:-0}" != "1" ]; then
    $ENV/python Grounding_scripts/preflight.py \
        --model "$BASE_MODEL" \
        --dataset "$DATASET" \
        --bench-suffix "$BENCH_SUFFIX" \
        --replay-rows "${PREFLIGHT_REPLAY_ROWS:-0}" \
        $([ "${TRAIN_ONLY:-0}" = 1 ] && echo --skip-bench) \
        || { echo "ERROR: preflight failed. Refusing to burn the allocation." >&2; exit 1; }
fi

# ================================ 1. TRAIN =====================================
# SKIP_TRAIN=1 turns this into an EVAL-ONLY job against the checkpoints already on disk, for
# when training overran --time and the eval stages never ran:
#   sbatch --export=ALL,SKIP_TRAIN=1,CFG=<cfg> --time=4:00:00 <this file>
if [ "${SKIP_TRAIN:-0}" = "1" ]; then
    echo "=== SKIP_TRAIN=1: evaluating the existing checkpoints under $VDIR ==="
    [ -d "$VDIR" ] || { echo "ERROR: $VDIR does not exist; nothing to evaluate" >&2; exit 1; }
else
    ${SLURM_JOB_ID:+srun --unbuffered} bash 'Grounding_scripts/opsd+grpo/train_opsd_grpo.sh'
    echo "=== training returned 0 ==="
fi

# Weight-sync health: if the fence failed, the run trained on garbage rollouts.
$ENV/python Grounding_scripts/rollout_health.py "$VDIR" 2>&1 | tail -8 || true

if [ "${TRAIN_ONLY:-0}" = 1 ]; then
    echo "=== TRAIN_ONLY=1: training finished; checkpoints are in $VDIR ==="
    exit 0
fi

# ================================ 3. EVAL ======================================
# EVALUATE A CURVE, NOT ONLY THE LAST CHECKPOINT. RL accuracy can peak mid-run and then decay
# as the actor entropy collapses, so the last checkpoint is often not the best. The last one is
# evaluated too, to show whether such a decay occurred.
EVAL_STEPS="${EVAL_STEPS:-50 75 100 125}"
LAST=$(ls "$VDIR" 2>/dev/null | sed -n 's/^checkpoint-\([0-9]*\)$/\1/p' | sort -n | tail -1)
[ -n "$LAST" ] || { echo "ERROR: no checkpoint-* under $VDIR; nothing to evaluate" >&2; exit 1; }
# de-duplicate, keep numeric order, and append the final step
EVAL_STEPS=$(printf '%s\n' $EVAL_STEPS "$LAST" | sort -n -u | tr '\n' ' ')
# Rough per-checkpoint cost: sharded inference scales with GPU count, model loads and
# analysis do not.
echo "=== eval plan: steps [${EVAL_STEPS}] (last=${LAST}); ~$(( 19 * 8 / NGPU + 6 )) min of ${NGPU}-GPU time each ==="

EVALUATED=""
for STEP in $EVAL_STEPS; do
    CKPT="checkpoint-$STEP"
    if ! { test -s "$VDIR/$CKPT/model.safetensors.index.json" || test -s "$VDIR/$CKPT/model.safetensors"; }; then
        echo "--- SKIP $CKPT (no safetensors; SAVE_STEPS may not have produced this step)"
        continue
    fi
    echo; echo "########## EVAL $CKPT ##########"
    # run_name must be 'opsd+grpo' (an existing dir under output/): test_6gpu.sh checks
    # output/<run_name> exists before it honours VDIR. Infer logs are already per-checkpoint.
    # EXTRA_ARGS='--temperature 0' is REQUIRED: the init's generation_config samples (T=0.7),
    # which changes a few percent of rows between runs -- too noisy for a per-step curve.
    # MODEL= (empty) is REQUIRED: test_6gpu.sh treats a MODEL in the environment as "evaluate
    # only this model", so an inherited MODEL would silently re-score the init at every step
    # and leave the symlink below dangling.
    MODEL= VDIR="$VDIR" CKPT="$CKPT" GPUS="$ALL_GPUS" MAX_GPUS="$NGPU" MAX_BATCH_SIZE=4 \
        BENCH_SUFFIX="$BENCH_SUFFIX" \
        EXTRA_ARGS='--temperature 0' \
        bash Grounding_scripts/test_6gpu.sh 'opsd+grpo' \
            screenspotv2 screenspotpro mmbench osworldg osworldg_r
    EXP_STEP="exp/GroundCUA/${CFG}/step-${STEP}"
    mkdir -p "$EXP_STEP"
    # The symlink MUST carry BENCH_SUFFIX: pointed at a bare `infer_result` it would dangle and
    # analyze_grounding.py would skip all datasets and still exit 0 with an empty summary.
    ln -sfnT "$PWD/$VDIR/$CKPT/$RESULT_DIRNAME" "$EXP_STEP/infer"
    $ENV/python Grounding_scripts/analyze_grounding.py --exp-dir "$EXP_STEP" --verify
    EVALUATED="$EVALUATED $STEP"
done
[ -n "$EVALUATED" ] || { echo "ERROR: no checkpoint was successfully evaluated" >&2; exit 1; }

# ---- 3b. the ZERO POINT (the init itself, same prompt, same decode) ----
# Required baseline, evaluated once and cached. Items are paired on the instruction text, so
# baselines evaluated under a different prompt do not pair (ITEM MISMATCH). Written
# outside the read-only init via RESULT_ROOT, and keyed on the init's name so different inits
# never share a cached baseline.
BASE_EXP="${BASE_EXP:-exp/GroundCUA/base_$(basename "$BASE_MODEL")${BENCH_SUFFIX}}"
if [ ! -s "$BASE_EXP/analysis/summary.json" ]; then
    echo; echo "########## EVAL ZERO POINT: $BASE_MODEL ##########"
    mkdir -p "$BASE_EXP/infer"
    MODEL="$BASE_MODEL" \
    RESULT_ROOT="$BASE_EXP/infer" \
    GPUS="$ALL_GPUS" MAX_GPUS="$NGPU" MAX_BATCH_SIZE=4 \
    BENCH_SUFFIX="$BENCH_SUFFIX" \
    EXTRA_ARGS='--temperature 0' \
        bash Grounding_scripts/test_6gpu.sh "gcua-base-$(basename "$BASE_MODEL")" \
            screenspotv2 screenspotpro mmbench osworldg osworldg_r
    $ENV/python Grounding_scripts/analyze_grounding.py --exp-dir "$BASE_EXP" --verify
else
    echo "=== zero point already analysed at $BASE_EXP/analysis -- reusing ==="
fi

# ================================ 4. THE CURVE =================================
# The deliverable of this job is the curve, not a single number. Read the peak, not the end.
echo; echo "########## POOLED ACCURACY vs STEP ##########"
$ENV/python - "$BASE_EXP/analysis/summary.json" "exp/GroundCUA/${CFG}" $EVALUATED <<'PYCURVE'
import json, os, sys
base_p, root = sys.argv[1], sys.argv[2]
steps = [int(x) for x in sys.argv[3:]]

def pooled(p):
    if not os.path.exists(p):
        return None
    d = json.load(open(p))["datasets"]
    c = sum(v["correct"] for v in d.values())
    n = sum(v["n"] for v in d.values())
    return (100.0 * c / n, n, d) if n else None

b = pooled(base_p)
if b:
    print(f"  step {'0 (init)':>10}  pooled {b[0]:6.2f}%  n={b[1]}")
best = None
for s_ in steps:
    r = pooled(os.path.join(root, f"step-{s_}", "analysis", "summary.json"))
    if not r:
        print(f"  step {s_:>10}  (no analysis)")
        continue
    delta = f"{r[0]-b[0]:+.2f}" if b else "  n/a"
    sp = r[2].get("screenspotpro", {})
    ssp = f"  ss-pro {100.0*sp['correct']/sp['n']:5.2f}%" if sp.get("n") else ""
    print(f"  step {s_:>10}  pooled {r[0]:6.2f}%  vs init {delta} pp{ssp}")
    if best is None or r[0] > best[1]:
        best = (s_, r[0])
if best:
    print()
    print(f"  BEST: step {best[0]} at {best[1]:.2f}% pooled.")
    if best[0] == max(steps):
        print("  NOTE: the best step is the LAST one, so the peak may lie beyond this run -- the")
        print("        curve can still decay later, so verify before extending.")
    else:
        print("  The curve peaked before the last step; compare runs at their best step, not their last.")
    print(f"  Use exp/GroundCUA/{os.path.basename(root)}/step-{best[0]}/analysis for any comparison,")
    print("  and say WHICH step every reported number came from.")
PYCURVE

# For osworldg read `accuracy_nonrefusal`, not `accuracy`: its 54 refusal items are unreachable
# under the internvl2_5_desktop_grounding_v1 schema (no grounding `terminate` action), so
# `accuracy` caps at 90.43%.

echo "=== DONE. The curve is above; per-step analyses are in"
echo "    exp/GroundCUA/${CFG}/step-*/analysis ; the zero point is in $BASE_EXP/analysis. ==="
