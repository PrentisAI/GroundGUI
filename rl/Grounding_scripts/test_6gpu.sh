#!/bin/bash
# Grounding benchmark evaluation: `swift infer` (pt backend, sharded over GPUs) for every
# checkpoint of a run -- or one CKPT, or one MODEL dir -- on the chosen ground_benchmark files.
# Called by the eval stage of opsd+grpo/train_opsd_grpo.sbatch; also usable standalone.
#
# Compared with a plain fixed 8-GPU eval script, it:
#   * auto-detects free GPUs (<1 GiB used), capped at MAX_GPUS, or takes GPUS= explicitly, and
#     restricts them to the allocation of the Slurm job named HOLD_JOB if one runs on this host;
#   * defaults MASTER_ADDR / MASTER_PORT, so it works outside a distributed launcher;
#   * can pin one checkpoint (CKPT=) or one version dir (VDIR=);
#   * calls the RL env's swift ($ENV_BIN/swift), not whatever `swift` is on $PATH.
#
# Usage:
#     ENV_BIN=<rl env>/bin bash Grounding_scripts/test_6gpu.sh <run_name> <dataset1> [dataset2 ...]
#
#     datasets: screenspotpro screenspotv2 mmbench osworldg osworldg_r
#               (uivision is mapped but build_ground_benchmark.py does not build it)
#
# Environment overrides:
#     ENV_BIN=<env>/bin   REQUIRED: bin/ of the RL env (see rl/requirements.lock.txt)
#     GPUS=1,2,3,5,6      explicit GPU list (default: auto-detect free GPUs)
#     MAX_GPUS=6          cap on auto-detected GPUs (default 6)
#     HOLD_JOB=<name>     Slurm job name whose GPU allocation bounds the auto-detection
#     CKPT=checkpoint-118 evaluate only this checkpoint (name under the vdir, or
#                         an absolute path). Default: every checkpoint in the vdir.
#     VDIR=<path>         pin a specific output/<run>/<version> dir (default: newest with ckpts)
#     MODEL=<dir>         evaluate this model dir instead of a run's checkpoints;
#     RESULT_ROOT=<dir>   ... and write its results here instead of into the model dir
#     BENCH_SUFFIX=_v3    read ground_benchmark/<ds>_v3.jsonl and write into
#                         <ckpt>/infer_result_v3/ instead of infer_result/.
#                         Also keeps subset runs (e.g. _smoke) from colliding with real results.
#     MAX_BATCH_SIZE=4    per-rank inference batch size
#     EVAL_TEMPERATURE=<T>  decoding temperature; DEFAULT 0 (greedy)
#     EXTRA_ARGS='...'
#                         extra flags appended to `swift infer`
#     SCORE=1             run swift/metrics/total_metric.py afterwards
#                         (only valid when BENCH_SUFFIX is empty -- total_metric.py
#                          hardcodes the infer_result/ directory name)
#
# Examples:
#     GPUS=0,1,2,3 CKPT=checkpoint-100 BENCH_SUFFIX=_v3 \
#         bash Grounding_scripts/test_6gpu.sh opsd+grpo screenspotv2
#
#     MODEL=models/<init> RESULT_ROOT=exp/base/infer BENCH_SUFFIX=_v3 \
#         bash Grounding_scripts/test_6gpu.sh base screenspotv2 screenspotpro

set -euo pipefail

# This script builds NO prompt: it only selects ground_benchmark/<stem>${BENCH_SUFFIX}.jsonl.
# Match BENCH_SUFFIX to the system prompt the model was trained with; for the default RL init
# that is _v3 (internvl2_5_desktop_grounding_v1). The unsuffixed files use a different system
# prompt and would measure it off-distribution.
REPO="$(cd "$(dirname "$(readlink -f "$0")")/.." && pwd)"
cd "$REPO"

ENV="${ENV_BIN:?set ENV_BIN=<rl env>/bin, see rl/requirements.lock.txt}"
LOGDIR="${LOGDIR:-$REPO/logs}"
mkdir -p "$LOGDIR"
# Make `import swift` (and `swift infer`) resolve to this repo's copy; see train_grpo.sh.
export PYTHONPATH="$REPO${PYTHONPATH:+:$PYTHONPATH}"

run_name="${1:-}"
shift || true
dataset_names=("$@")

if [ -z "$run_name" ] || [ ${#dataset_names[@]} -eq 0 ]; then
    sed -n '/^# Usage:/,/^$/p' "$0"
    exit 1
fi

# ---- dataset -> ground_benchmark file -------------------------------------
declare -A DATASET_MAP
DATASET_MAP["screenspotpro"]="screenspotpro"
DATASET_MAP["screenspotv2"]="screenspotv2"
DATASET_MAP["uivision"]="ui-vision"
DATASET_MAP["mmbench"]="mmbench"
DATASET_MAP["osworldg"]="osworldg"
DATASET_MAP["osworldg_r"]="osworldg_r"

BENCH_SUFFIX="${BENCH_SUFFIX:-}"
MAX_BATCH_SIZE="${MAX_BATCH_SIZE:-4}"
# GREEDY BY DEFAULT. The model's generation_config.json has do_sample=true T=0.7 top_p=0.8
# top_k=20, so an eval that passes no temperature is SAMPLED. Sampling flips a few percent of
# rows between runs: invisible in a pooled delta, but on a paired per-item comparison that
# decoder noise can be the same order as the effect being measured. Use one value for every arm.
EVAL_TEMPERATURE="${EVAL_TEMPERATURE:-0}"
RESULT_DIRNAME="infer_result${BENCH_SUFFIX}"

# ---- GPU selection ---------------------------------------------------------
# Free VRAM does not prove a GPU is unclaimed (another job's reservation can sit idle between
# steps), so if a Slurm job named HOLD_JOB runs on this host, only its GPUs are eligible.
MAX_GPUS="${MAX_GPUS:-6}"
HOLD_JOB="${HOLD_JOB:-}"

allowed=""
if command -v squeue >/dev/null 2>&1; then
    jobid=$(squeue -h -o "%i %j %N" 2>/dev/null \
            | awk -v n="$HOLD_JOB" -v h="$(hostname -s)" '$2 == n && $3 == h {print $1; exit}')
    if [ -n "$jobid" ]; then
        # `scontrol -d` (detail) is required: the plain form reports only
        # gres/gpu=6, without the GRES=gpu:6(IDX:0-3,5-6) index list.
        allowed=$(scontrol -d show job "$jobid" 2>/dev/null \
                  | grep -om1 'IDX:[0-9,-]*' | cut -d: -f2 \
                  | awk -F',' '{for(i=1;i<=NF;i++){
                        if (split($i,r,"-")==2) {for(g=r[1];g<=r[2];g++) printf "%s%s",(c++?",":""),g}
                        else printf "%s%s",(c++?",":""),$i}}')
        [ -n "$allowed" ] && echo "slurm job ${jobid} (${HOLD_JOB}) holds GPUs: ${allowed}"
    fi
fi

if [ -n "${GPUS:-}" ]; then
    gpu_list="$GPUS"
    if [ -n "$allowed" ]; then
        for g in ${gpu_list//,/ }; do
            grep -qx "$g" <<< "${allowed//,/$'\n'}" \
                || echo "WARNING: GPU ${g} is outside ${HOLD_JOB}'s allocation (${allowed})" >&2
        done
    fi
else
    free=$(nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits \
           | awk -F', ' '$2 < 1024 {printf "%s%s", (c++?",":""), $1}')
    gpu_list=$(awk -F',' -v allow="$allowed" -v n="$MAX_GPUS" '{
        for (i=1;i<=NF;i++) {
            if (allow != "" && index(","allow",", ","$i",") == 0) continue
            if (c++ < n) printf "%s%s", (c>1?",":""), $i
        }}' <<< "$free")
    if [ -z "$gpu_list" ]; then
        echo "ERROR: no free GPU inside the allocation. Set GPUS= to override." >&2
        exit 1
    fi
fi
nproc_per_node=$(awk -F',' '{print NF}' <<< "$gpu_list")

# ---- model selection -------------------------------------------------------
# MODEL= evaluates an arbitrary model directory and bypasses the checkpoint sweep entirely;
# RESULT_ROOT= says where its results go, so a read-only base model dir is not written to.
if [ -n "${MODEL:-}" ]; then
    [ -d "$MODEL" ] || { echo "ERROR: no such model dir: $MODEL" >&2; exit 1; }
    ckpts=("$MODEL")
    vdir='(MODEL override)'
else
    base_dir="output/${run_name}"
    [ -d "$base_dir" ] || { echo "ERROR: no such run: $base_dir" >&2; exit 1; }

    if [ -n "${VDIR:-}" ]; then
        vdir="$VDIR"
    else
        # Newest subdir that actually holds checkpoints, not `ls -1dt $base_dir/v*`: config-named
        # dirs (--add_version false) are invisible to a v* glob, and the eval would silently fall
        # back to another run's numbers. `|| true` and the braces are load-bearing under
        # `set -euo pipefail`: if the last dir examined has no checkpoints the loop exits 1 and
        # the script would abort with no message.
        vdir=$({ for d in "$base_dir"/*/; do
                   [ -n "$(echo "$d"checkpoint-* 2>/dev/null | grep -v '\*')" ] && printf '%s\n' "${d%/}"
               done || true; } | xargs -r ls -1dt 2>/dev/null | head -n 1)
    fi
    [ -n "${vdir:-}" ] && [ -d "$vdir" ] \
        || { echo "ERROR: no directory containing checkpoint-* under $base_dir. Pass VDIR=." >&2; exit 1; }
    echo "vdir: $vdir  ${VDIR:+(pinned via VDIR)}${VDIR:-(auto-selected: newest dir with checkpoints)}"

    if [ -n "${CKPT:-}" ]; then
        case "$CKPT" in
            /*) ckpts=("$CKPT") ;;
            *)  ckpts=("${vdir}/${CKPT}") ;;
        esac
        [ -d "${ckpts[0]}" ] || { echo "ERROR: no such checkpoint: ${ckpts[0]}" >&2; exit 1; }
    else
        mapfile -t ckpts < <(ls -1d ${vdir}/checkpoint-* 2>/dev/null | sort -V)
        [ ${#ckpts[@]} -gt 0 ] || { echo "ERROR: no checkpoint-* under $vdir" >&2; exit 1; }
    fi
fi

echo "==============================================="
echo " run:        ${run_name}"
echo " vdir:       ${vdir}"
echo " checkpoints:${#ckpts[@]}  ${ckpts[*]##*/}"
echo " datasets:   ${dataset_names[*]}"
echo " GPUs:       ${gpu_list}  (nproc_per_node=${nproc_per_node})"
echo " benchmarks: ground_benchmark/<ds>${BENCH_SUFFIX}.jsonl"
if [ -n "${RESULT_ROOT:-}" ]; then
    echo " results:    ${RESULT_ROOT}/<ds>.jsonl"
else
    echo " results:    <ckpt>/${RESULT_DIRNAME}/<ds>.jsonl"
fi
echo "==============================================="

# Fail before burning a model load if a benchmark file is missing.
for dataset_name in "${dataset_names[@]}"; do
    stem="${DATASET_MAP[$dataset_name]:-}"
    [ -n "$stem" ] || { echo "ERROR: unknown dataset '${dataset_name}'" >&2; exit 1; }
    f="ground_benchmark/${stem}${BENCH_SUFFIX}.jsonl"
    [ -f "$f" ] || { echo "ERROR: missing $f (build it with Grounding_scripts/build_ground_benchmark.py; note the *_v3.jsonl files have NO builder in this repo -- they are inputs, copied as data)" >&2; exit 1; }
done

for ckpt in "${ckpts[@]}"; do
    for dataset_name in "${dataset_names[@]}"; do
        stem="${DATASET_MAP[$dataset_name]}"
        val_dataset="ground_benchmark/${stem}${BENCH_SUFFIX}.jsonl"
        if [ -n "${RESULT_ROOT:-}" ]; then
            output_path="${RESULT_ROOT}/${dataset_name}.jsonl"
        else
            output_path="${ckpt}/${RESULT_DIRNAME}/${dataset_name}.jsonl"
        fi
        mkdir -p "$(dirname "$output_path")"

        echo
        echo "--- $(basename "$ckpt") / ${dataset_name}  ($(wc -l < "$val_dataset") samples) ---"

        # swift infer appends; a stale file would double-count.
        rm -f "$output_path"

        CUDA_VISIBLE_DEVICES="${gpu_list}" \
        IMAGE_MAX_TOKEN_NUM="${IMAGE_MAX_TOKEN_NUM:-10000}" \
        NNODES=1 \
        NODE_RANK=0 \
        MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}" \
        MASTER_PORT="${MASTER_PORT:-29531}" \
        NPROC_PER_NODE="${nproc_per_node}" \
        PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True' \
        $ENV/swift infer \
            --model "${ckpt}" \
            --model_type qwen3_vl \
            --stream false \
            --infer_backend pt \
            --max_length "${MAX_LENGTH:-20000}" \
            --max_new_tokens 1024 \
            --result_path "${output_path}" \
            --val_dataset "${val_dataset}" \
            --max_batch_size "${MAX_BATCH_SIZE}" \
            --remove_unused_columns false \
            --dataset_shuffle false \
            --write_batch_size 800 \
            --temperature "${EVAL_TEMPERATURE}" \
            ${EXTRA_ARGS:-} \
            2>&1 | tee "$LOGDIR/infer-${run_name}-$(basename "$ckpt")-${dataset_name}${BENCH_SUFFIX}.log"

        echo ">>> saved $(wc -l < "$output_path" 2>/dev/null || echo 0) rows to ${output_path}"
    done
done

echo
echo "Inference done."

if [ "${SCORE:-0}" = "1" ]; then
    if [ -n "$BENCH_SUFFIX" ]; then
        echo "SCORE=1 ignored: total_metric.py hardcodes infer_result/, but results" >&2
        echo "went to ${RESULT_DIRNAME}/. Score this subset directly, e.g.:" >&2
        echo "  $ENV/python -c \"from swift.utils import read_from_jsonl; from swift.metrics.screenspotv2_metric import compute_screenspotv2; compute_screenspotv2(read_from_jsonl('${ckpts[0]}/${RESULT_DIRNAME}/${dataset_names[0]}.jsonl'), 'screenspotv2_navi_qwen3')\"" >&2
    else
        echo "=== scoring (metric=navi_qwen3) ==="
        $ENV/python swift/metrics/total_metric.py \
            --run_name "${run_name}" \
            --metric navi_qwen3 \
            --datasets "${dataset_names[@]}"
    fi
else
    echo "To score:  $ENV/python swift/metrics/total_metric.py --run_name ${run_name} --metric navi_qwen3 --datasets ${dataset_names[*]}"
fi
