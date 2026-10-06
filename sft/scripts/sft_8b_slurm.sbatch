#!/usr/bin/env bash
# Slurm wrapper around scripts/sft_local.sh (Qwen3-VL-8B SFT, single node).
# Sets up the Python env, runs preflight checks, then calls the inner launcher, which
# starts torchrun itself (NNODES=1, MASTER_ADDR=127.0.0.1). Slurm therefore allocates a
# single task and torchrun forks one rank per GPU.
#
# Usage:
#   CONDA_ENV=<sft env prefix> sbatch --export=ALL scripts/sft_8b_slurm.sbatch <meta.json> <output_dir> <pretrained>
#   More GPUs: override --gres together with CPU/memory, e.g.
#     sbatch --gres=gpu:8 --cpus-per-task=128 --mem=2800G --export=ALL scripts/sft_8b_slurm.sbatch ...
#
# Required env vars:
#   CONDA_ENV   prefix of the SFT env (conda env or venv; see sft/requirements.lock.txt)
# Optional env vars: everything sft_local.sh reads (MAX_GRAD_NORM, GRAD_NORM_PROBE,
#   GRAD_NORM_SKIP_MULT, PACK_CARRY, ...), plus CUDA_HOME. Pass them with --export=ALL,...;
#   without ALL the job environment is replaced and PATH is lost. Example:
#     sbatch --export=ALL,MAX_GRAD_NORM=30 scripts/sft_8b_slurm.sbatch ...
#   The "[grad-norm] patch installed: ..." log line shows the values in effect.
#
# Submit from sft/ (all paths are relative to it). Adjust --partition / resources below to
# your cluster.
#
# Resuming: no extra switch. If output_dir already contains checkpoint-*, train_qwen.py calls
#   trainer.train(resume_from_checkpoint=True) and restores optimizer, LR schedule, global
#   step and data position. Resubmit with the same three arguments and the same number of
#   GPUs: ZeRO-3 checkpoints are sharded per rank, and the data position is restored by
#   skipping already-consumed samples, which is only meaningful with the same meta file.
#   Look for "checkpoint found, resume training" in the .out file. Count the shards with
#     ls <ckpt>/global_step*/ | grep -oP 'zero_pp_rank_\d+' | sort -u | wc -l
#
#SBATCH --job-name=groundgui-qwen3vl8b-sft
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:4
# Host RAM: annotations are loaded once per rank, so memory grows with the GPU count.
#SBATCH --cpus-per-task=64
#SBATCH --mem=1400G
# Set --time explicitly: without it the scheduler cannot backfill the job. When the limit
# is reached the job is killed without a final save; resubmit to resume (see above).
#SBATCH --time=7-00:00:00
#SBATCH --output=work_dirs/slurm-%j.out
#SBATCH --error=work_dirs/slurm-%j.err

set -euo pipefail

mkdir -p work_dirs

# Slurm restricts CUDA_VISIBLE_DEVICES to the allocated GPUs; torchrun must start exactly
# that many ranks.
GPUS="${SLURM_GPUS_ON_NODE:-4}"

# Derive the rendezvous port from the job id so that concurrent jobs on the same node do
# not collide on the inner script's default port.
MASTER_PORT=$(( 20000 + SLURM_JOB_ID % 20000 ))

echo "[$(TZ=Asia/Shanghai date '+%Y-%m-%d %H:%M')] job=${SLURM_JOB_ID} node=$(hostname) GPUS=${GPUS} CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset} MASTER_PORT=${MASTER_PORT} cwd=$(pwd)"

# The env must be given explicitly; there is no fallback to whatever python is on PATH.
CONDA_ENV="${CONDA_ENV:?set CONDA_ENV=<prefix of the sft env, see sft/requirements.lock.txt>}"
if [ ! -x "${CONDA_ENV}/bin/python" ]; then
  echo "ERROR: no python at ${CONDA_ENV}/bin/python (node=$(hostname))" >&2
  exit 1
fi
# Prepending bin/ is equivalent to `conda activate` for an env without activate.d hooks.
export PATH="${CONDA_ENV}/bin:${PATH}"
export CONDA_PREFIX="${CONDA_ENV}"
# CONDA_ENV may be a venv layered on a base env (python -m venv --system-site-packages).
# A venv's bin/ lacks the base env's console scripts such as torchrun, so without this
# torchrun would resolve to some other python on PATH. Take the base bin from pyvenv.cfg
# (home=) and put it right after the venv bin.
if [ -f "${CONDA_ENV}/pyvenv.cfg" ]; then
  _venv_home=$(sed -n 's/^home *= *//p' "${CONDA_ENV}/pyvenv.cfg" | head -1)
  if [ -n "${_venv_home}" ] && [ -d "${_venv_home}" ]; then
    export PATH="${CONDA_ENV}/bin:${_venv_home}:${PATH}"
    echo "[env] venv overlay detected, base bin appended: ${_venv_home}"
  fi
fi
# torchrun must resolve inside CONDA_ENV (or its venv base). A torchrun from another
# install runs a different python and fails later with misleading import errors.
_torchrun=$(command -v torchrun || true)
case "${_torchrun}" in
  "${CONDA_ENV}"/*|"${_venv_home:-${CONDA_ENV}}"/*) echo "[env] torchrun=${_torchrun}" ;;
  *) echo "ERROR: torchrun resolves to '${_torchrun:-NOT FOUND}' (node-local or missing), refusing to launch" >&2
     exit 1 ;;
esac
# Ignore ~/.local user site-packages: they take precedence over the env's site-packages,
# so a stale package there (e.g. a broken pandas) silently shadows the env's copy.
export PYTHONNOUSERSITE=1

# Preflight: verify the dependency stack before starting torchrun, so a broken env fails
# here instead of after all ranks are up (or, worse, trains silently wrong).
python - <<'PYCHK' || { echo "ERROR: preflight failed on $(hostname), refusing to launch" >&2; exit 1; }
# Keep this light: the transformers symbols that correctness depends on are really
# imported; flash_attn / deepspeed are only checked by package metadata, because importing
# deepspeed can be slow and torchrun will import it anyway.
import importlib.metadata as md
import transformers, torch
# pandas / datasets are imported for real: train_qwen.py -> trainer.py imports datasets,
# which imports pandas, and this is where a user-site copy would shadow the env's.
import pandas, datasets                                            # noqa: F401
assert '/.local/' not in pandas.__file__, \
    f"pandas is imported from user site-packages: {pandas.__file__} (PYTHONNOUSERSITE did not take effect)"
from transformers import Qwen3VLForConditionalGeneration          # noqa: F401
from transformers.modeling_flash_attention_utils import _is_packed_sequence   # noqa: F401
from transformers.masking_utils import find_packed_sequence_indices           # noqa: F401
fa, ds = md.version('flash_attn'), md.version('deepspeed')
major = int(transformers.__version__.split('.')[0])
assert major >= 5, (
    f"transformers {transformers.__version__} too old; the qwen3vl packing path in "
    "FlattenedDataCollatorForSupervisedDataset (4-row position_ids + attention_mask=None) "
    "needs the 5.x native packed-sequence detection, otherwise cross-document attention "
    "is NOT blocked and training is silently wrong."
)
print(f"[preflight] transformers {transformers.__version__} torch {torch.__version__} "
      f"flash_attn {fa} deepspeed {ds} pandas {pandas.__version__} OK")
print(f"[preflight] pandas from {pandas.__file__}")
PYCHK

# The inner script defaults CUDA_HOME to /usr/local/cuda-13.2. torch ships its own CUDA
# runtime, so CUDA_HOME only affects toolchain lookups (e.g. ptxas); pick one that exists
# rather than putting a missing directory on PATH.
if [ ! -d "${CUDA_HOME:-/usr/local/cuda-13.2}" ]; then
  for _cu in /usr/local/cuda-13.2 /usr/local/cuda; do
    [ -d "${_cu}" ] && export CUDA_HOME="${_cu}" && break
  done
fi

echo "[env] ${CONDA_ENV} | python=$(python -V 2>&1) | transformers=$(python -c 'import transformers;print(transformers.__version__)') | CUDA_HOME=${CUDA_HOME:-/usr/local/cuda-13.2}"

GPUS="${GPUS}" MASTER_PORT="${MASTER_PORT}" bash scripts/sft_local.sh "$@"
