#!/usr/bin/env bash
# Slurm wrapper for the refusal-SFT recipe (recipes/refusal/sft_refusal.sh, Qwen3-VL-8B,
# single node). Same env setup and preflight checks as scripts/sft_8b_slurm.sbatch; the
# inner launcher starts torchrun itself, so Slurm allocates a single task.
#
# Usage (from sft/):
#   CONDA_ENV=<sft env> META=recipes/refusal/meta_refusal.json OUTD=<out> BASE=<model dir> \
#     sbatch --export=ALL recipes/refusal/sft_refusal.sbatch
#
# Required env vars:
#   CONDA_ENV   prefix of the SFT env (conda env or venv; see sft/requirements.lock.txt)
#   META        refusal meta json
#   OUTD        output dir; must not already contain checkpoint-*
#   BASE        model dir to fine-tune (made from the RL checkpoint by tools/make_sft_base.sh); its path must
#               contain "qwen3" and "vl"
# Optional env vars: RECIPE_DIR (recipes/refusal), CUDA_HOME, and everything
#   sft_refusal.sh reads (MAX_STEPS, SAVE_STEPS, RUN_NAME, MAX_GRAD_NORM, ...). Pass them
#   with --export=ALL,...; without ALL the job environment is replaced and PATH is lost.
#
# Adjust --partition / resources below to your cluster.
#
#SBATCH --job-name=groundgui-refusal
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:8
# Host RAM: annotations are loaded once per rank, so memory grows with the GPU count.
#SBATCH --cpus-per-task=128
#SBATCH --mem=2800G
# Set --time explicitly: without it the scheduler cannot backfill the job. When the limit
# is reached the job is killed without a final save.
#SBATCH --time=01:00:00
#SBATCH --output=slurm-%j.out
#SBATCH --error=slurm-%j.err

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

# Paths come only from META / OUTD / BASE; positional arguments are rejected so that a
# typo on the sbatch command line cannot redirect the run.
META="${META:?set META=<refusal meta json>}"
OUTD="${OUTD:?set OUTD=<output dir>}"
BASE="${BASE:?set BASE=<dir made by tools/make_sft_base.sh from the RL checkpoint>}"

if [ "$#" -ne 0 ]; then
  echo "ERROR: this sbatch takes no positional arguments (got $#). Pass paths via META / OUTD / BASE."; exit 1
fi
for _p in "${META}" "${BASE}"; do
  [ -e "${_p}" ] || { echo "ERROR: input does not exist: ${_p}"; exit 1; }
done
# Refuse an OUTD that already has checkpoints: train_qwen.py would silently resume from them
# and overwrite that run's outputs.
if compgen -G "${OUTD}/checkpoint-*" > /dev/null 2>&1; then
  echo "ERROR: ${OUTD} already contains checkpoint-*; refusing to start (prevents overwriting / silently resuming a run)."; exit 1
fi
# train_qwen.py picks the model class from the model_name_or_path string (it does not read
# config.json). A path without "qwen3" and "vl" silently takes the Qwen2-VL branch and then
# crashes on vision-tower shape mismatches.
case "$(echo "${BASE}" | tr 'A-Z' 'a-z')" in
  *qwen3*vl*) : ;;
  *) echo "ERROR: BASE path does not contain both 'qwen3' and 'vl'. train_qwen.py picks the model class from the model_name_or_path string (not config.json); without them it silently takes the Qwen2-VL branch and crashes on vision-tower shapes: ${BASE}"; exit 1 ;;
esac
echo "[refusal] META=${META}"
echo "[refusal] OUT =${OUTD}"
echo "[refusal] BASE=${BASE}"

GPUS="${GPUS}" MASTER_PORT="${MASTER_PORT}" bash "${RECIPE_DIR:-recipes/refusal}/sft_refusal.sh" "${META}" "${OUTD}" "${BASE}"
