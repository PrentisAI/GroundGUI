#!/usr/bin/env bash
# RLOO (rl/), trained on this machine (all its GPUs: 1 for rollouts, the rest for training).
#
# Usage:
#   ENV_BIN=<rl env>/bin DATASET=<rl corpus> scripts/rloo.sh <checkpoint> <run_name>
#     checkpoint  the policy to start from: an SFT checkpoint (converted once into
#                 rl/models/<run>-<checkpoint>-tf4 by prepare_ckpt.py) or an RL checkpoint;
#                 for D2RL, a checkpoint-N of a finished opsd.sh run
#     run_name    run directory rl/output/opsd_rloo/<run_name>/
#   DATASET may be the downloaded Parquet release; it is converted once into rl/data/.
#
# One epoch is 300 optimizer steps with a checkpoint every 25 steps (checkpoint-N).
# Evaluate with scripts/eval.sh.
set -euo pipefail
source "$(dirname "$0")/_common.sh"

[ "$#" -eq 2 ] || { sed -n '2,13p' "$0"; exit 2; }
need ENV_BIN "<rl env>/bin, see rl/requirements.lock.txt"
need DATASET "<the RL corpus: its Parquet release (file or directory) or a .jsonl>"
export DATASET="$(rl_dataset "$DATASET")"
INIT="$(rl_init "$1")"
export MODEL="$INIT" BASE_MODEL="$INIT" TEACHER_MODEL="${TEACHER_MODEL:-$INIT}"
run_arm rloo "$2"
