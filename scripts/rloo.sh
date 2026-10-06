#!/usr/bin/env bash
# RLOO (rl/), trained on this machine (all its GPUs: 1 for rollouts, the rest for training).
#
# Usage:
#   ENV_BIN=<rl env>/bin DATASET=<rl corpus .jsonl> scripts/rloo.sh <checkpoint> <run_name>
#     checkpoint  the policy to start from: an SFT checkpoint (converted once into
#                 rl/models/<run>-<checkpoint>-tf4 by prepare_ckpt.py) or an RL checkpoint;
#                 for D2RL, the checkpoint-200 of a finished opsd.sh run
#     run_name    run directory rl/output/opsd+grpo/<run_name>/
#
# One epoch is 300 optimizer steps with a checkpoint every 25 steps; the paper reports
# checkpoint-200. Evaluate with scripts/eval.sh.
set -euo pipefail
source "$(dirname "$0")/_common.sh"

[ "$#" -eq 2 ] || { sed -n '2,12p' "$0"; exit 2; }
need ENV_BIN "<rl env>/bin, see rl/requirements.lock.txt"
need DATASET "<absolute path of the RL corpus .jsonl>"
INIT="$(rl_init "$1")"
export MODEL="$INIT" BASE_MODEL="$INIT" TEACHER_MODEL="${TEACHER_MODEL:-$INIT}"
run_arm rloo "$2"
