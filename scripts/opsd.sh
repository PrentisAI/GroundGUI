#!/usr/bin/env bash
# On-policy self-distillation, OPSD (rl/), trained on this machine (all its GPUs: 1 for rollouts,
# the rest for training).
#
# Usage:
#   ENV_BIN=<rl env>/bin DATASET=<rl corpus .jsonl> scripts/opsd.sh <checkpoint> <run_name>
#     checkpoint  the SFT checkpoint to start from (converted once into rl/models/<run>-<checkpoint>-tf4 by
#                 prepare_ckpt.py); it is also the privileged teacher, as in the paper
#                 (override with TEACHER_MODEL=<dir>)
#     run_name    run directory rl/output/opsd+grpo/<run_name>/
#
# One rollout per prompt (NUM_GEN=1, GAS=3, ALLOW_SINGLE_GEN=true): the OPSD term distils one
# rollout per prompt, so sampling more would only cost time. One epoch is 300 optimizer steps with a
# checkpoint every 25 steps; the paper reports checkpoint-200. Evaluate with scripts/eval.sh.
set -euo pipefail
source "$(dirname "$0")/_common.sh"

[ "$#" -eq 2 ] || { sed -n '2,15p' "$0"; exit 2; }
need ENV_BIN "<rl env>/bin, see rl/requirements.lock.txt"
need DATASET "<absolute path of the RL corpus .jsonl>"
INIT="$(rl_init "$1")"
export MODEL="$INIT" BASE_MODEL="$INIT" TEACHER_MODEL="${TEACHER_MODEL:-$INIT}"
export NUM_GEN="${NUM_GEN:-1}" GAS="${GAS:-3}" ALLOW_SINGLE_GEN="${ALLOW_SINGLE_GEN:-true}"
run_arm opsd "$2"
