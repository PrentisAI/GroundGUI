#!/bin/bash
# OPSD+GRPO arm defaults -- thin wrapper around train_grpo.sh (RSTG, arXiv:2608.00782):
# routing on negative zero-variance (all-miss) groups plus a GT tool-call SFT term, optionally
# the A_OPD privileged-teacher term.
#
# Chain:  submit.sh <arm> -> train_opsd_grpo.sbatch -> train_opsd_grpo.sh (this) -> train_grpo.sh
#
# Run by the sbatch under srun; it only sets defaults (every one is ${VAR:-default}, so the
# arm's --export env wins) and execs train_grpo.sh. By hand, on a full 8-GPU node:
#   ENV_BIN=<rl env>/bin bash Grounding_scripts/opsd+grpo/train_opsd_grpo.sh
#
# This is a WRAPPER, not a fork: all of the recipe lives in train_grpo.sh, and with
# USE_OPSD_GRPO=false the command line is the plain RL recipe, so the arms differ only in the
# flags below. Compare arms only across runs with the same init, corpus and prompt, and
# identical BS/GAS/NUM_GEN/LR/TEMPERATURE.
#
# Memory: the SFT term adds one forward+backward per accumulation chunk on a sequence with
# ~10k image tokens. On OOM halve BS and double GAS (same effective batch and step count).
#
# Watch nzg/frac_groups, nzg/sft_rows, nzg/sft_loss and nzg/format_match in tensorboard; if
# format_match falls below ~0.9 the policy has drifted off the tool-call shape the SFT target
# assumes. frac_reward_zero_std over the first steps shows how much of each batch is
# zero-variance -- the problem this method addresses.
set -euo pipefail
cd "$(dirname "$(readlink -f "$0")")/../.."

# Topology defaults for a full 8-GPU node (1 rollout + 7 trainers); the sbatch derives and
# exports them from the actual allocation, so these apply only when run by hand.
export GPU_ROLLOUT="${GPU_ROLLOUT:-7}"
export GPUS_TRAINER="${GPUS_TRAINER:-0,1,2,3,4,5,6}"
export BS="${BS:-2}"
# GAS 24 sets the effective batch: BS 2 x 7 trainers x GAS 24 / NUM_GEN 8 = 42 prompts/step.
# With Adam the per-step update is ~LR regardless of batch, so the number of optimizer steps is
# what drives both progress and entropy collapse; a larger batch gives a quieter per-step
# gradient. GAS changes the number of micro-batches, not their size, so memory is unaffected.
# swift defaults steps_per_generation to GAS, so its equality assert holds at any GAS.
export GAS="${GAS:-24}"
export NUM_GEN="${NUM_GEN:-8}"
# Dense checkpoints (every 25 steps, ~17 GiB each): RL accuracy can peak mid-run and then
# decay, so the eval stage reads a curve, not just the last checkpoint.
export SAVE_STEPS="${SAVE_STEPS:-25}"
export DL_WORKERS="${DL_WORKERS:-12}"

# Binary in-box reward ONLY. The NZG branch is defined on the GroundAcc column; a continuous
# term (g2acc) would let all-miss groups keep reward variance and break the routing.
export REWARD_FUNCS="${REWARD_FUNCS:-ground-acc}"        # binary only: see the NZG guard in train_grpo.sh
export REWARD_WEIGHTS="${REWARD_WEIGHTS:-1.0}"      # must stay positive: the trainer refuses weight<=0 on GroundAcc
export SCALE_REWARDS="${SCALE_REWARDS:-group}"
export DYNAMIC_SAMPLE=false

# ${VAR:-default}, never a bare export: the `rloo` arm passes
# USE_OPSD_GRPO=false via --export, and a hard-coded value here would silently override it.
export USE_OPSD_GRPO="${USE_OPSD_GRPO:-true}"
export NZG_BETA_PEAK="${NZG_BETA_PEAK:-0.005}"     # RSTG beta_init = 5e-3
export NZG_BETA_VALLEY="${NZG_BETA_VALLEY:-0.001}" # RSTG beta_min  = 1e-3
export NZG_WARMUP="${NZG_WARMUP:-0}"
export NZG_DECAY="${NZG_DECAY:-0}"                 # 0 + warmup 0 => constant beta
export NZG_ALL_ROLLOUTS="${NZG_ALL_ROLLOUTS:-false}"
# Per-group routing log -> <OUT_DIR>/nzg_routing.jsonl. Keep ON for any run you intend to
# analyse: it is the only record of which prompts the branch fired on, and cannot be rebuilt.
export NZG_LOG_ROUTING="${NZG_LOG_ROUTING:-true}"

# ---- A_OPD (stage 2) ------------------------------------------------------------
# OFF by default (peak 0). NZG_OPD_BETA_PEAK>0 adds the privileged-teacher reverse-KL term on
# the same routed rows. An ABLATION: in all-miss groups the GT centre's digits rarely appear at
# their own positions in any rollout, so reverse KL has little to re-rank. Set NZG_BETA_PEAK=0
# alongside for an OPD-only arm (no SFT term).
export NZG_OPD_BETA_PEAK="${NZG_OPD_BETA_PEAK:-0}"
export NZG_OPD_BETA_VALLEY="${NZG_OPD_BETA_VALLEY:-0}"
export NZG_OPD_WARMUP="${NZG_OPD_WARMUP:-0}"
export NZG_OPD_DECAY="${NZG_OPD_DECAY:-0}"
export NZG_OPD_ALL_ROLLOUTS="${NZG_OPD_ALL_ROLLOUTS:-false}"
export NZG_ROUTE_ALL="${NZG_ROUTE_ALL:-false}"
# Teacher = the RL init with privileged input (masked image + hint), no EMA. It must be the
# same model as the policy init, so the privilege comes from the input, not from a capability
# gap; a different model would also not share the init's prompt and per-mille coordinates.
export TEACHER_MODEL="${TEACHER_MODEL:-}"   # required by the OPSD term; train_grpo.sh checks
# ZeRO-2, not ZeRO-3: a ZeRO-3 teacher all-gathers inside its forward, and the OPD forward is
# rank-conditional, so the step would hang (details in train_grpo.sh).
export TEACHER_DEEPSPEED="${TEACHER_DEEPSPEED:-zero2}"
# `hint`, not `hint_minimal`: see the note in train_grpo.sh.
export OPSD_HINT_MODE="${OPSD_HINT_MODE:-hint}"
export OPSD_MASK_MODE="${OPSD_MASK_MODE:-gaussian}"

export OUT_DIR="${OUT_DIR:-output/opsd+grpo}"
export RUN_TAG="${RUN_TAG:-gcua-run}"

exec bash Grounding_scripts/train_grpo.sh
