# GroundGUI: GUI grounding training and evaluation

Code to train a Qwen3-VL-8B GUI grounding model with SFT and RL (RLOO, on-policy self-distillation
(OPSD), and D2RL = OPSD then RLOO), and to evaluate it on five grounding benchmarks.
Training data: [ScreenRef](https://huggingface.co/datasets/PrentisAI/ScreenRef).

```
sft/      supervised fine-tuning
rl/       RLOO and OPSD (fork of GUI-SD / ms-swift)
eval/     evaluation on ScreenSpot-v2, ScreenSpot-Pro, MMBench-GUI, UI-Vision, OSWorld-G
tools/    checkpoint conversion between the environments
scripts/  one command per stage
```

## Setup

Each stage has its own Python environment; `requirements.lock.txt` in each directory is the exact
`pip freeze` used for the reported results (Python 3.12.13). Do not merge them: `sft/` needs
transformers 5.x (sequence packing silently trains wrong under 4.x), `rl/` and `eval/` use 4.57.
The scripts find the environments through `CONDA_ENV` (sft), `ENV_BIN` (rl, the env's `bin/`) and
`ENV_PREFIX` (eval).

## Run

```bash
# 1. SFT
CONDA_ENV=<sft env> DATA_ROOT=<ScreenRef dir> scripts/sft.sh out/sft <Qwen3-VL-8B-Instruct dir>

# 2. RL, starting from an SFT checkpoint
export ENV_BIN=<rl env>/bin DATASET=<RL corpus .jsonl>
scripts/rloo.sh out/sft/checkpoint-<N> my-rloo           # RLOO
scripts/opsd.sh out/sft/checkpoint-<N> my-opsd           # OPSD
scripts/rloo.sh rl/output/opsd+grpo/my-opsd/checkpoint-200 my-d2rl   # D2RL: RLOO after OPSD

# 3. Evaluate
ENV_PREFIX=<eval env> DATA_ROOT=<benchmark dir> scripts/eval.sh rl/output/opsd+grpo/my-rloo/checkpoint-200
```

Run any script without arguments to see its options.

- The RL scripts train on all GPUs of the machine (one serves rollouts, the rest train). One epoch
  is 300 optimizer steps with a checkpoint every 25 steps; reported results use `checkpoint-200`.
  On Slurm, submit the same job with `rl/Grounding_scripts/opsd+grpo/submit.sh <rloo|opsd> <run_name>`.
- Checkpoint formats are converted automatically: `sft/` saves transformers-5 checkpoints, which the
  RL scripts convert with `rl/Grounding_scripts/prepare_ckpt.py`; `eval.sh` builds a serve directory
  for them with `tools/build_serve_dir.sh` (set `INSTRUCT_DIR=<Qwen3-VL-8B-Instruct dir>`).

## Notes

- **Data.** `sft/configs/train_data_sft.json` documents the SFT data mix and format. Its
  `${DATA_ROOT}/...` paths are illustrative; adapt them to your copy of ScreenRef.
- **OPSD** uses uniform token weighting (`OPSD_TOKEN_WEIGHT_MODE=uniform`, set by `scripts/opsd.sh`).
- **RLOO** runs through ms-swift's GRPO trainer with `scale_rewards=none`: the advantage is
  R_i − mean(R), the leave-one-out advantage times the constant (K−1)/K.
- **Keep `--vllm_enforce_eager true`** on the rollout server (`rl/Grounding_scripts/train_grpo.sh`):
  it changes the outputs, not only the speed, and does not appear in `args.json`.
- **Evaluation** uses vLLM at temperature 0. Scoring is point-in-box in pixel space
  (`GROUNDING_OFFICIAL_SCORING=0`, the default; `=1` is up to ~1.4 pp looser and is not recorded in
  result files). OSWorld-G uses the prompt `scalecua_toolcall_abstain_schema`. `eval/rescore.py`
  re-scores an existing result file without re-running inference.

## License

Apache-2.0. `sft/` is derived from [ScaleCUA](https://github.com/OpenGVLab/ScaleCUA), `rl/` from
GUI-SD and [ms-swift](https://github.com/modelscope/ms-swift), `eval/` from
[GroundCUA](https://github.com/ServiceNow/GroundCUA); see [NOTICE](NOTICE) for attribution and the
list of modified files. ScreenRef is distributed under its own license and is not covered by this
repository's license.
