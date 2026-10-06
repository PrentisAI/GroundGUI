#!/usr/bin/env bash
# Make the starting directory (BASE) for refusal SFT from an RL checkpoint.
#
# Usage: tools/make_sft_base.sh [--copy] <rl_checkpoint_dir> <base_dir>
#   rl_checkpoint_dir  a checkpoint-<step>/ written by rl/ (ms-swift; transformers-4.57 format:
#                      sharded safetensors + index, config, tokenizer and processor files)
#   base_dir           new directory to create. Its basename must contain "qwen3" and "vl"
#                      (case-insensitive): sft/ picks the model class from the
#                      model_name_or_path string, not from config.json.
#   --copy             copy the files instead of symlinking them
#
# By default every file of the checkpoint is symlinked into base_dir, so nothing is copied and
# the checkpoint itself is never modified. The sft/ environment (transformers 5.x) loads the
# 4.57-format checkpoint as-is; no file conversion is needed in this direction.
# Then: BASE=<base_dir> sbatch --export=ALL sft/recipes/refusal/sft_refusal.sbatch
set -euo pipefail

mode=link
if [ "${1:-}" = "--copy" ]; then mode=copy; shift; fi
[ "$#" -eq 2 ] || { echo "usage: $0 [--copy] <rl_checkpoint_dir> <base_dir>" >&2; exit 2; }
src=$(cd "$1" && pwd); dst=$2

case "$(basename "$dst" | tr 'A-Z' 'a-z')" in
  *qwen3*vl*) : ;;
  *) echo "ERROR: basename of base_dir must contain 'qwen3' and 'vl' (got $(basename "$dst"))" >&2; exit 1 ;;
esac
[ -e "$dst" ] && { echo "ERROR: $dst already exists; refusing to overwrite" >&2; exit 1; }

for f in config.json model.safetensors.index.json tokenizer.json tokenizer_config.json \
         preprocessor_config.json chat_template.jinja; do
  [ -f "$src/$f" ] || { echo "ERROR: $src/$f missing (is this an rl/ checkpoint?)" >&2; exit 1; }
done
# every shard named in the index must be present
python3 - "$src" <<'PY'
import json, os, sys
src = sys.argv[1]
shards = sorted(set(json.load(open(f"{src}/model.safetensors.index.json"))["weight_map"].values()))
missing = [s for s in shards if not os.path.isfile(os.path.join(src, s))]
if missing:
    sys.exit(f"ERROR: shards listed in the index are missing: {missing}")
print(f"  {len(shards)} weight shards present")
PY

mkdir -p "$dst"
for f in "$src"/*; do
  [ -f "$f" ] || continue
  if [ "$mode" = copy ]; then cp -p "$f" "$dst/"; else ln -s "$f" "$dst/$(basename "$f")"; fi
done
echo "  $mode: $dst <- $src ($(ls "$dst" | wc -l) files)"
