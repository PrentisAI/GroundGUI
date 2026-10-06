#!/usr/bin/env bash
# Build a serve directory that the transformers-4.57 eval environment can load from a
# checkpoint saved by transformers 5.x. The checkpoint itself is not modified.
#
# Usage: build_serve_dir.sh <serve_dir> <checkpoint_dir (absolute)> <template_dir>
#   serve_dir       output directory (deleted and recreated)
#   checkpoint_dir  absolute path, because the weights are symlinked, not copied
#   template_dir    the base model this run was fine-tuned from (e.g. Qwen3-VL-8B-Instruct),
#                   in tf4.57 format; supplies the tokenizer files and chat_template.json
#
# Why:
#   - tf5 `save_only_model` does not write preprocessor_config.json. It is generated here
#     from THIS checkpoint's processor_config.json. Never copy one from another run: a
#     mismatch silently changes image resizing.
#   - tf5 writes `extra_special_tokens` in tokenizer_config.json as a list; tf4.57 expects a
#     dict (AttributeError: 'list' object has no attribute 'keys'), so the tokenizer files
#     come from the template dir. Never mix tokenizer files across runs/base models.
#   - The original tf5 files are kept next to them as *.tf5.bak for reference.
set -euo pipefail
d=$1; c=$2; T=$3
rm -rf "$d"; mkdir -p "$d"
cp "$T"/{chat_template.json,merges.txt,vocab.json,tokenizer_config.json,tokenizer.json} "$d"/
cp "$c"/{config.json,generation_config.json,trainer_state.json} "$d"/
python3 - "$c" "$d" <<'PY'
import json,sys
c,d=sys.argv[1],sys.argv[2]
ip=json.load(open(f"{c}/processor_config.json"))["image_processor"]
ip["processor_class"]="Qwen3VLProcessor"
json.dump(ip,open(f"{d}/preprocessor_config.json","w"),indent=2,sort_keys=True)
PY
cp "$c"/chat_template.jinja   "$d"/chat_template.jinja.tf5.bak
cp "$c"/processor_config.json "$d"/processor_config.json.tf5.bak
cp "$c"/tokenizer_config.json "$d"/tokenizer_config.json.tf5.bak
cp "$c"/tokenizer.json        "$d"/tokenizer.json.tf5.bak
ln -sfn "$c"/model.safetensors "$d"/model.safetensors
chmod -R o+rX "$d"
echo "  built $1 <- $c"
