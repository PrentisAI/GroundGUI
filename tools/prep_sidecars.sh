#!/usr/bin/env bash
# In-place side-car completion for transformers-5.x checkpoints, so that the
# transformers-4.57 eval environment can load them. (tools/build_serve_dir.sh is the
# preferred, non-destructive alternative.)
#
# Usage:
#   SRC=<side-car dir> ROOT=<run output dir> STEPS="1000 2000" \
#   EVAL_PY=<eval env>/bin/python  tools/prep_sidecars.sh
#
#   SRC      tf4.57-format side-car files (tokenizer, chat template, preprocessor config)
#            for the exact base model this run was trained from
#   ROOT     directory containing checkpoint-<step>/ subdirectories
#   STEPS    space-separated list of steps to process
#   EVAL_PY  python of the eval env; used to verify tokenizer+processor load
#
# Why this is needed:
#   - tf5 `save_only_model` does not write preprocessor_config.json.
#   - tf5 writes `extra_special_tokens` in tokenizer_config.json as a list; tf4.57 expects a
#     dict and fails with AttributeError: 'list' object has no attribute 'keys'.
# Files in REQ must exist in SRC; files in OPT are copied only if SRC has them.
# Never mix tokenizer files across runs/base models, otherwise the train-time and eval-time
# vocabularies differ. preprocessor_config.json must correspond to this run's own
# processor_config.json; one copied from another run silently changes image resizing.
# Exit code is non-zero if any check fails, so a wrapper can refuse to start eval.
set -uo pipefail
SRC="${SRC:?}"; ROOT="${ROOT:?}"; STEPS="${STEPS:?}"
REQ="preprocessor_config.json chat_template.json tokenizer.json tokenizer_config.json"
OPT="video_preprocessor_config.json vocab.json merges.txt added_tokens.json special_tokens_map.json processor_config.json"
# Always overwritten from SRC when they differ (tf5 tokenizer_config.json has the list-valued
# extra_special_tokens); the checkpoint's original is kept as <file>.tf5bak.
FORCE="tokenizer_config.json tokenizer.json vocab.json merges.txt"
EVAL_PY="${EVAL_PY:?set EVAL_PY=<eval env>/bin/python}"
fail=0
ok(){ printf '  [%s] %s%s\n' "$([ "$1" = 0 ] && echo PASS || echo FAIL)" "$2" "${3:+  -- $3}"; [ "$1" = 0 ] || fail=1; }
echo "SRC  = $SRC"; echo "ROOT = $ROOT"
for f in $REQ; do [ -f "$SRC/$f" ] || { echo "  ABORT: SRC is missing required file $f"; exit 1; }; done
echo "required side-car files present in SRC ✓"
for st in $STEPS; do
  CK="$ROOT/checkpoint-$st"; echo "=== checkpoint-$st ==="
  [ -d "$CK" ] || { ok 1 "directory exists"; continue; }
  added=""
  for f in $REQ $OPT; do
    [ -f "$SRC/$f" ] || continue
    [ -f "$CK/$f" ] || { cp "$SRC/$f" "$CK/$f" && added="$added $f"; }
  done
  echo "   added:${added:- (none, already complete)}"
  for f in $FORCE; do
    [ -f "$SRC/$f" ] || continue
    if [ -f "$CK/$f" ] && ! cmp -s "$SRC/$f" "$CK/$f"; then
      cp -n "$CK/$f" "$CK/$f.tf5bak" 2>/dev/null || true
      cp -f "$SRC/$f" "$CK/$f"; echo "   overwritten: $f (original kept as $f.tf5bak)"
    fi
  done
  PYTHONNOUSERSITE=1 "$EVAL_PY" - "$CK" <<'PYCHK'
import sys
from transformers import AutoTokenizer, AutoProcessor
ck=sys.argv[1]
t=AutoTokenizer.from_pretrained(ck,trust_remote_code=True)
p=AutoProcessor.from_pretrained(ck,trust_remote_code=True)
print(f"   [eval-env] tokenizer={type(t).__name__} vocab={len(t)} processor={type(p).__name__}")
PYCHK
  ok $? "eval env can load tokenizer+processor"
  miss=""; for f in $REQ model.safetensors config.json; do [ -f "$CK/$f" ] || miss="$miss $f"; done
  [ -z "$miss" ]; ok $? "all files needed by eval present" "${miss:+missing:$miss}"
  # size sanity check against truncated/partial weight files (8B model in bf16)
  sz=$(stat -c%s "$CK/model.safetensors" 2>/dev/null || echo 0)
  [ "$sz" -gt 17000000000 ]; ok $? "weights > 17 GB" "$((sz/1000000)) MB"
  for f in preprocessor_config.json vocab.json merges.txt; do
    [ -f "$SRC/$f" ] || continue
    a=$(sha256sum "$SRC/$f"|cut -c1-16); b=$(sha256sum "$CK/$f"|cut -c1-16)
    [ "$a" = "$b" ] || ok 1 "$f has same hash as SRC" "$a vs $b"
  done
done
echo
[ $fail = 0 ] && echo "SIDE-CAR all passed -> eval may start" || echo "SIDE-CAR failures -> eval will not start"
exit $fail
