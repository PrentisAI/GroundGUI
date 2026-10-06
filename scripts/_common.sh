# Shared helpers for scripts/*.sh. Sourced, not executed.
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

die() { echo "ERROR: $*" >&2; exit 1; }

need() {  # need VAR "hint": fail unless the environment variable is set
  [ -n "${!1:-}" ] || die "set $1=$2"
}

abspath() { python3 -c 'import os,sys; print(os.path.abspath(sys.argv[1]))' "$1"; }

# render_meta <meta.json> <out.json>: substitute ${DATA_ROOT} (required if the meta uses it)
render_meta() {
  local src="$1" dst="$2"
  [ -f "$src" ] || die "meta not found: $src"
  if grep -q '\${DATA_ROOT}' "$src"; then
    need DATA_ROOT "<directory that holds the training data>"
  fi
  python3 - "$src" "$dst" <<'PY'
import json, os, sys
src, dst = sys.argv[1], sys.argv[2]
text = open(src).read().replace("${DATA_ROOT}", os.environ.get("DATA_ROOT", "").rstrip("/"))
json.loads(text)
open(dst, "w").write(text)
PY
}

# ckpt_format <dir>: prints "tf5" for a checkpoint saved by transformers 5.x (sft/), else "tf4"
ckpt_format() {
  python3 - "$1" <<'PY'
import json, sys
c = json.load(open(sys.argv[1] + "/config.json"))
print("tf5" if str(c.get("transformers_version", "")).startswith("5") else "tf4")
PY
}

# rl_init <checkpoint>: prints a checkpoint the RL stage can load. An SFT checkpoint
# (transformers 5.x) is mirrored once into rl/models/<run>-<checkpoint>-tf4 by prepare_ckpt.py.
rl_init() {
  local src; src="$(abspath "$1")"
  [ -f "$src/config.json" ] || die "not a checkpoint directory: $src"
  if [ "$(ckpt_format "$src")" = tf4 ]; then echo "$src"; return; fi
  local dst="$REPO/rl/models/$(basename "$(dirname "$src")")-$(basename "$src")-tf4"
  if [ ! -f "$dst/config.json" ]; then
    "$ENV_BIN/python" "$REPO/rl/Grounding_scripts/prepare_ckpt.py" --src "$src" --dst "$dst" >&2 \
      || die "prepare_ckpt.py failed for $src"
  fi
  echo "$dst"
}

# run_arm <arm> <run_name>: train one RL arm on this machine -- rl/'s submit.sh with LOCAL=1 runs
# the same job script in the foreground, and TRAIN_ONLY=1 stops after training.
run_arm() {
  LOCAL=1 TRAIN_ONLY="${TRAIN_ONLY:-1}" bash "$REPO/rl/Grounding_scripts/opsd+grpo/submit.sh" "$1" "$2"
}
