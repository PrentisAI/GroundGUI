#!/usr/bin/env python3
"""Build a local, self-contained, transformers-4-loadable mirror of a ScaleCUA SFT checkpoint.

WHY THIS EXISTS -- two independent reasons:

1. THE ROPE BLOCKER. An SFT checkpoint saved by transformers 5.x (e.g. 5.12.1) has
   `text_config.rope_scaling` renamed to `text_config.rope_parameters`, with `rope_theta`
   folded inside it. The RL env runs transformers 4.57.6, whose
   Qwen3VLTextConfig.__init__ never looks for `rope_parameters`, so it leaves
   `self.rope_scaling = None`; Qwen3VLTextRotaryEmbedding.__init__ then guards on
   `config.rope_scaling is not None` (modeling_qwen3_vl.py:283) but unconditionally calls
   `config.rope_scaling.get('mrope_section', ...)` at :297 ->
       AttributeError: 'NoneType' object has no attribute 'get'
   The failure is ASYMMETRIC and therefore a trap: vLLM 0.22.1 has its own
   patch_rope_parameters (transformers_utils/config.py:458-500) and reads the tf-5 form
   happily, so `swift rollout` starts fine and only `swift rlhf` dies. Do not diagnose
   this from the rollout log.

2. THE PRUNING RACE. The source may sit in a training output dir that is still running under
   `--save_total_limit`, so a later save can delete it. A symlink would break; a HARDLINK of
   model.safetensors survives, because rmtree only unlinks. Cost: 0 new bytes for the 17.5 GB
   of weights.

WHY NOT `--rope_scaling '{...}'` ON THE swift rlhf COMMAND LINE. That flag does work
(swift/model/register.py:220-228 injects it into the config before from_pretrained), but it
only patches the TRAINER's in-memory config. It does not reach vLLM, the eval harness,
model_merger, or any future consumer that reads config.json off disk. Fixing the file fixes
every consumer at once.

WHAT IS AND IS NOT COPIED
  hardlink : model.safetensors                (17.5 GB, 0 new bytes, deletion-proof)
  copy     : the 10 tokenizer/processor/template sidecars  (~11.5 MB, real copies)
  rewrite  : config.json                       (rope_parameters -> rope_scaling+rope_theta)
  SKIPPED  : global_step*/ (DeepSpeed ZeRO state, ~95 GB for 8B), rng_state_*.pth, scheduler.pt,
             training_args.bin, latest, zero_to_fp32.py -- inert for loading but noise, and
             `latest` in particular makes the dir look resumable when it is not.
  SKIPPED  : tokenizer_config.json.tf5bak. It is a LANDMINE, not a backup: it is the
             transformers-5 emission whose `extra_special_tokens` is a LIST, and loading it
             under 4.57.6 raises `AttributeError: 'list' object has no attribute 'keys'`.
             The ACTIVE tokenizer_config.json is the good tf-4 one and is kept.

NEVER `sed -i` OR `>` THE SOURCE config.json. Other hardlink mirrors of the same checkpoint
may share its inodes, and an in-place edit would silently change every one of them. This
script only ever writes NEW files into its own target directory.
"""
import argparse, hashlib, json, os, shutil, sys

SRC_DEFAULT = None  # no default: pass --src <SFT checkpoint dir>
DST_DEFAULT = None  # no default: pass --dst <new directory>

# Everything a loader needs, and nothing a loader must not see.
SIDECARS = ["generation_config.json", "preprocessor_config.json", "processor_config.json",
            "video_preprocessor_config.json", "chat_template.json", "chat_template.jinja",
            "tokenizer_config.json", "tokenizer.json", "vocab.json", "merges.txt"]
WEIGHTS = ["model.safetensors"]


def repair_config(cfg):
    """tf-5 `rope_parameters` -> tf-4 `rope_scaling` + `rope_theta`. Returns (cfg, [notes])."""
    notes = []
    tc = cfg.get("text_config", {})
    rp = tc.pop("rope_parameters", None)
    if rp is not None:
        rp = dict(rp)
        theta = rp.pop("rope_theta", None)
        tc["rope_scaling"] = rp                      # {mrope_interleaved, mrope_section, rope_type}
        if theta is not None:
            tc["rope_theta"] = theta
        notes.append(f"text_config.rope_parameters -> rope_scaling={rp} + rope_theta={theta}")
    elif "rope_scaling" in tc:
        notes.append("text_config already tf-4 shaped (rope_scaling present); left alone")
    else:
        raise SystemExit("ERROR: text_config has neither rope_parameters nor rope_scaling.")

    # transformers 4.57.6 registers the vision sub-config under 'qwen3_vl', not
    # 'qwen3_vl_vision'. AutoConfig parses it correctly either way (Qwen3VLConfig builds the
    # sub-config by class), but matching the base model removes one gratuitous difference.
    vc = cfg.get("vision_config", {})
    if vc.get("model_type") == "qwen3_vl_vision":
        vc["model_type"] = "qwen3_vl"
        notes.append("vision_config.model_type qwen3_vl_vision -> qwen3_vl")

    cfg["transformers_version"] = "4.57.6"
    notes.append("transformers_version stamped 4.57.6 (this file is now a tf-4 config)")
    return cfg, notes


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--src", required=True, help="SFT checkpoint dir to mirror")
    ap.add_argument("--dst", required=True, help="new directory to write the transformers-4.57 copy to")
    ap.add_argument("--force", action="store_true",
                    help="rebuild even if --dst already exists (removes only files this script writes)")
    a = ap.parse_args()

    if not os.path.isdir(a.src):
        raise SystemExit(f"ERROR: source checkpoint not found: {a.src}")
    if os.path.exists(a.dst) and os.listdir(a.dst) and not a.force:
        print(f"{a.dst} already exists and is not empty -- nothing to do (use --force to rebuild).")
        return 0
    os.makedirs(a.dst, exist_ok=True)

    # ---- weights: hardlink, never copy -------------------------------------------------
    for f in WEIGHTS:
        s, d = os.path.join(a.src, f), os.path.join(a.dst, f)
        if not os.path.exists(s):
            raise SystemExit(f"ERROR: missing {s}")
        if os.path.exists(d):
            os.unlink(d)
        try:
            os.link(s, d)
            how = "hardlink"
        except OSError as e:                      # different filesystem -> fall back to a copy
            shutil.copy2(s, d)
            how = f"copy (hardlink failed: {e})"
        print(f"  {how:>10}  {f}  ({os.path.getsize(d)/2**30:.2f} GiB)")

    # ---- sidecars: real copies, so nothing here shares an inode with the source ---------
    for f in SIDECARS:
        s, d = os.path.join(a.src, f), os.path.join(a.dst, f)
        if not os.path.exists(s):
            print(f"  {'SKIP':>10}  {f} (absent in source)")
            continue
        if os.path.exists(d):
            os.unlink(d)
        shutil.copy2(s, d)                        # copy2 on a hardlinked source makes a NEW inode
        print(f"  {'copy':>10}  {f}")

    # ---- config.json: rewritten, never linked ------------------------------------------
    with open(os.path.join(a.src, "config.json"), encoding="utf-8") as fh:
        cfg = json.load(fh)
    cfg, notes = repair_config(cfg)
    out = os.path.join(a.dst, "config.json")
    if os.path.exists(out):
        os.unlink(out)                            # break any hardlink BEFORE writing
    with open(out, "w", encoding="utf-8") as fh:
        json.dump(cfg, fh, indent=2, ensure_ascii=False)
        fh.write("\n")
    print(f"  {'rewrite':>10}  config.json")
    for n in notes:
        print(f"             - {n}")

    # ---- args.json: the ONLY thing that makes `swift rollout` work on this dir ----------
    # `--model_type qwen3_vl` on the command line is NOT sufficient. swift resolves model_type
    # from the directory BASENAME first; the default --dst basename matches no registered model, so
    # it falls back to `architectures`, and Qwen3VLForConditionalGeneration maps to THREE types
    # (qwen3_vl / qwen3_vl_emb / qwen3_vl_reranker) -> ValueError.
    #
    # Every other entry point passes model_type down and is fine. The rollout server is not:
    # rollout.py:255 constructs the engine WITH a pre-built template, and vllm_engine.py:141-146
    # takes the `else` branch, which calls
    #     get_model_info_meta(model_id_or_path, hub_token=..., use_hf=..., revision=...)
    # -- dropping model_type entirely. Symptom: the trainer banner is correct, but the rollout
    # server dies at startup and the job exits before step 1.
    #
    # _get_model_info's own escape hatch (model_meta.py:222-224) is to read `args.json` from the
    # model dir -- the file swift writes into its own output checkpoints. Providing it fixes the
    # rollout server, the trainer and every eval at once, with ZERO edits to the swift tree.
    #
    # `swift_version` is LOAD-BEARING, not decoration: load_args_from_ckpt (base_args.py:271-273)
    # does `if swift_version is None or < '4.0.0.dev': load_keys.remove('model_type')`, i.e. it
    # SILENTLY IGNORES model_type without it, and you get the identical ValueError back.
    #
    # Keep this file minimal. Its presence makes swift treat the dir as a checkpoint
    # (model/utils.py:334 tests only for args.json) and merge these keys into the runtime args;
    # `model_type` is in `load_keys`, which only applies when the current value is None, so an
    # explicit --model_type still wins. Adding training keys here would leak them into every run.
    args_json = {"model_type": "qwen3_vl", "swift_version": "4.0.0.dev0"}
    ap_path = os.path.join(a.dst, "args.json")
    if os.path.exists(ap_path):
        os.unlink(ap_path)
    with open(ap_path, "w", encoding="utf-8") as fh:
        json.dump(args_json, fh, indent=2)
        fh.write("\n")
    print(f"  {'write':>10}  args.json  {args_json}")

    # ---- provenance --------------------------------------------------------------------
    with open(os.path.join(a.src, "config.json"), "rb") as fh:
        src_md5 = hashlib.md5(fh.read()).hexdigest()
    prov = {
        "source": a.src,
        "source_config_md5": src_md5,
        "weights": "hardlinked from source (survives the source run's --save_total_limit pruning)",
        "config_repair": notes,
        "args_json": ("written so `swift rollout` can resolve model_type: vllm_engine.py's "
                      "template!=None branch drops model_type, and this dir's basename matches "
                      "no registered model. swift_version is required or model_type is ignored."),
        "skipped": ["global_step*/", "latest", "rng_state_*.pth", "scheduler.pt",
                    "trainer_state.json", "training_args.bin", "zero_to_fp32.py",
                    "tokenizer_config.json.tf5bak"],
        "note": ("Coordinate space is 0-1000 per-mille (the SFT ran with coord_norm=True). "
                 "System prompt is internvl2_5_desktop_grounding_v1, md5 "
                 "bc0d22f21d3e1b5c451e7bf03ce49cf6; it is NOT baked into the chat template -- "
                 "it must arrive in the dataset row's messages[0]."),
    }
    with open(os.path.join(a.dst, "MIGRATION_PROVENANCE.json"), "w", encoding="utf-8") as fh:
        json.dump(prov, fh, indent=2)

    print(f"\nOK -> {a.dst}")
    print("Verify with:  Grounding_scripts/preflight.py --model", a.dst)
    return 0


if __name__ == "__main__":
    sys.exit(main())
