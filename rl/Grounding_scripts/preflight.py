#!/usr/bin/env python3
"""Pre-launch gate for the GroundCUA OPSD+GRPO pipeline. CPU-only, takes seconds.

Every check here corresponds to a failure that is either SILENT or costs hours of a scarce
multi-GPU allocation before it surfaces. Run it before every submission.

  1  IMPORTED SWIFT IS THE LOCAL COPY.  If the env has ms-swift installed EDITABLE from another
     checkout (e.g. upstream GUI-SD), its sys.meta_path finder is hardcoded to that checkout's
     swift/. `cd`-ing here does NOT shadow it -- the console script sets sys.path[0] to the
     env's bin dir, not to cwd. Only PYTHONPATH does, because the finder APPENDS itself after
     PathFinder. Without the export the other checkout's trainer would run while everything
     looks correct.
  2  MODEL CONFIG LOADS UNDER transformers 4.57.6.  An SFT checkpoint saved by transformers 5
     has a transformers-5 shaped config.json, which raises AttributeError in the rotary
     embedding. vLLM reads it fine, so `swift rollout` starts and only `swift rlhf` dies -- an
     asymmetry that wastes a full startup. Grounding_scripts/prepare_ckpt.py builds the
     repaired mirror.
  3  swift CANNOT auto-resolve model_type for a dir named `checkpoint-*` or any other basename
     that matches no registered model: it matches on the directory BASENAME, falls back to
     `architectures`, and Qwen3VLForConditionalGeneration maps to THREE model types
     (qwen3_vl / qwen3_vl_emb / qwen3_vl_reranker) -> ValueError. `--model_type qwen3_vl` is
     mandatory; this check just proves the flag is still needed.
  4  CORPUS SCHEMA.  The trainer and reward read five columns and they fail at three DIFFERENT
     times: a missing `additional_paras` dies at the first reward call; a missing `image_size`
     inside it dies only when a row first becomes an NZG owner (hours in, uncaught KeyError);
     a missing `sample_id` dies only when the A_OPD branch renders its first mask.
  5  SYSTEM PROMPT md5.  The corpus must carry internvl2_5_desktop_grounding_v1
     (bc0d22f21d3e1b5c451e7bf03ce49cf6, 3511 chars) -- the prompt the SFT checkpoint was
     trained under. Any other system prompt is off-distribution for the policy; a prompt whose
     only coordinate statement is "pixels" is worse still, because the reward scores per-mille.
  6  GT REPLAY.  Score each row's OWN stored assistant turn with the real GroundAcc. Anything
     below 1.0 means rows exist that the policy could never be rewarded for.
  7  COORDINATE SPACE, the highest-risk item and the only SILENT one.  A model emitting
     absolute pixels scores near-zero ground-acc while ground-format stays 1.0 --
     indistinguishable from "the model is just bad", and identical in signature to the vLLM
     DeepStack collapse. This check replays the GT centre re-expressed in each wrong space and
     prints what each would look like, so the number in the training log has a reference.
  8  BENCHMARKS.  The eval must use the _v3 files; the unsuffixed ones carry a different system
     prompt and measure the model off-distribution.

Exit 0 = safe to launch. Exit 1 = do not launch.
"""
import argparse, collections, hashlib, json, os, subprocess, sys, tempfile

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROMPT_MD5 = "bc0d22f21d3e1b5c451e7bf03ce49cf6"
PROMPT_LEN = 3511
REQUIRED = ("solution", "images", "messages", "additional_paras", "sample_id")
BENCHES = ["screenspotv2", "screenspotpro", "mmbench", "osworldg", "osworldg_r"]

ok = True
def check(name, passed, detail=""):
    global ok
    ok &= bool(passed)
    print(f"[{'PASS' if passed else 'FAIL'}] {name}" + (f"  --  {detail}" if detail else ""))

def warn(name, detail=""):
    print(f"[warn] {name}" + (f"  --  {detail}" if detail else ""))


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True, help="checkpoint the run starts from")
    ap.add_argument("--dataset", required=True, help="RL corpus .jsonl")
    ap.add_argument("--bench-suffix", default="_v3")
    ap.add_argument("--replay-rows", type=int, default=3000, help="0 = every row")
    ap.add_argument("--skip-model", action="store_true")
    ap.add_argument("--skip-bench", action="store_true",
                    help="skip the benchmark-file checks (training without the in-job evaluation)")
    a = ap.parse_args()

    print("=" * 78)
    print(f"preflight  repo={REPO}")
    print("=" * 78)

    # ---- 1. the imported swift ---------------------------------------------------------
    # Probe from a SCRATCH DIRECTORY, in a subprocess. Checking `import swift` in THIS process
    # would be vacuous: the sys.path.insert below (and the fact that `python foo.py` puts the
    # script's own directory on sys.path) would resolve the local tree whether or not
    # PYTHONPATH is set -- while the real `swift` console script, whose sys.path[0] is the
    # env's bin/, would still load the other checkout. A probe file in a directory that contains no
    # `swift` reproduces the console script's import context exactly.
    local = os.path.realpath(os.path.join(REPO, "swift"))
    with tempfile.TemporaryDirectory() as td:
        probe = os.path.join(td, "_swift_probe.py")
        with open(probe, "w", encoding="utf-8") as fh:
            fh.write("import os, swift\nprint(os.path.realpath(swift.__path__[0]))\n")
        got = subprocess.run([sys.executable, probe], capture_output=True, text=True,
                             cwd=td).stdout.strip()
    check("`swift` console script would import THIS repo's copy", got == local, got or "<no output>")
    if got != local:
        print(f"       fix: export PYTHONPATH={REPO}")
        print("       (ms-swift is installed EDITABLE with a meta_path finder pinned to")
        print("        another checkout's swift/; cd-ing here does NOT shadow it, only PYTHONPATH does.)")

    # Everything below this line needs the local tree in THIS process too.
    sys.path.insert(0, REPO)
    import swift  # noqa: F401

    from swift.cus_rewards.ground_reward import GroundAcc
    from swift.custom_utils.format_func import extract_action  # noqa: F401

    # ---- 2/3. the model ----------------------------------------------------------------
    if not a.skip_model:
        from transformers import AutoConfig
        try:
            cfg = AutoConfig.from_pretrained(a.model)
            rs = getattr(cfg.text_config, "rope_scaling", None)
            check("model config is transformers-4 shaped (rope_scaling present)",
                  isinstance(rs, dict) and "mrope_section" in rs, str(rs))
            import torch
            from transformers import Qwen3VLForConditionalGeneration
            with torch.device("meta"):
                m = Qwen3VLForConditionalGeneration(cfg)
            check("model instantiates under transformers 4.57.6",
                  True, f"{sum(p.numel() for p in m.parameters()):,} params")
            ds = getattr(cfg.vision_config, "deepstack_visual_indexes", None)
            check("DeepStack indexes present (needs --vllm_enforce_eager true on the rollout)",
                  bool(ds), str(ds))
        except Exception as e:
            check("model instantiates under transformers 4.57.6", False,
                  f"{type(e).__name__}: {e}  -- run Grounding_scripts/prepare_ckpt.py")

        # THE ROLLOUT-SERVER RESOLUTION PATH. This is not the same as "--model_type is passed":
        # rollout.py:255 builds the engine with a pre-made template, so vllm_engine.py:141-146
        # takes its `else` branch and calls get_model_info_meta() WITHOUT model_type. The dir
        # basename matches no registered model, so it falls back to `architectures`, which maps
        # to three types, and raises -- at rollout-server startup, after a correct trainer banner.
        # The fix is args.json in the model dir (model_meta.py:222-224 reads it), and it is only
        # honoured when args.json also carries swift_version >= 4.0.0.dev (base_args.py:271-273
        # silently drops model_type otherwise) -- so test the real call, not the file's presence.
        from swift.model.model_meta import get_matched_model_meta, get_model_info_meta
        if get_matched_model_meta(a.model) is None:
            try:
                info, _ = get_model_info_meta(a.model, download_model=False)
                mt = info.model_type
            except Exception as e:
                mt = f"RAISED {type(e).__name__}: {str(e)[:80]}"
            check("rollout server can resolve model_type without the CLI flag (needs args.json)",
                  mt == "qwen3_vl", f"resolved -> {mt}")
            if mt != "qwen3_vl":
                print("       fix: python Grounding_scripts/prepare_ckpt.py --force")
        else:
            check("model dir basename auto-matches a registered model", True, "no args.json needed")

        for f in ("model.safetensors", "tokenizer.json", "chat_template.json", "preprocessor_config.json"):
            p = os.path.join(a.model, f)
            check(f"model dir has {f}", os.path.exists(p),
                  f"{os.path.getsize(p)/2**20:.1f} MiB" if os.path.exists(p) else "MISSING")
        if os.path.exists(os.path.join(a.model, "tokenizer_config.json.tf5bak")):
            warn("tokenizer_config.json.tf5bak present in the model dir",
                 "it is a transformers-5 file that CRASHES tf 4.57.6 if ever renamed back; delete it")

    # ---- 4/5/6/7. the corpus -----------------------------------------------------------
    if not os.path.exists(a.dataset):
        check(f"dataset exists: {a.dataset}", False)
        return 0 if ok else 1

    reward = GroundAcc()
    n = miss_col = miss_size = bad_md5 = 0
    # HF datasets loads the corpus through pyarrow, which types each column ONCE from the first
    # rows and then hard-fails on a value of another type: "Column(/sample_id) changed from number
    # to string in row 2". That is a DatasetGenerationError minutes into the job, after the rollout
    # server is already up, and it names only the column -- not the slice that introduced it. It
    # happens whenever a corpus merges rows from two emitters that disagree, as corpus-assembly
    # scripts do. Cheap to catch here; expensive to debug from a job log.
    coltypes = collections.defaultdict(set)
    gt_hits = 0
    px_hits = frac_hits = 0          # what a WRONG coordinate space would score
    with open(a.dataset, encoding="utf-8") as fh:
        for line in fh:
            if a.replay_rows and n >= a.replay_rows:
                break
            row = json.loads(line)
            n += 1
            for k, v in row.items():
                coltypes[k].add(type(v).__name__)
            if any(k not in row for k in REQUIRED):
                miss_col += 1
                continue
            extra = row["additional_paras"]
            extra_d = json.loads(extra) if isinstance(extra, str) else extra
            if "image_size" not in extra_d:
                miss_size += 1
                continue
            if hashlib.md5(row["messages"][0]["content"].encode()).hexdigest() != PROMPT_MD5:
                bad_md5 += 1
            comp = row["messages"][-1]["content"]
            gt_hits += reward([comp], [row["solution"]], [extra])[0]
            # the same click, re-expressed in the two spaces a mis-trained model would use
            w, h = extra_d["image_size"]
            pt = json.loads(comp.split("<tool_call>\n")[1].split("\n</tool_call>")[0])["arguments"]["coordinate"]
            tmpl = '<tool_call>\n{"name": "computer_use", "arguments": {"action": "left_click", "coordinate": [%s, %s]}}\n</tool_call>'
            px_hits += reward([tmpl % (int(pt[0] / 1000 * w), int(pt[1] / 1000 * h))],
                              [row["solution"]], [extra])[0]
            frac_hits += reward([tmpl % (round(pt[0] / 1000, 4), round(pt[1] / 1000, 4))],
                                [row["solution"]], [extra])[0]

    check("corpus rows carry all 5 required columns", miss_col == 0, f"{n} rows, {miss_col} bad")
    mixed = {k: sorted(v) for k, v in coltypes.items() if len(v) > 1}
    check("every column has ONE json type (pyarrow types a column once and then hard-fails)",
          not mixed, "clean" if not mixed else f"MIXED: {mixed}")
    check("every row's additional_paras carries image_size", miss_size == 0, f"{miss_size} bad")
    check(f"system prompt md5 == {PROMPT_MD5} (internvl2_5_desktop_grounding_v1, {PROMPT_LEN} chars)",
          bad_md5 == 0, f"{bad_md5} rows off-prompt")
    check("GT replay: every row's own answer scores ground-acc 1.0",
          gt_hits == n, f"{gt_hits:.0f}/{n} = {gt_hits/max(n,1):.4f}")
    print(f"       coordinate-space reference for the training log:")
    print(f"         0-1000 per-mille (CORRECT, what SFT model emits) -> ground-acc {gt_hits/max(n,1):.4f}")
    print(f"         absolute pixels  (WRONG, format stays 1.0)       -> ground-acc {px_hits/max(n,1):.4f}")
    print(f"         normalised 0-1   (WRONG, format stays 1.0)       -> ground-acc {frac_hits/max(n,1):.4f}")
    print(f"       => if rewards/GroundAcc/mean starts near a WRONG-space value above, the model is")
    print(f"          emitting the wrong SPACE, or the vLLM DeepStack branch is compiled out.")

    # ---- 8. benchmarks -----------------------------------------------------------------
    for b in ([] if a.skip_bench else BENCHES):
        p = os.path.join(REPO, "ground_benchmark", f"{b}{a.bench_suffix}.jsonl")
        if not os.path.exists(p):
            check(f"benchmark {b}{a.bench_suffix}.jsonl", False, "MISSING")
            continue
        with open(p, encoding="utf-8") as fh:
            first = json.loads(fh.readline())
            rows = 1 + sum(1 for _ in fh)
        md5 = hashlib.md5(first["messages"][0]["content"].encode()).hexdigest()
        check(f"benchmark {b}{a.bench_suffix} prompt matches the corpus", md5 == PROMPT_MD5,
              f"{rows} rows, md5 {md5[:8]}")

    print("=" * 78)
    print("PREFLIGHT OK -- safe to launch" if ok else "PREFLIGHT FAILED -- do not launch")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
