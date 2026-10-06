"""HuggingFace backend for **allenai/MolmoPoint-GUI-8B** (`MolmoPointForConditionalGeneration`).

Why not vLLM
============
Two independent reasons, either of which rules vLLM out:

1. **The architecture is not in the registry.** config.json's architectures is
   `MolmoPointForConditionalGeneration`, while the vLLM 0.19.1 model registry (323 entries)
   only has `Molmo2ForConditionalGeneration` and `MolmoForCausalLM`, neither of which is
   this model. The bundled `modeling_molmo_point.py` can only be loaded through
   transformers' trust_remote_code; vLLM would need its own reimplementation, which does
   not exist.

2. **It does not emit text coordinates.** This is independent of vLLM: other models emit
   `{"x":956,"y":78}` / `<tool_call>{...}</tool_call>` and a regex extracts the
   coordinates. MolmoPoint encodes points as **special tokens**; turning them into pixels
   requires three pieces of metadata that the preprocessor produces at **input time**
   (token_pooling / subpatch_mapping / image_sizes), decoded via
   `model.extract_image_points()`, and generation must be constrained with
   `build_logit_processor_from_inputs()` to keep point tokens valid. eval.py's
   "regex over the output string" path cannot accommodate this.

So it runs in-process with HF.
Generation is one sample at a time and slower than vLLM, but prompts and scoring are
unchanged, so results are comparable with the other models in output/.

Coordinate space (critical, do not change)
==========================================
`extract_image_points` returns `[object_id, image_ix, x, y]`, where

    x = (p_x / mapping.shape[1]) * w        # (w, h) from metadata["image_sizes"]

i.e. **the pixel space of the image we pass in**. eval.py has already resized the image
according to MAX_IMAGE_PIXELS/IMAGE_FACTOR and scaled the GT box into the same space, so
the two line up by construction:

  * Do **not** set COORD_SPACE=norm1000 -- this is not 0-1000 space; setting it divides
    pixel values by 1000 again and every predicted point collapses to the top-left corner.
  * The matching `molmopoint` prompt processor's calculate_metrics does **no scaling**; it
    uses BasicPrompt's pixel-space containment test directly.
  * Do **not** pass max_pixels to the model to let it resize on its own either -- the
    predicted points would then land in a different space.

dtype
=====
config.json's dtype is **float32** (33 GB on disk). The model card's inference example uses
`dtype="auto"` (i.e. fp32 weights) + `torch.autocast("cuda", bfloat16)`, and this backend
copies that exactly, because pure bf16 weights are not numerically equivalent to
"fp32 weights + bf16 autocast" and we want to match the numbers reported on the model card.
33 GB fits comfortably on a single H200 (143 GB). Set MOLMO_DTYPE=bfloat16 to save memory,
but the result is then no longer the official setting.

Abstention / the 54 OSWorld-G refusal samples
=============================================
MolmoPoint is a dedicated pointing model; the model card states it "will output a single
point" and there is **no abstention protocol**. `no_more_points_class` in the config is the
"done pointing" stop class for multi-point tasks, not "this task cannot be done" -- treating
it as abstention would be over-interpreting it. So "no point could be decoded" is recorded
as **no answer** (None), not as the refusal sentinel [-1,-1]:

  => all 54 OSWorld-G refusal samples are misses, ceiling 510/564 = 90.4%, the same
     situation as Holo2. Comparing against models that do have an abstain action
     (e.g. GUI-Owl) on the 564 denominator is unfair; compare on the 510-sample subset.
"""

import os
import re
from typing import Any, Dict, List, Optional

import torch
from PIL import Image
from tqdm import tqdm
from transformers import AutoModelForImageTextToText, AutoProcessor


class MolmoPoint:
    def __init__(
        self,
        model_path: str = "",
        max_new_tokens: Optional[int] = None,
    ):
        # A point is only a few special tokens; 200 is the model card's value, ample headroom.
        self.max_new_tokens = max_new_tokens if max_new_tokens else 200
        self.debug_mode = False

        dtype = os.environ.get("MOLMO_DTYPE", "auto").strip() or "auto"
        if dtype != "auto":
            dtype = getattr(torch, dtype)

        self.model = AutoModelForImageTextToText.from_pretrained(
            model_path,
            trust_remote_code=True,
            dtype=dtype,
            device_map="auto",
        ).eval()
        self.processor = AutoProcessor.from_pretrained(
            model_path,
            trust_remote_code=True,
            padding_side="left",
        )

    @staticmethod
    def _to_molmo_messages(messages: List[Dict[str, Any]], image: Image.Image) -> List[Dict]:
        """eval.py's {'role','content':str} -> Molmo's {'role','content':[{type..}]}.

        eval.py marks the image insertion point with an '<image>' placeholder. The
        **original order is preserved**: MolmoPointPrompt sends f"{instruction}<image>",
        i.e. text first, then image, matching the content order of the model card example;
        the chat_template renders content in order.
        """
        rgb = image.convert("RGB")
        out = []
        for m in messages:
            text = m["content"]
            content: List[Dict[str, Any]] = []
            if "<image>" in text:
                head, tail = text.split("<image>", 1)
                if head.strip():
                    content.append({"type": "text", "text": head})
                content.append({"type": "image", "image": rgb})
                text = tail
            if text.strip() or not content:
                content.append({"type": "text", "text": text})
            out.append({"role": m["role"], "content": content})
        return out

    @torch.inference_mode()
    def chat(self, messages: List[Dict[str, Any]], image: Image.Image) -> str:
        mm = self._to_molmo_messages(messages, image)

        inputs = self.processor.apply_chat_template(
            mm,
            tokenize=True,
            add_generation_prompt=True,
            return_tensors="pt",
            return_dict=True,
            padding=True,
            return_pointing_metadata=True,
        )
        metadata = inputs.pop("metadata")
        # Build the logits processor BEFORE .to(cuda): it reads image_token_pooling, which
        # is a numpy array and would be lost/altered by the device transfer below.
        logits_processor = self.model.build_logit_processor_from_inputs(inputs)
        inputs = {k: (v.to(self.model.device) if hasattr(v, "to") else v)
                  for k, v in inputs.items()}

        with torch.autocast("cuda", dtype=torch.bfloat16):
            output = self.model.generate(
                **inputs,
                logits_processor=logits_processor,
                max_new_tokens=self.max_new_tokens,
                do_sample=False,
            )

        gen = output[:, inputs["input_ids"].size(1):]
        text = self.processor.post_process_image_text_to_text(
            gen, skip_special_tokens=False, clean_up_tokenization_spaces=False
        )[0]

        points = self.model.extract_image_points(
            text,
            metadata["token_pooling"],
            metadata["subpatch_mapping"],
            metadata["image_sizes"],
        )

        if self.debug_mode:
            print("Molmo raw:", repr(text)[:300])
            print("points:", points, "| image size:", image.size)
            print("=" * 50)

        if not points:
            # No decodable point = no answer. Not mapped to the refusal sentinel; see module docstring.
            return ""
        # [object_id, image_ix, x, y]; take the first point. x/y are already in input-image pixels.
        _, _, x, y = points[0]
        return '{"x": %.4f, "y": %.4f}' % (float(x), float(y))

    def get_responses(self, args, messages_list, images):
        self.debug_mode = getattr(args, "debug_mode", 0) == 1
        res = []
        for messages, image in tqdm(zip(messages_list, images), total=len(messages_list),
                                    desc="Generating responses"):
            try:
                res.append(self.chat(messages, image))
            except Exception as e:
                # One failing sample must not abort the run: record an empty string (scored
                # as "no coordinate"), so failed samples are visible in the result file.
                print(f"[molmopoint] sample failed: {type(e).__name__}: {e}")
                res.append("")
        return res
