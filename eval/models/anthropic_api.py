import os
import io
import re
import time
import base64
from typing import List, Dict, Any, Optional, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed

from tqdm import tqdm
from PIL import Image
import anthropic

# Anthropic silently downscales images whose long edge exceeds ~1568px or whose
# area exceeds ~1.15MP, and the model then reports pixel coordinates in that
# hidden downscaled space. We pre-fit every image *below* those limits and send
# it ourselves, so (a) the model sees exactly the pixels we know about and
# (b) any pixel answer can be normalized by the exact dimensions we sent. Kept
# conservatively under the documented thresholds.
_MAX_LONG_EDGE = 1512
_MAX_AREA = 1_100_000


class AnthropicModel:
    """
    Anthropic (Claude) backend for the grounding eval.

    Mirrors the public surface of the vLLM `Qwen2VL` backend and of
    `OpenAIModel`: exposes `get_responses(args, messages_list, images)` and
    returns a list of raw response strings aligned 1:1 with `messages_list`.

    Each entry of `messages_list` is a prompt-processor output: a `system`
    message plus a `user` message whose `content` contains exactly one
    `<image>` placeholder. We hoist the `system` message to Anthropic's
    top-level `system` parameter and turn the user turn into a content-block
    list (image block + text block(s)), then call the Messages API. The prompt
    processor tells the model the coordinate convention (normalized [0,1]) and
    the model returns a click point in that space, which `eval.py` scales into
    the resized-image / gt-bbox space.

    Concurrency: requests are issued from a thread pool (the SDK client is
    thread-safe). Order is preserved by writing results back by index.
    """

    def __init__(
        self,
        model_name: str,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        max_workers: int = 4,
        thinking_budget: int = 0,
        max_retries: int = 5,
        computer_use: bool = True,
        cua_long_edge: int = 1280,
        allow_refusal: bool = False,
    ):
        self.model_name = model_name
        self.computer_use = computer_use
        self.cua_long_edge = cua_long_edge  # Anthropic degrades coords above ~1280
        self.allow_refusal = allow_refusal  # OSWorld-G: model may abstain (no click)
        client_kwargs: Dict[str, Any] = {}
        if api_key or os.environ.get("ANTHROPIC_API_KEY"):
            client_kwargs["api_key"] = api_key or os.environ.get("ANTHROPIC_API_KEY")
        if base_url or os.environ.get("ANTHROPIC_BASE_URL"):
            client_kwargs["base_url"] = base_url or os.environ.get("ANTHROPIC_BASE_URL")
        self.client = anthropic.Anthropic(**client_kwargs)

        self.max_workers = max_workers
        self.thinking_budget = thinking_budget
        self.max_retries = max_retries

        # Claude 4.7+ / Opus 4.6+ / Sonnet 4.6+ / Sonnet 5 / Fable 5 reject the
        # `temperature` parameter. The requested models here (Opus 4 / Sonnet 4)
        # accept it, but guard so pointing this at a newer model doesn't 400.
        n = model_name.lower()
        # "Newer" Claude generation (Opus 4.5+/Sonnet 4.6+/Sonnet 5/Fable): rejects
        # `temperature`, and uses the newer computer-use tool version.
        self._newer = (
            "opus-4-5" in n or "opus-4-6" in n or "opus-4-7" in n or "opus-4-8" in n
            or "sonnet-4-6" in n or "sonnet-5" in n or "fable" in n or "mythos" in n
        )
        self.supports_temperature = not (
            "opus-4-6" in n or "opus-4-7" in n or "opus-4-8" in n
            or "sonnet-4-6" in n or "sonnet-5" in n or "fable" in n or "mythos" in n
        )
        # computer-use tool version differs by model generation. Ordered
        # candidates: heuristic-best first, other as fallback (the backend
        # auto-switches on a "does not support tool types" error).
        _new = ("computer_20251124", "computer-use-2025-11-24")
        _old = ("computer_20250124", "computer-use-2025-01-24")
        self.cua_candidates = [_new, _old] if self._newer else [_old, _new]
        print(
            f"[AnthropicModel] model={model_name} computer_use={computer_use} "
            f"cua_long_edge={cua_long_edge} temperature={'on' if self.supports_temperature else 'off'} "
            f"thinking_budget={thinking_budget} workers={max_workers}"
        )

    # ------------------------------------------------------------------ utils
    @staticmethod
    def _fit_for_anthropic(image: Image.Image) -> Tuple[Image.Image, int, int]:
        """Downscale so Anthropic won't further resize; return (image, w, h)."""
        w, h = image.size
        scale = 1.0
        long_edge = max(w, h)
        if long_edge > _MAX_LONG_EDGE:
            scale = _MAX_LONG_EDGE / long_edge
        if (w * scale) * (h * scale) > _MAX_AREA:
            scale = min(scale, (_MAX_AREA / (w * h)) ** 0.5)
        if scale < 1.0:
            w, h = max(1, int(w * scale)), max(1, int(h * scale))
            image = image.resize((w, h))
        return image, w, h

    @staticmethod
    def _img_to_b64(image: Image.Image) -> str:
        if image.mode != "RGB":
            image = image.convert("RGB")
        buf = io.BytesIO()
        image.save(buf, format="PNG")
        return base64.b64encode(buf.getvalue()).decode("ascii")

    def _split_messages(self, messages: List[Dict[str, Any]], image: Image.Image):
        """Return (system_str, anthropic_messages) from `<image>`-templated messages."""
        b64 = self._img_to_b64(image)
        system_parts: List[str] = []
        out: List[Dict[str, Any]] = []
        for m in messages:
            role = m["role"]
            content = m["content"]
            if role == "system":
                system_parts.append(content)
                continue
            if isinstance(content, str) and "<image>" in content:
                parts = content.split("<image>")
                blocks: List[Dict[str, Any]] = []
                for i, txt in enumerate(parts):
                    if i > 0:
                        blocks.append({
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": "image/png",
                                "data": b64,
                            },
                        })
                    if txt:
                        blocks.append({"type": "text", "text": txt})
                out.append({"role": role, "content": blocks})
            else:
                out.append({"role": role, "content": content})
        system = "\n\n".join(system_parts) if system_parts else None
        return system, out

    @staticmethod
    def _to_normalized(text: str, w: int, h: int) -> str:
        """
        Parse Claude's coordinate answer and re-emit canonical normalized JSON.
        A fraction (<=1) is kept; a pixel value (>1), which is in the (w, h)
        space we sent, is divided by that dimension. Returns "" if unparseable.
        """
        mx = re.search(r'"x"\s*:\s*(-?\d*\.?\d+)', text)
        my = re.search(r'"y"\s*:\s*(-?\d*\.?\d+)', text)
        if mx and my:
            x, y = float(mx.group(1)), float(my.group(1))
        else:
            m = re.search(r'[\(\[]\s*(-?\d*\.?\d+)\s*,\s*(-?\d*\.?\d+)\s*[\)\]]', text)
            if m:
                x, y = float(m.group(1)), float(m.group(2))
            else:
                nums = re.findall(r'-?\d*\.?\d+', text)
                if len(nums) < 2:
                    return ""
                x, y = float(nums[0]), float(nums[1])
        # pixel answer -> fraction of the image we actually sent
        if x > 1.5 or y > 1.5:
            x, y = x / w, y / h
        x = min(max(x, 1e-4), 1.0 - 1e-4)
        y = min(max(y, 1e-4), 1.0 - 1e-4)
        return f'{{"x": {x:.4f}, "y": {y:.4f}}}'

    @staticmethod
    def _resize_long(image: Image.Image, long_edge: int) -> Tuple[Image.Image, int, int]:
        w, h = image.size
        f = min(1.0, long_edge / max(w, h))
        if f < 1.0:
            w, h = max(1, int(w * f)), max(1, int(h * f))
            image = image.resize((w, h))
        return image, w, h

    @staticmethod
    def _system_and_user_text(messages: List[Dict[str, Any]]) -> Tuple[Optional[str], str]:
        system, user = None, ""
        for m in messages:
            if m["role"] == "system":
                system = m["content"]
            elif m["role"] in ("user", "human"):
                user = str(m["content"]).replace("<image>", "").strip()
        return system, user

    # ---------------------------------------------------- computer-use one call
    def _one_cua(self, args, messages: List[Dict[str, Any]], image: Image.Image) -> str:
        img, w, h = self._resize_long(image, self.cua_long_edge)
        system, user_text = self._system_and_user_text(messages)
        content = [
            {"type": "image", "source": {"type": "base64", "media_type": "image/png",
                                         "data": self._img_to_b64(img)}},
            {"type": "text", "text": user_text or "Click on the described element."},
        ]
        base: Dict[str, Any] = {
            "model": self.model_name,
            "max_tokens": int(args.max_tokens),
            "messages": [{"role": "user", "content": content}],
        }
        # Force a tool call so the model can't reply with a refusal / question /
        # "let me take a screenshot" instead of clicking. In refusal mode
        # (OSWorld-G) we must NOT force it -- the model needs the option to
        # abstain when the element is absent.
        if not self.allow_refusal:
            base["tool_choice"] = {"type": "any"}
        if system:
            base["system"] = system
        if self.supports_temperature:
            base["temperature"] = float(args.temperature)

        # Retry on API errors (long backoff) AND on a no-click response (short
        # retry) -- Claude occasionally emits a non-click action (e.g. screenshot)
        # even under forced tool use, and a no-click otherwise scores as a miss.
        # `cand` selects the computer-tool version; auto-switch if the model
        # rejects it ("does not support tool types").
        last_err, last_txt, cand = None, "", 0
        for attempt in range(self.max_retries):
            tool_type, beta = self.cua_candidates[cand]
            kwargs = dict(base)
            kwargs["betas"] = [beta]
            kwargs["tools"] = [{"type": tool_type, "name": "computer",
                                "display_width_px": w, "display_height_px": h}]
            try:
                resp = self.client.beta.messages.create(**kwargs)
            except Exception as e:  # noqa: BLE001
                if "does not support tool types" in str(e) and cand + 1 < len(self.cua_candidates):
                    cand += 1  # wrong tool version for this model -- switch and retry
                    continue
                last_err = e
                time.sleep(min(2 ** attempt, 30))
                continue
            coord = None
            for b in resp.content:
                if getattr(b, "type", None) == "tool_use" and isinstance(getattr(b, "input", None), dict) \
                        and "coordinate" in b.input:
                    coord = b.input["coordinate"]
                    break
            if coord is not None and len(coord) == 2:
                nx = min(max(float(coord[0]) / w, 1e-4), 1.0 - 1e-4)
                ny = min(max(float(coord[1]) / h, 1e-4), 1.0 - 1e-4)
                return f'{{"x": {nx:.4f}, "y": {ny:.4f}}}'
            if self.allow_refusal:
                # No click == the model abstained (element absent). Emit a
                # negative point so the eval's refusal check scores it correctly.
                return '{"x": -1.0000, "y": -1.0000}'
            last_txt = "".join(b.text for b in resp.content if getattr(b, "type", None) == "text")
            time.sleep(0.5)  # no-click: brief pause, then retry
        if last_err is not None:
            print(f"[AnthropicModel] CUA request failed after {self.max_retries} retries: {last_err}")
        return self._to_normalized(last_txt, w, h)  # last resort: coords from text

    # ---------------------------------------------------------------- one call
    def _one(self, args, messages: List[Dict[str, Any]], image: Image.Image) -> str:
        if self.computer_use:
            return self._one_cua(args, messages, image)
        fitted, w, h = self._fit_for_anthropic(image)
        system, anth_msgs = self._split_messages(messages, fitted)
        kwargs: Dict[str, Any] = {
            "model": self.model_name,
            "max_tokens": int(args.max_tokens),
            "messages": anth_msgs,
        }
        if system:
            kwargs["system"] = system
        if self.thinking_budget and self.thinking_budget > 0:
            # Extended thinking (Claude 4.0 uses the enabled/budget_tokens form);
            # budget must be < max_tokens and temperature is not allowed with it.
            kwargs["thinking"] = {"type": "enabled", "budget_tokens": self.thinking_budget}
            kwargs["max_tokens"] = max(int(args.max_tokens), self.thinking_budget + 1024)
        elif self.supports_temperature:
            kwargs["temperature"] = float(args.temperature)

        last_err = None
        for attempt in range(self.max_retries):
            try:
                resp = self.client.messages.create(**kwargs)
                raw = "".join(
                    b.text for b in resp.content if getattr(b, "type", None) == "text"
                )
                # normalize using the exact dims we sent (w, h)
                return self._to_normalized(raw, w, h) or raw
            except Exception as e:  # noqa: BLE001 - surface then backoff/retry
                last_err = e
                time.sleep(min(2 ** attempt, 30))
        print(f"[AnthropicModel] request failed after {self.max_retries} retries: {last_err}")
        return ""

    # ------------------------------------------------------------ public batch
    def get_responses(self, args, messages_list, images) -> List[str]:
        assert len(messages_list) == len(images), "messages/images length mismatch"
        results: List[str] = [""] * len(messages_list)
        with ThreadPoolExecutor(max_workers=self.max_workers) as ex:
            futures = {
                ex.submit(self._one, args, messages_list[i], images[i]): i
                for i in range(len(messages_list))
            }
            for fut in tqdm(
                as_completed(futures),
                total=len(futures),
                desc="Generating responses (Anthropic)",
            ):
                idx = futures[fut]
                try:
                    results[idx] = fut.result()
                except Exception as e:  # noqa: BLE001
                    print(f"[AnthropicModel] sample {idx} errored: {e}")
                    results[idx] = ""
        return results
