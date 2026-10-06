import os
import io
import time
import base64
from typing import List, Dict, Any, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed

from tqdm import tqdm
from PIL import Image
from openai import OpenAI


class OpenAIModel:
    """
    OpenAI (GPT-4o / GPT-5 family) backend for the grounding eval.

    Mirrors the public surface of the vLLM `Qwen2VL` backend: exposes
    `get_responses(args, messages_list, images)` and returns a list of raw
    response strings aligned 1:1 with `messages_list`.

    Each entry of `messages_list` is a prompt-processor output: a `system`
    message plus a `user` message whose `content` contains exactly one
    `<image>` placeholder. We swap that placeholder for the (already resized)
    screenshot encoded as a base64 data URL and send it to the Chat Completions
    API. The prompt processor tells the model the image resolution, and the
    model returns a click coordinate in that same pixel space -- which equals
    the eval's resized-image / gt-bbox space -- so no rescaling is needed
    downstream.

    Concurrency: requests are issued from a thread pool (OpenAI's client is
    thread-safe). Order is preserved by writing results back by index.
    """

    def __init__(
        self,
        model_name: str,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        max_workers: int = 8,
        reasoning_effort: Optional[str] = None,
        image_detail: str = "high",
        max_retries: int = 5,
    ):
        self.model_name = model_name
        self.client = OpenAI(
            api_key=api_key or os.environ.get("OPENAI_API_KEY"),
            base_url=base_url or os.environ.get("OPENAI_BASE_URL") or None,
        )
        self.max_workers = max_workers
        self.reasoning_effort = reasoning_effort
        self.image_detail = image_detail
        self.max_retries = max_retries

        # Reasoning models (gpt-5*, o1/o3/o4*) reject `temperature != 1`, use
        # `max_completion_tokens` (not `max_tokens`), and count hidden reasoning
        # tokens against that budget. `gpt-5-chat*` is the non-reasoning chat
        # variant and behaves like a normal chat model.
        n = model_name.lower()
        self.is_reasoning = (
            (n.startswith("gpt-5") and "chat" not in n)
            or n.startswith(("o1", "o3", "o4"))
        )
        # "pro" models (gpt-5-pro, gpt-5.5-pro, o*-pro) are NOT chat models -- they
        # are only served by the Responses API, and require reasoning effort >=
        # medium (low/minimal are rejected).
        self.use_responses = "pro" in n
        print(
            f"[OpenAIModel] model={model_name} reasoning={self.is_reasoning} "
            f"api={'responses' if self.use_responses else 'chat'} "
            f"effort={reasoning_effort or '(default)'} detail={image_detail} "
            f"workers={max_workers}"
        )

    def _responses_input(self, messages, image):
        """Convert `<image>`-templated messages to (instructions, Responses input)."""
        data_url = self._img_to_data_url(image)
        instructions = None
        inp = []
        for m in messages:
            role, content = m["role"], m["content"]
            if role == "system":
                instructions = content if instructions is None else instructions + "\n\n" + content
                continue
            if isinstance(content, str) and "<image>" in content:
                parts = content.split("<image>")
                seg = []
                for i, txt in enumerate(parts):
                    if i > 0:
                        seg.append({"type": "input_image", "image_url": data_url, "detail": self.image_detail})
                    if txt:
                        seg.append({"type": "input_text", "text": txt})
                inp.append({"role": role, "content": seg})
            else:
                inp.append({"role": role, "content": [{"type": "input_text", "text": content}]})
        return instructions, inp

    def _one_responses(self, args, messages, image):
        instructions, inp = self._responses_input(messages, image)
        eff = (self.reasoning_effort or "medium").lower()
        if eff in ("low", "minimal"):  # pro models require >= medium
            eff = "medium"
        kwargs = {
            "model": self.model_name,
            "input": inp,
            "max_output_tokens": max(int(args.max_tokens), 8000),
            "reasoning": {"effort": eff},
        }
        if instructions:
            kwargs["instructions"] = instructions
        last_err = None
        for attempt in range(self.max_retries):
            try:
                resp = self.client.responses.create(**kwargs)
            except Exception as e:  # noqa: BLE001
                last_err = e
                time.sleep(min(2 ** attempt, 30))
                continue
            txt = (resp.output_text or "").strip()
            if txt:
                return txt
            # empty output (e.g. truncated by max_output_tokens, or a refusal):
            # brief pause then retry so it doesn't silently score as a miss.
            time.sleep(0.5)
        if last_err is not None:
            print(f"[OpenAIModel] responses request failed after {self.max_retries} retries: {last_err}")
        return ""

    # ------------------------------------------------------------------ utils
    @staticmethod
    def _img_to_data_url(image: Image.Image) -> str:
        if image.mode != "RGB":
            image = image.convert("RGB")
        buf = io.BytesIO()
        image.save(buf, format="PNG")
        b64 = base64.b64encode(buf.getvalue()).decode("ascii")
        return f"data:image/png;base64,{b64}"

    def _to_openai_messages(
        self, messages: List[Dict[str, Any]], image: Image.Image
    ) -> List[Dict[str, Any]]:
        """Convert internal `<image>`-templated messages to the OpenAI schema."""
        data_url = self._img_to_data_url(image)
        out: List[Dict[str, Any]] = []
        for m in messages:
            role = m["role"]
            content = m["content"]
            if isinstance(content, str) and "<image>" in content:
                parts = content.split("<image>")
                seg: List[Dict[str, Any]] = []
                for i, txt in enumerate(parts):
                    if i > 0:
                        seg.append(
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": data_url,
                                    "detail": self.image_detail,
                                },
                            }
                        )
                    if txt:
                        seg.append({"type": "text", "text": txt})
                out.append({"role": role, "content": seg})
            else:
                out.append({"role": role, "content": content})
        return out

    # ---------------------------------------------------------------- one call
    def _one(self, args, messages: List[Dict[str, Any]], image: Image.Image) -> str:
        if self.use_responses:
            return self._one_responses(args, messages, image)
        oai_msgs = self._to_openai_messages(messages, image)
        kwargs: Dict[str, Any] = {"model": self.model_name, "messages": oai_msgs}

        if self.is_reasoning:
            # Reasoning tokens are charged against this budget, so keep it
            # generous even though the visible answer (a coordinate) is tiny.
            kwargs["max_completion_tokens"] = max(int(args.max_tokens), 4096)
            if self.reasoning_effort:
                kwargs["reasoning_effort"] = self.reasoning_effort
        else:
            kwargs["max_tokens"] = int(args.max_tokens)
            kwargs["temperature"] = float(args.temperature)

        last_err = None
        for attempt in range(self.max_retries):
            try:
                resp = self.client.chat.completions.create(**kwargs)
                return resp.choices[0].message.content or ""
            except Exception as e:  # noqa: BLE001 - surface then backoff/retry
                last_err = e
                time.sleep(min(2 ** attempt, 30))
        print(f"[OpenAIModel] request failed after {self.max_retries} retries: {last_err}")
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
                desc="Generating responses (OpenAI)",
            ):
                idx = futures[fut]
                try:
                    results[idx] = fut.result()
                except Exception as e:  # noqa: BLE001
                    print(f"[OpenAIModel] sample {idx} errored: {e}")
                    results[idx] = ""
        return results
