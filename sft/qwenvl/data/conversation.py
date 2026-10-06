"""
Conversation prompt templates.

We kindly request that you import fastchat instead of copying this file if you wish to use it.
If you have changes in mind, please contribute back so the community can benefit collectively and continue to maintain these valuable templates.
"""

import dataclasses
import json
from enum import IntEnum, auto
from typing import Any, Dict, List, Tuple, Union


class SeparatorStyle(IntEnum):
    """Separator styles."""

    ADD_COLON_SINGLE = auto()
    ADD_COLON_TWO = auto()
    ADD_COLON_SPACE_SINGLE = auto()
    NO_COLON_SINGLE = auto()
    NO_COLON_TWO = auto()
    ADD_NEW_LINE_SINGLE = auto()
    LLAMA2 = auto()
    CHATGLM = auto()
    CHATML = auto()
    CHATINTERN = auto()
    DOLLY = auto()
    RWKV = auto()
    PHOENIX = auto()
    ROBIN = auto()
    FALCON_CHAT = auto()
    CHATGLM3 = auto()
    INTERNVL_ZH = auto()
    MPT = auto()


@dataclasses.dataclass
class Conversation:
    """A class that manages prompt templates and keeps all conversation history."""

    # The name of this template
    name: str
    # The template of the system prompt
    system_template: str = "{system_message}"
    # The system message
    system_message: str = ""
    # The names of two roles
    roles: Tuple[str] = ("USER", "ASSISTANT")
    # All messages. Each item is (role, message).
    messages: List[List[str]] = ()
    # The number of few shot examples
    offset: int = 0
    # The separator style and configurations
    sep_style: SeparatorStyle = SeparatorStyle.ADD_COLON_SINGLE
    sep: str = "\n"
    sep2: str = None
    # Stop criteria (the default one is EOS token)
    stop_str: Union[str, List[str]] = None
    # Stops generation if meeting any token in this list
    stop_token_ids: List[int] = None

    def get_prompt(self) -> str:
        """Get the prompt for generation."""
        system_prompt = self.system_template.format(system_message=self.system_message)
        if self.sep_style == SeparatorStyle.ADD_COLON_SINGLE:
            ret = system_prompt + self.sep
            for role, message in self.messages:
                if message:
                    ret += role + ": " + message + self.sep
                else:
                    ret += role + ":"
            return ret
        elif self.sep_style == SeparatorStyle.ADD_COLON_TWO:
            seps = [self.sep, self.sep2]
            ret = system_prompt + seps[0]
            for i, (role, message) in enumerate(self.messages):
                if message:
                    ret += role + ": " + message + seps[i % 2]
                else:
                    ret += role + ":"
            return ret
        elif self.sep_style == SeparatorStyle.ADD_COLON_SPACE_SINGLE:
            ret = system_prompt + self.sep
            for role, message in self.messages:
                if message:
                    ret += role + ": " + message + self.sep
                else:
                    ret += role + ": "  # must be end with a space
            return ret
        elif self.sep_style == SeparatorStyle.ADD_NEW_LINE_SINGLE:
            ret = "" if system_prompt == "" else system_prompt + self.sep
            for role, message in self.messages:
                if message:
                    ret += role + "\n" + message + self.sep
                else:
                    ret += role + "\n"
            return ret
        elif self.sep_style == SeparatorStyle.NO_COLON_SINGLE:
            ret = system_prompt
            for role, message in self.messages:
                if message:
                    ret += role + message + self.sep
                else:
                    ret += role
            return ret
        elif self.sep_style == SeparatorStyle.NO_COLON_TWO:
            seps = [self.sep, self.sep2]
            ret = system_prompt
            for i, (role, message) in enumerate(self.messages):
                if message:
                    ret += role + message + seps[i % 2]
                else:
                    ret += role
            return ret
        elif self.sep_style == SeparatorStyle.RWKV:
            ret = system_prompt
            for i, (role, message) in enumerate(self.messages):
                if message:
                    ret += (
                        role
                        + ": "
                        + message.replace("\r\n", "\n").replace("\n\n", "\n")
                    )
                    ret += "\n\n"
                else:
                    ret += role + ":"
            return ret
        elif self.sep_style == SeparatorStyle.LLAMA2:
            seps = [self.sep, self.sep2]
            if self.system_message:
                ret = system_prompt
            else:
                ret = "[INST] "
            for i, (role, message) in enumerate(self.messages):
                tag = self.roles[i % 2]
                if message:
                    if i == 0:
                        ret += message + " "
                    else:
                        ret += tag + " " + message + seps[i % 2]
                else:
                    ret += tag
            return ret
        elif self.sep_style == SeparatorStyle.CHATGLM:
            # source: https://huggingface.co/THUDM/chatglm-6b/blob/1d240ba371910e9282298d4592532d7f0f3e9f3e/modeling_chatglm.py#L1302-L1308
            # source2: https://huggingface.co/THUDM/chatglm2-6b/blob/e186c891cf64310ac66ef10a87e6635fa6c2a579/modeling_chatglm.py#L926
            round_add_n = 1 if self.name == "chatglm2" else 0
            if system_prompt:
                ret = system_prompt + self.sep
            else:
                ret = ""

            for i, (role, message) in enumerate(self.messages):
                if i % 2 == 0:
                    ret += f"[Round {i//2 + round_add_n}]{self.sep}"

                if message:
                    ret += f"{role}：{message}{self.sep}"
                else:
                    ret += f"{role}："
            return ret
        elif self.sep_style == SeparatorStyle.CHATML:
            ret = "" if system_prompt == "" else system_prompt + self.sep + "\n"
            for role, message in self.messages:
                if message:
                    ret += role + "\n" + message + self.sep + "\n"
                else:
                    ret += role + "\n"
            return ret
        elif self.sep_style == SeparatorStyle.CHATGLM3:
            ret = ""
            if self.system_message:
                ret += system_prompt
            for role, message in self.messages:
                if message:
                    ret += role + "\n" + " " + message
                else:
                    ret += role
            return ret
        elif self.sep_style == SeparatorStyle.CHATINTERN:
            # source: https://huggingface.co/internlm/internlm-chat-7b-8k/blob/bd546fa984b4b0b86958f56bf37f94aa75ab8831/modeling_internlm.py#L771
            seps = [self.sep, self.sep2]
            ret = system_prompt
            for i, (role, message) in enumerate(self.messages):
                # if i % 2 == 0:
                #     ret += "<s>"
                if message:
                    ret += role + ":" + message + seps[i % 2] + "\n"
                else:
                    ret += role + ":"
            return ret
        elif self.sep_style == SeparatorStyle.DOLLY:
            seps = [self.sep, self.sep2]
            ret = system_prompt
            for i, (role, message) in enumerate(self.messages):
                if message:
                    ret += role + ":\n" + message + seps[i % 2]
                    if i % 2 == 1:
                        ret += "\n\n"
                else:
                    ret += role + ":\n"
            return ret
        elif self.sep_style == SeparatorStyle.PHOENIX:
            ret = system_prompt
            for role, message in self.messages:
                if message:
                    ret += role + ": " + "<s>" + message + "</s>"
                else:
                    ret += role + ": " + "<s>"
            return ret
        elif self.sep_style == SeparatorStyle.ROBIN:
            ret = system_prompt + self.sep
            for role, message in self.messages:
                if message:
                    ret += role + ":\n" + message + self.sep
                else:
                    ret += role + ":\n"
            return ret
        elif self.sep_style == SeparatorStyle.FALCON_CHAT:
            ret = ""
            if self.system_message:
                ret += system_prompt + self.sep
            for role, message in self.messages:
                if message:
                    ret += role + ": " + message + self.sep
                else:
                    ret += role + ":"

            return ret
        elif self.sep_style == SeparatorStyle.INTERNVL_ZH:
            seps = [self.sep2, self.sep]
            ret = self.system_message + seps[0]
            for i, (role, message) in enumerate(self.messages):
                if message:
                    ret += role + ": " + message + seps[i % 2]
                else:
                    ret += role + ":"
            return ret
        elif self.sep_style == SeparatorStyle.MPT:
            ret = system_prompt + self.sep
            for role, message in self.messages:
                if message:
                    if type(message) is tuple:
                        message, _, _ = message
                    ret += role + message + self.sep
                else:
                    ret += role
            return ret
        else:
            raise ValueError(f"Invalid style: {self.sep_style}")

    def set_system_message(self, system_message: str):
        """Set the system message."""
        self.system_message = system_message

    def append_message(self, role: str, message: str):
        """Append a new message."""
        self.messages.append([role, message])

    def update_last_message(self, message: str):
        """Update the last output.

        The last message is typically set to be None when constructing the prompt,
        so we need to update it in-place after getting the response from a model.
        """
        self.messages[-1][1] = message

    def to_gradio_chatbot(self):
        """Convert the conversation to gradio chatbot format."""
        ret = []
        for i, (role, msg) in enumerate(self.messages[self.offset :]):
            if i % 2 == 0:
                ret.append([msg, None])
            else:
                ret[-1][-1] = msg
        return ret

    def to_openai_api_messages(self):
        """Convert the conversation to OpenAI chat completion format."""
        ret = [{"role": "system", "content": self.system_message}]

        for i, (_, msg) in enumerate(self.messages[self.offset :]):
            if i % 2 == 0:
                ret.append({"role": "user", "content": msg})
            else:
                if msg is not None:
                    ret.append({"role": "assistant", "content": msg})
        return ret

    def copy(self):
        return Conversation(
            name=self.name,
            system_template=self.system_template,
            system_message=self.system_message,
            roles=self.roles,
            messages=[[x, y] for x, y in self.messages],
            offset=self.offset,
            sep_style=self.sep_style,
            sep=self.sep,
            sep2=self.sep2,
            stop_str=self.stop_str,
            stop_token_ids=self.stop_token_ids,
        )

    def dict(self):
        return {
            "template_name": self.name,
            "system_message": self.system_message,
            "roles": self.roles,
            "messages": self.messages,
            "offset": self.offset,
        }


# A global registry for all conversation templates
conv_templates: Dict[str, Conversation] = {}


def register_conv_template(template: Conversation, override: bool = False):
    """Register a new conversation template."""
    if not override:
        assert (
            template.name not in conv_templates
        ), f"{template.name} has been registered."

    conv_templates[template.name] = template


def get_conv_template(name: str) -> Conversation:
    """Get a conversation template."""
    return conv_templates[name].copy()


# InternVL-Chat-V1-1 template
register_conv_template(
    Conversation(
        name="internvl_zh",
        system_template="",
        roles=("<human>", "<bot>"),
        sep_style=SeparatorStyle.INTERNVL_ZH,
        sep="</s>",
        sep2=" ",
    )
)


# Both Hermes-2 and internlm2-chat are chatml-format conversation templates. The difference
# is that during training, the preprocessing function for the Hermes-2 template doesn't add
# <s> at the beginning of the tokenized sequence, while the internlm2-chat template does.
# Therefore, they are completely equivalent during inference.
register_conv_template(
    Conversation(
        name="Hermes-2",
        system_template="<|im_start|>system\n{system_message}",
        # note: The new system prompt was not used here to avoid changes in benchmark performance.
        # system_message: the newer (Chinese) InternVL identity prompt, roughly "I am InternVL
        # (Shusheng Wanxiang), a multimodal LLM developed by Shanghai AI Laboratory, Tsinghua
        # University and partners."
        system_message="你是由上海人工智能实验室联合商汤科技开发的书生多模态大模型，英文名叫InternVL, 是一个有用无害的人工智能助手。",
        roles=("<|im_start|>user\n", "<|im_start|>assistant\n"),
        sep_style=SeparatorStyle.MPT,
        sep="<|im_end|>",
        stop_str="<|endoftext|>",
    )
)


register_conv_template(
    Conversation(
        name="internlm2-chat",
        system_template="<|im_start|>system\n{system_message}",
        # note: The new system prompt was not used here to avoid changes in benchmark performance.
        # system_message: the newer (Chinese) InternVL identity prompt, roughly "I am InternVL
        # (Shusheng Wanxiang), a multimodal LLM developed by Shanghai AI Laboratory, Tsinghua
        # University and partners."
        system_message="你是由上海人工智能实验室联合商汤科技开发的书生多模态大模型，英文名叫InternVL, 是一个有用无害的人工智能助手。",
        roles=("<|im_start|>user\n", "<|im_start|>assistant\n"),
        sep_style=SeparatorStyle.MPT,
        sep="<|im_end|>",
    )
)


register_conv_template(
    Conversation(
        name="phi3-chat",
        system_template="<|system|>\n{system_message}",
        # note: The new system prompt was not used here to avoid changes in benchmark performance.
        # system_message: the newer (Chinese) InternVL identity prompt, roughly "I am InternVL
        # (Shusheng Wanxiang), a multimodal LLM developed by Shanghai AI Laboratory, Tsinghua
        # University and partners."
        system_message="你是由上海人工智能实验室联合商汤科技开发的书生多模态大模型，英文名叫InternVL, 是一个有用无害的人工智能助手。",
        roles=("<|user|>\n", "<|assistant|>\n"),
        sep_style=SeparatorStyle.MPT,
        sep="<|end|>",
    )
)


register_conv_template(
    Conversation(
        name="internvl2_5",
        system_template="<|im_start|>system\n{system_message}",
        system_message="你是书生·万象，英文名是InternVL，是由上海人工智能实验室、清华大学及多家合作单位联合开发的多模态大语言模型。",
        roles=("<|im_start|>user\n", "<|im_start|>assistant\n"),
        sep_style=SeparatorStyle.MPT,
        sep="<|im_end|>\n",
    )
)


register_conv_template(
    Conversation(
        name="internvl_grounding",
        system_template="<|im_start|>system\n{system_message}",
        system_message="你是书生·万象，英文名是InternVL，是由上海人工智能实验室、清华大学及多家合作单位联合开发的多模态大语言模型。",
        roles=("<|im_start|>user\n", "<|im_start|>assistant\n"),
        sep_style=SeparatorStyle.MPT,
        sep="<|im_end|>\n",
    )
)


register_conv_template(
    Conversation(
        name="internvl_referring",
        system_template="<|im_start|>system\n{system_message}",
        system_message="你是书生·万象，英文名是InternVL，是由上海人工智能实验室、清华大学及多家合作单位联合开发的多模态大语言模型。",
        roles=("<|im_start|>user\n", "<|im_start|>assistant\n"),
        sep_style=SeparatorStyle.MPT,
        sep="<|im_end|>\n",
    )
)


register_conv_template(
    Conversation(
        name="internvl2_5_aguvis",
        system_template="<|im_start|>system\n{system_message}",
        system_message="你是书生·万象，英文名是InternVL，是由上海人工智能实验室、清华大学及多家合作单位联合开发的多模态大语言模型。",
        roles=("<|im_start|>user\n", "<|im_start|>assistant\n"),
        sep_style=SeparatorStyle.MPT,
        sep="<|im_end|>\n",
    )
)


register_conv_template(
    Conversation(
        name="internvl2_5_aguvis_v2",
        system_template="<|im_start|>system\n{system_message}",
        system_message="你是书生·万象，英文名是InternVL，是由上海人工智能实验室、清华大学及多家合作单位联合开发的多模态大语言模型。",
        roles=("<|im_start|>user\n", "<|im_start|>assistant\n"),
        sep_style=SeparatorStyle.MPT,
        sep="<|im_end|>\n",
    )
)


register_conv_template(
    Conversation(
        name="internvl2_5_aguvis_v3",
        system_template="<|im_start|>system\n{system_message}",
        system_message="You are a GUI agent. You are given a task and a screenshot of the screen. You need to perform a series of pyautogui actions to complete the task.",
        roles=("<|im_start|>user\n", "<|im_start|>assistant\n"),
        sep_style=SeparatorStyle.MPT,
        sep="<|im_end|>\n",
    )
)


register_conv_template(
    Conversation(
        name="internvl2_5_aguvis_v4",
        system_template="<|im_start|>system\n{system_message}",
        system_message="You are a GUI agent. You are given a task and a screenshot of the screen. You need to perform a series of pyautogui actions to complete the task.",
        roles=("<|im_start|>user\n", "<|im_start|>assistant\n"),
        sep_style=SeparatorStyle.MPT,
        sep="<|im_end|>\n",
    )
)


register_conv_template(
    Conversation(
        name="internvl2_5_private_v1",
        system_template="<|im_start|>system\n{system_message}",
        system_message="你是书生·万象，英文名是InternVL，是由上海人工智能实验室、清华大学及多家合作单位联合开发的多模态大语言模型。",
        roles=("<|im_start|>user\n", "<|im_start|>assistant\n"),
        sep_style=SeparatorStyle.MPT,
        sep="<|im_end|>\n",
    )
)


# GUI

# ---------------------------------------------------------------------------
# Action space (aligned with Mobile-Agent-v3.5 / GUI-Owl, Qwen tool-calling)
# ---------------------------------------------------------------------------
# Every platform exposes ONE function tool. The model emits a tool call as a
#   <tool_call>{"name": ..., "arguments": {...}}</tool_call>
# block, exactly like Mobile-Agent-v3.5's mobile_use / computer_use tools:
#   * mobile  -> `mobile_use`    (touchscreen)
#   * desktop -> `computer_use`  (mouse + keyboard)
#   * web     -> `browser_use`   (computer_use + browser navigation)
# Coordinates are integers in a 1000x1000 space, matching the dataset's
# per-mille (0-1000) convention. Each action carries the parameters it needs
# and a `grd` flag marking whether it belongs to the grounding subset (the
# pointing actions consumed by the *_grounding_* templates).


def _required_by(names):
    """Render the "Required only by `action=a`, `action=b`, and `action=c`." clause."""
    toks = ["`action=%s`" % n for n in names]
    if len(toks) == 1:
        joined = toks[0]
    elif len(toks) == 2:
        joined = "%s and %s" % (toks[0], toks[1])
    else:
        joined = ", ".join(toks[:-1]) + ", and " + toks[-1]
    return "Required only by " + joined + "."


def _optional_for(names, default=None):
    """Render the sentence for an optional parameter: the action can use it, but it may be omitted."""
    toks = ["`action=%s`" % n for n in names]
    joined = toks[0] if len(toks) == 1 else (
        "%s and %s" % (toks[0], toks[1]) if len(toks) == 2
        else ", ".join(toks[:-1]) + ", and " + toks[-1])
    tail = " Omit it to use the default%s." % ("" if default is None else " of %s" % default)
    return "Optional for " + joined + "; it is never required." + tail


def _build_tool(tool, actions):
    """Assemble one Qwen-style function tool JSON string from an ordered action list."""
    bullets = ["The action to perform. The available actions are:"]
    enum = []
    param_order = []
    for act in actions:
        enum.append(act["name"])
        bullets.append("* `%s`: %s" % (act["name"], act["desc"]))
        for p in act["params"]:
            if p not in param_order:
                param_order.append(p)
    properties = {
        "action": {
            "description": "\n".join(bullets),
            "enum": enum,
            "type": "string",
        }
    }
    for p in param_order:
        spec = tool["param_specs"][p]
        _users = [a["name"] for a in actions if p in a["params"]]
        req = (_optional_for(_users, spec.get("default"))
               if spec.get("optional") else _required_by(_users))
        base = spec.get("base", "")
        prop = {
            "description": (base + " " + req).strip() if base else req,
            "type": spec["type"],
        }
        if "enum" in spec:
            prop["enum"] = spec["enum"]
        properties[p] = prop
    fn = {
        "type": "function",
        "function": {
            "name": tool["name"],
            "description": tool["description"],
            "parameters": {
                "properties": properties,
                "required": ["action"],
                "type": "object",
            },
        },
    }
    return json.dumps(fn, ensure_ascii=False)


# --- mobile: `mobile_use` -----------------------------------------------------
MOBILE_USE = {
    "name": "mobile_use",
    "description": (
        "Use a touchscreen to interact with a mobile device, and take screenshots.\n"
        "* This is an interface to a mobile device with touchscreen. You can perform "
        "actions like clicking, typing, swiping, etc.\n"
        "* Some applications may take time to start or process actions, so you may need "
        "to wait and take successive screenshots to see the results of your actions.\n"
        "* The screen's resolution is 1000x1000.\n"
        "* Make sure to click any buttons, links, icons, etc with the cursor tip in the "
        "center of the element. Don't click boxes on their edges unless asked."
    ),
    "param_specs": {
        "coordinate": {
            "type": "array",
            "base": (
                "(x, y): the point on the screen the action acts on — the element to touch, the "
                "region to scroll, or the object to start dragging. x is measured from the left "
                "edge and y from the top edge, both on a 0-1000 scale, so y gets larger toward the "
                "bottom of the screen."
            ),
        },
        "coordinate2": {
            "type": "array",
            "base": (
                "(x2, y2): where the dragged object must end up, on the same axes as coordinate. "
                "It is a destination, not a direction."
            ),
        },
        # The two parameters of `scroll`.
        "direction": {
            "type": "string",
            "enum": ["up", "down", "left", "right"],
            # Do not write "Required only by ..." in `base`: _build_tool appends it
            # automatically from the action subset.
            "base": "The direction THE VIEW travels over the content, not a finger direction.",
        },
        # Optional, for two reasons: (1) AndroidControl scores a scroll by direction only
        # and ignores its magnitude, which matters only when executing on a real device,
        # so it should not be a mandatory field on every step; (2) trajectory data only
        # has screen-normalised coordinates, not the size of the scrollable region, so the
        # amount is expressed as a fraction of the screen (matching the data converter).
        "amount": {
            "type": "number",
            "optional": True,
            "default": 0.5,
            "base": (
                "How far to scroll along that axis, as a fraction of the screen: 0.5 moves the "
                "view by about half a screen."
            ),
        },
        "text": {"type": "string", "base": ""},
        "time": {"type": "number", "base": "The seconds to wait."},
        "button": {
            "type": "string",
            "enum": ["Back", "Home", "Menu", "Enter"],
            "base": (
                "Back means returning to the previous interface, Home means returning to the "
                "desktop, Menu means opening the application background menu, and Enter means "
                "pressing the enter."
            ),
        },
        "status": {
            "type": "string",
            "enum": ["success", "failure"],
            "base": "The status of the task.",
        },
    },
    "actions": [
        {
            "name": "key",
            "desc": (
                "Perform a key event on the mobile device."
                "\n    - This supports adb's `keyevent` syntax."
                "\n    - Examples: \"volume_up\", \"volume_down\", \"power\", \"camera\", \"clear\"."
            ),
            "params": ["text"],
            "grd": False,
        },
        {
            "name": "click",
            "desc": "Click the point on the screen with coordinate (x, y).",
            "params": ["coordinate"],
            "grd": True,
        },
        {
            "name": "long_press",
            "desc": "Press the point on the screen with coordinate (x, y) for specified seconds.",
            "params": ["coordinate", "time"],
            "grd": True,
        },
        # Why scrolling is its own action rather than a use of `swipe`: with one swipe
        # covering both (a) scrolling/panning and (b) dragging an object, the direction can
        # only be encoded as the displacement between two points, and "swipe down" (finger
        # moves down) and "scroll down" (view moves down) point in opposite directions in
        # English, so the model has to untangle two layers at once. Measured on AndroidControl
        # with a single-swipe schema: across four runs a checkpoint emitted 1,274 scroll
        # gestures of which only 9 were downward finger swipes, and got 1 of the 46 steps
        # whose ground truth is an upward scroll right, even when the low-level instruction
        # literally said "Swipe down to view the reviews". Re-running with a prompt that
        # reproduced the training format verbatim left the metric unchanged
        # (0.0217 -> 0.0217), ruling out the eval scaffolding. Hence `direction` is an enum
        # and the finger is removed from this action entirely.
        #
        # `direction` has the same meaning as the sign of desktop computer_use.scroll's
        # `clicks` (positive = up = back toward the top) and maps one-to-one onto
        # AndroidControl's ground-truth direction, so the evaluator needs no
        # swipe-to-direction conversion.
        #
        # grd=False: the coordinate says WHERE to scroll, not a target to localise.
        {
            "name": "scroll",
            "desc": (
                "Scroll the view at coordinate (x, y) in the given `direction`. `direction` names "
                "the direction THE VIEW travels over the content, never a finger or wheel motion: "
                "`down` advances downward through the content, revealing what was below the fold; "
                "`up` goes back toward the top, revealing what was above; `right` and `left` work "
                "the same way on the horizontal axis. This action does not describe a gesture — do "
                "not think in terms of which way a finger moves. Put (x, y) anywhere inside the "
                "scrollable region that should move, which matters when a horizontally scrolling "
                "row sits inside a vertically scrolling page. `amount` is optional — it is how "
                "far to scroll as a fraction of the screen, and defaults to 0.5 when omitted."
            ),
            "params": ["coordinate", "direction", "amount"],
            "grd": False,
        },
        {
            "name": "swipe",
            # Drag only. Scrolling/panning always goes through `scroll` (see the note above it).
            #
            # grd=False: swipe is used for task/trajectory training but not for grounding
            # training, because its end point has no localisable object. For scroll-like
            # swipes the two ends only carry direction and distance -- extracted end-point
            # descriptions look like "upper part of the list" / "lower-right area of the
            # canvas" / "rightward on the photo edge", which match no element on screen, so a
            # grounding model could only invent a coordinate. Supervising grounding on such
            # samples teaches the marginal distribution of gesture end points instead of
            # "look at the screen -> pick the coordinate", and dilutes the real localisation
            # signal. grd=False does not mean "no coordinates": coordinate and coordinate2
            # are still produced and the tool_call is still valid; the sample is just not
            # used as localisation supervision.
            "desc": (
                "Drag whatever sits under coordinate (x, y) to coordinate2 (x2, y2). Use this ONLY "
                "to move an object or a control: put (x, y) on the object or its handle, and put "
                "(x2, y2) exactly where it must end up — a destination slot, another element, or "
                "the position on a track that encodes the wanted value. (x2, y2) is decided by the "
                "destination alone and may lie in any direction from (x, y). To scroll or pan a "
                "view, use `scroll` instead; this action never means scrolling."
            ),
            "params": ["coordinate", "coordinate2"],
            "grd": False,
        },
        {
            "name": "type",
            "desc": "Input the specified text into the activated input box.",
            "params": ["text"],
            "grd": False,
        },
        {
            "name": "system_button",
            "desc": "Press the system button.",
            "params": ["button"],
            "grd": False,
        },
        {
            "name": "open",
            "desc": "Open an app on the device.",
            "params": ["text"],
            "grd": False,
        },
        {
            "name": "wait",
            "desc": "Wait specified seconds for the change to happen.",
            "params": ["time"],
            "grd": False,
        },
        {
            "name": "answer",
            "desc": "Terminate the current task and output the answer.",
            "params": ["text"],
            "grd": False,
        },
        {
            "name": "interact",
            "desc": "Resolve the blocking window by interacting with the user.",
            "params": ["text"],
            "grd": False,
        },
        {
            "name": "terminate",
            "desc": "Terminate the current task and report its completion status.",
            "params": ["status"],
            "grd": False,
        },
    ],
}


# --- desktop: `computer_use` --------------------------------------------------
COMPUTER_USE = {
    "name": "computer_use",
    "description": (
        "Use a mouse and keyboard to interact with a computer, and take screenshots.\n"
        "* This is an interface to a desktop GUI. You do not have access to a terminal or "
        "applications menu. You must click on desktop icons to start applications.\n"
        "* Some applications may take time to start or process actions, so you may need to "
        "wait and take successive screenshots to see the results of your actions. E.g. if "
        "you click on Firefox and a window doesn't open, try wait and taking another "
        "screenshot.\n"
        "* The screen's resolution is 1000x1000.\n"
        "* Make sure to click any buttons, links, icons, etc with the cursor tip in the "
        "center of the element. Don't click boxes on their edges unless asked."
    ),
    "param_specs": {
        "keys": {"type": "array", "base": ""},
        "text": {"type": "string", "base": ""},
        "coordinate": {
            "type": "array",
            "base": (
                "(x, y): The x (pixels from the left edge) and y (pixels from the top edge) "
                "coordinates to move the mouse to."
            ),
        },
        "coordinate2": {
            "type": "array",
            "base": (
                "(x, y): The x (pixels from the left edge) and y (pixels from the top edge) "
                "coordinates to drag the cursor to."
            ),
        },
        "clicks": {
            "type": "number",
            "base": (
                "The number of scroll clicks to perform. For `action=scroll`, positive values scroll "
                "up (back toward the top) and negative values scroll down (revealing content further "
                "down). For `action=hscroll`, positive values scroll right (revealing content "
                "further right) and negative values scroll left. The sign always names the direction "
                # Phrased from the VIEW's point of view ("the direction the content scrolls"
                # would contradict "(back toward the top)": when the view moves down, the
                # content pixels move up). Same wording as mobile `scroll.direction`, so all
                # three platforms describe scrolling the same way.
                "THE VIEW travels over the content, never the direction the wheel or finger "
                "travels."
            ),
        },
        "time": {"type": "number", "base": "The seconds to wait."},
        "status": {
            "type": "string",
            "enum": ["success", "failure"],
            "base": "The status of the task.",
        },
    },
    "actions": [
        {
            "name": "key",
            "desc": (
                "Performs key down presses on the arguments passed in order, then performs key "
                "releases in reverse order."
            ),
            "params": ["keys"],
            "grd": False,
        },
        {
            "name": "type",
            "desc": "Type a string of text on the keyboard.",
            "params": ["text"],
            "grd": False,
        },
        {
            "name": "mouse_move",
            "desc": "Move the cursor to a specified (x, y) pixel coordinate on the screen.",
            "params": ["coordinate"],
            "grd": True,
        },
        {
            "name": "left_click",
            "desc": "Click the left mouse button at a specified (x, y) pixel coordinate on the screen.",
            "params": ["coordinate"],
            "grd": True,
        },
        {
            "name": "left_click_drag",
            "desc": (
                "Press the left mouse button at the start point with coordinate (x, y), drag to the end "
                "point with coordinate2 (x2, y2), and release."
            ),
            "params": ["coordinate", "coordinate2"],
            "grd": True,
        },
        {
            "name": "right_click",
            "desc": "Click the right mouse button at a specified (x, y) pixel coordinate on the screen.",
            "params": ["coordinate"],
            "grd": True,
        },
        {
            "name": "middle_click",
            "desc": (
                "Click the middle mouse button at a specified (x, y) pixel coordinate on the screen."
            ),
            "params": ["coordinate"],
            "grd": True,
        },
        {
            "name": "double_click",
            "desc": (
                "Double-click the left mouse button at a specified (x, y) pixel coordinate on the "
                "screen."
            ),
            "params": ["coordinate"],
            "grd": True,
        },
        {
            "name": "triple_click",
            "desc": (
                "Triple-click the left mouse button at a specified (x, y) pixel coordinate on the "
                "screen."
            ),
            "params": ["coordinate"],
            "grd": True,
        },
        {
            "name": "scroll",
            "desc": "Performs a scroll of the mouse scroll wheel.",
            "params": ["coordinate", "clicks"],
            "grd": False,
        },
        {
            "name": "hscroll",
            "desc": "Performs a horizontal scroll.",
            "params": ["coordinate", "clicks"],
            "grd": False,
        },
        {
            "name": "wait",
            "desc": "Wait specified seconds for the change to happen.",
            "params": ["time"],
            "grd": False,
        },
        {
            "name": "terminate",
            "desc": "Terminate the current task and report its completion status.",
            "params": ["status"],
            "grd": False,
        },
        {
            "name": "answer",
            "desc": "Answer a question.",
            "params": ["text"],
            "grd": False,
        },
        {
            "name": "interact",
            "desc": "Resolve the blocking window by interacting with the user.",
            "params": ["text"],
            "grd": False,
        },
    ],
}


# --- web: `browser_use` = computer_use actions + browser navigation ----------
# browser_use has no `swipe`: on the web it expresses the same intent as `scroll`/`hscroll`,
# having both makes the action extractor pick one at random (94% of web swipes were in
# fact scrolls), and swipe carries no coordinate and has grd=False, so it is strictly less
# informative than scroll. Scrolling always goes through scroll/hscroll; `swipe` exists only
# on mobile, where it is a real gesture.
_WEB_EXTRA_ACTIONS = [
    {
        "name": "select",
        "desc": (
            "Select an option in a dropdown or list element at the specified (x, y) coordinate; the "
            "option to choose is given by text."
        ),
        "params": ["coordinate", "text"],
        "grd": True,
    },
    {
        "name": "open_url",
        "desc": "Navigate the browser to the URL specified by text.",
        "params": ["text"],
        "grd": False,
    },
    {
        "name": "go_back",
        "desc": "Go back to the previous page in the browser history.",
        "params": [],
        "grd": False,
    },
    {
        "name": "go_forward",
        "desc": "Go forward to the next page in the browser history.",
        "params": [],
        "grd": False,
    },
]

# browser_use uses exactly computer_use's parameter set (no direction/amount: there is no swipe).
_BROWSER_PARAM_SPECS = dict(COMPUTER_USE["param_specs"])

_web_actions = []
for _act in COMPUTER_USE["actions"]:
    if _act["name"] == "wait":
        _web_actions.extend(_WEB_EXTRA_ACTIONS)
    _web_actions.append(_act)

BROWSER_USE = {
    "name": "browser_use",
    "description": (
        "Use a mouse and keyboard to interact with a web browser, and take screenshots.\n"
        "* This is an interface to a web browser. You interact with the rendered web page by "
        "clicking, typing, scrolling, selecting options and navigating between pages.\n"
        "* Some pages may take time to load or process actions, so you may need to wait and "
        "take successive screenshots to see the results of your actions.\n"
        "* The screen's resolution is 1000x1000.\n"
        "* Make sure to click any buttons, links, icons, etc with the cursor tip in the "
        "center of the element. Don't click boxes on their edges unless asked."
    ),
    "param_specs": _BROWSER_PARAM_SPECS,
    "actions": _web_actions,
}


# `ACTIONS` / `GRD_ACTIONS`: the ordered action list per platform, and its
# grounding (pointing-only) subset. Kept as named definitions for readability
# and for any downstream consumer; the templates below are built from them.
ACTIONS = {
    "mobile": MOBILE_USE["actions"],
    "desktop": COMPUTER_USE["actions"],
    "web": BROWSER_USE["actions"],
}

GRD_ACTIONS = {
    platform: [a for a in acts if a["grd"]]
    for platform, acts in ACTIONS.items()
}

# platform -> tool table. "mind2web" is an alias of "web".
_TOOL_BY_PLATFORM = {
    "mobile": MOBILE_USE,
    "desktop": COMPUTER_USE,
    "web": BROWSER_USE,
    "mind2web": BROWSER_USE,
}



def _tools_for(platform):
    if platform == "all":
        return [MOBILE_USE, COMPUTER_USE, BROWSER_USE]
    return [_TOOL_BY_PLATFORM[platform]]


def _wrap_tools(tool_jsons):
    """Wrap one or more tool JSON strings in the Qwen tool-calling system section."""
    return (
        "# Tools\n\n"
        "You may call one or more functions to assist with the user query.\n\n"
        "You are provided with function signatures within <tools></tools> XML tags:\n"
        "<tools>\n"
        + "\n".join(tool_jsons)
        + "\n</tools>\n\n"
        "For each function call, return a json object with function name and arguments "
        "within <tool_call></tool_call> XML tags:\n"
        "<tool_call>\n"
        '{"name": <function-name>, "arguments": <args-json-object>}\n'
        "</tool_call>"
    )


# Every tool schema is rendered from the tables above by _build_tool; the tables are the
# single source of truth. Compared with the tool JSON in Mobile-Agent-v3.5's SYSTEM_PROMPT,
# the rendered mobile_use / computer_use schemas differ only in:
#   * mobile's function has no upstream-specific name_for_human / args_format keys;
#   * parameters are ordered by first use in the action list
#     (action, text, coordinate, time, coordinate2, ...);
#   * three upstream typos are fixed: mobile `text` "`action=answer`,and" (missing space),
#     mobile `button` missing its final period, desktop `text` "answer` and" (missing comma);
#   * desktop `coordinate`'s "Required only by" lists left_click / right_click /
#     middle_click / double_click / triple_click, which upstream omitted.
# The action bullet texts themselves are unchanged. Any edit to these tables changes the
# rendered system prompts, so checkpoints trained before the edit will be evaluated under a
# different prompt.


def _full_tool_json(tool):
    """Full schema for one tool: every action in the table, in table order."""
    return _build_tool(tool, tool["actions"])


def build_action_space(platform):
    """Full navigation/planning tool schema block for a platform (or 'all')."""
    return _wrap_tools([_full_tool_json(t) for t in _tools_for(platform)])


def build_grounding_space(platform):
    """Grounding-subset tool schema block (pointing actions only) for a platform (or 'all')."""
    return _wrap_tools(
        [_build_tool(t, [a for a in t["actions"] if a["grd"]]) for t in _tools_for(platform)]
    )


# `terminate` as the abstain action, described in terms of THIS task instead of the shared
# ACTIONS wording. Used only by build_grounding_refusal_space; the plain grounding space
# drops terminate entirely (grd=False) and the navigation/full spaces keep the shared text.
_REFUSAL_TERMINATE_DESC = (
    "Abort the task without clicking and report the outcome in `status`. Use "
    '`status="failure"` when the element named by the instruction does not exist anywhere '
    "on the screen, so no click could be correct."
)


def build_grounding_refusal_space(platform):
    """Grounding subset PLUS `terminate` -- the schema for "grounding that may abstain".

    Why this exists: OSWorld-G's 564 samples include **54 (9.6%)** whose ground truth is
    `{"refusal": -1}` -- an element that is genuinely absent, scored correct only when the
    model abstains. Without an abstain action the ceiling is 510/564 = 90.4%; a plain
    grounding checkpoint scoring 71.99% therefore loses 9.6pt of its 18.4pt gap to refusal alone.

    `terminate` carries `grd: False`, so `build_grounding_space` drops it and the plain
    `*_grounding_v1` prompts never declare it -- while their `## Note` still says "Choose
    exactly one function whose name and arguments are defined in the tool schema above".
    Measured consequence: prompting alone cannot recover the capability. Appending a prose
    "you may return terminate" clause to the frozen grounding prompt at eval time left
    that checkpoint emitting `terminate` **0 out of 564 times** and the score bit-identical
    (0.7199 -> 0.7199). The action has to be in the schema, and the schema has to be the
    same one the model was trained under -- hence this builder plus the
    `*_refusal_grounding_v1` templates below.

    Ordering note: actions keep their original ACTIONS order (terminate is already last in
    every tool's list), so the rendered enum is the grounding enum with `terminate`
    appended.

    `terminate` is re-described here (see _REFUSAL_TERMINATE_DESC): the shared ACTIONS entry
    says "Terminate the current task and report its completion status", which is about task
    completion and says nothing about an absent element -- i.e. the only textual signal that
    distinguishes this prompt from the plain one pointed the wrong way, leaving the
    "element absent -> terminate(failure)" mapping to be inferred from data alone.
    """
    return _wrap_tools([
        _build_tool(t, [
            # dict(a, ...) makes a COPY: ACTIONS is shared with the full navigation/planning
            # action spaces (build_action_space -> _build_tool), so an in-place desc edit
            # would leak into the trajectory templates.
            dict(a, desc=_REFUSAL_TERMINATE_DESC) if a["name"] == "terminate" else a
            for a in t["actions"] if a["grd"] or a["name"] == "terminate"
        ])
        for t in _tools_for(platform)
    ])


# ---- abstain: an abstain action lexically separate from task-level terminate ----------
#
# Why not reuse terminate: one model is trained on both grounding and trajectory tasks,
# and both action spaces contain terminate with status="failure":
#   planning_cot_v2   terminate(failure) = the **whole task** failed; stop the agent loop
#   refusal_grounding terminate(failure) = the target of **this step** is not on screen;
#                                          do not guess a coordinate
# The samples do not conflict (they use different conv_styles), but joint training
# conflicts at the weight level. The risk is one-directional and asymmetric: training
# "element not found -> terminate(failure)" can generalise, under a planning prompt, to the
# normal case "the target only becomes visible after scrolling" and end the whole task --
# a hard failure in an agent loop, whereas the reverse confusion only costs eval score.
#
# abstain takes no arguments, so the schema produced by build_grounding_abstain_space has
# no `status` property at all (_build_tool's param_order only collects params of included
# actions), which also removes the success / failure enum from the grounding space --
# "success" is meaningless for single-frame localisation anyway.
#
# The *_refusal_grounding_v1 templates are kept unchanged: eval/prompts.py detects abstention
# by the literal '"action": "terminate"', OSWorld-G scores 54 of its 564 samples (9.6%) on
# it, and existing checkpoints were trained under that schema. Both variants coexist, each
# with its own eval handler.
_ABSTAIN_ACTION = {
    "name": "abstain",
    "desc": ("Do not click and return no coordinate. Use this if, and only if, the "
             "element named by the instruction does not exist anywhere on the screen, "
             "so no click could be correct. This reports only about the current "
             "screenshot; it does not end the task."),
    "params": [],
    "grd": False,
}


def build_grounding_abstain_space(platform):
    """Grounding subset PLUS the no-argument `abstain` action.

    The only difference from build_grounding_refusal_space is the name and signature of
    the abstain action: there it is terminate(status="failure"), here it is `abstain`
    (no arguments). See the comment above _ABSTAIN_ACTION for why.
    """
    return _wrap_tools([
        _build_tool(t, [a for a in t["actions"] if a["grd"]] + [_ABSTAIN_ACTION])
        for t in _tools_for(platform)
    ])


grounding_system_prompt = """You are an autonomous GUI agent capable of operating on desktops, mobile devices, and web browsers. Your primary function is to analyze a screen capture, locate the target UI element, and issue the single appropriate action that fulfills the given instruction.

{action_space}

## Input Specification
- A screenshot of the current screen + an instruction describing the action to perform.

## Output Format
- Emit exactly one <tool_call></tool_call> block containing a single JSON object with the function name and its arguments.
- Coordinates are integers in a 1000x1000 space (x measured from the left edge, y from the top edge).

## Note
- Choose exactly one function whose name and arguments are defined in the tool schema above.
- Do not output anything other than the <tool_call></tool_call> block."""


# ---- the abstain contract, appended to `## Note` for the refusal templates only ----------
#
# Without this the refusal prompt and the plain one differ ONLY inside the tool schema (231
# chars in a single 3000+ char JSON line: the terminate bullet, its enum entry and the
# `status` param). Nothing in the prose says WHEN abstaining is correct, so the
# "element absent -> terminate(failure)" rule can only be inferred from the training mix --
# and it latches onto whatever superficial cue correlates with it. In one training mix,
# question-phrased inputs carried a 49.2% refusal rate vs 8.4% for imperatives (5.9x), purely
# because one family of negatives (anchor present, no neighbour in that direction) was
# phrased as questions. Balancing the phrasing of negatives fixes the data side; this clause
# is the textual anchor so the rule does not have to be guessed.
#
# Appended to the plain prompt rather than copy-pasted: the shared part is then provably
# identical, and a sync check can assert refusal == plain + clause. The two bullets
# land inside `## Note` (order within the section is not load-bearing).
#
# NOT free: pushing the model toward abstention can cost accuracy on hard-but-PRESENT
# elements. OSWorld-G is 54 refusal + 510 locatable, so always read the refusal-subset recall
# and the other-510 accuracy together, never overall alone.
GROUNDING_REFUSAL_CLAUSE = (
    "\n- If, and only if, the element named by the instruction genuinely does not exist "
    'anywhere in the screenshot, do not click: emit `terminate` with `status="failure"`.'
    "\n- Strongly prefer locating and clicking the element; abstain only when you are "
    "confident it is truly absent."
)

grounding_refusal_system_prompt = grounding_system_prompt + GROUNDING_REFUSAL_CLAUSE

# Sentence-for-sentence the same as GROUNDING_REFUSAL_CLAUSE, with only the action name
# changed. Likewise appended to the plain prompt rather than copy-pasted, so a sync check
# can assert abstain == plain + clause.
GROUNDING_ABSTAIN_CLAUSE = (
    "\n- If, and only if, the element named by the instruction genuinely does not exist "
    "anywhere in the screenshot, do not click: emit `abstain`."
    "\n- Strongly prefer locating and clicking the element; abstain only when you are "
    "confident it is truly absent."
)

grounding_abstain_system_prompt = grounding_system_prompt + GROUNDING_ABSTAIN_CLAUSE


navigation_system_prompt = """You are an autonomous GUI agent operating on the **{platform}** platform(s). Your primary function is to analyze screen captures and perform appropriate UI actions to complete assigned tasks.

{action_space}

## Input Specification
- A screenshot of the current screen + the task description + your past interaction history with the UI.

## Output Format
Response format for every step:
1) <action>...</action>: a short imperative sentence describing what to do in the UI.
2) A single <tool_call></tool_call> block containing one JSON object with the function name and its arguments.

## Note
- Output exactly in the order: the <action></action> block first, then the <tool_call></tool_call> block.
- The imperative sentence describing the action must be enclosed in <action></action> tags.
- Be brief: one sentence inside <action></action>.
- Coordinates are integers in a 1000x1000 space (x measured from the left edge, y from the top edge).
- The function call must use a function and arguments defined in the tool schema above.
- When the task is complete, call `terminate`; use `answer` when a textual answer must be returned."""


planning_system_prompt = """You are an autonomous GUI agent operating on the **{platform}** platform. Your primary function is to analyze screen captures and perform appropriate UI actions to complete assigned tasks.

{action_space}

## Input Specification
- A screenshot of the current screen + the task description + your past interaction history with the UI.

## Output Format
Response format for every step:
1) <think>...</think>: your reasoning process for the next move.
2) <action>...</action>: a short imperative sentence describing what to do in the UI.
3) A single <tool_call></tool_call> block containing one JSON object with the function name and its arguments.

## Note
- Output exactly in the order: the <think></think> block, the <action></action> block, then the <tool_call></tool_call> block.
- The imperative sentence describing the action must be enclosed in <action></action> tags.
- Coordinates are integers in a 1000x1000 space (x measured from the left edge, y from the top edge).
- The function call must use a function and arguments defined in the tool schema above.
- When the task is complete, call `terminate`; use `answer` when a textual answer must be returned."""


os_genesis_mobile_system_prompt = """You are a Mobile GUI Agent trained to assist with executing user instructions on Android devices by interacting with the graphical user interface (GUI). At each step, you will be given a high-level instruction (or a low-level reasoning step), a history of past actions, a screenshot of the current screen, and its corresponding accessibility tree. Your task is to reason about the next step and decide the most appropriate action to take using the available action space.

You must select from the following **action space**:

* `click`: Clicks at the target element.
* `long_press`: Presses and holds on the target element.
* `type`: Types the specified text at the current cursor location.
* `scroll`: Scrolls in a specified direction on the screen.
* `navigate_home`: Navigates to the device’s home screen.
* `navigate_back`: Returns to the previous screen or page.
* `open_app`: Launches the specified application.
* `wait`: Waits without taking any immediate action.
* `terminate`: Indicates the task has been completed.
* `keyboard_enter`: Presses the Enter key.


You must base your decision on the visual and structural information provided in the screenshot and accessibility tree, as well as the task instruction and previous actions. Respond only with the next step to take."""


os_genesis_web_system_prompt = """You are a highly capable Web Use Agent designed to assist with completing web-based tasks through intelligent, step-by-step interaction with user interfaces. At each step, you will receive a high-level task instruction, the current webpage screenshot, the corresponding accessibility tree, and a history of previous actions. Your objective is to reason over this input and select the most appropriate next action to progress toward completing the task.

You must select from the following **action space**:

* `click [id]`: Clicks on an element identified by its unique accessibility ID.
* `type [id] [content] [press_enter_after=0|1]`: Types the provided content into the field with the specified ID. Optionally presses "Enter" after typing (1 = press, 0 = don’t press).
* `hover [id]`: Moves the cursor to hover over the element with the given ID.
* `press [key_comb]`: Simulates keyboard input for the given key combination (e.g., `Ctrl+v`).
* `scroll [direction=down|up]`: Scrolls the page vertically in the specified direction.
* `new_tab`: Opens a new empty browser tab.
* `tab_focus [tab_index]`: Switches focus to a tab by its index.
* `close_tab`: Closes the currently focused tab.
* `goto [url]`: Navigates directly to the specified URL.
* `go_back`: Navigates to the previous page in the tab history.
* `go_forward`: Navigates to the next page if previously navigated back.
* `stop [answer]`: Ends the task. Provide the final answer in brackets, or `"N/A"` if the task is impossible to complete.


You must base your decision on the visual and structural information provided in the screenshot and accessibility tree, as well as the task instruction and previous actions. Respond only with the next step to take."""


odyssey_plus_systemp_prompt = """You are an intelligent Mobile Use Agent trained to complete goal-driven tasks on mobile interfaces by observing screen images and reasoning step-by-step. At each step, your objective is to analyze the current UI screenshot, understand the user’s high-level instruction, review prior actions, and determine the next most appropriate low-level UI operation.

**Action Space:**
You may output only one of the following actions per step:

* `"CLICK"`: Tap a specific screen coordinate.
  Format: `{ "action": "CLICK", "args": { "x": float, "y": float } }` where `x` and `y` are normalized to [0,1] relative to screen width and height.

* `"LONG_PRESS"`: Long press on the screen at a specific screen coordinate.
  Format: `{ "action": "LONG_PRESS", "args": { "x": float, "y": float } }` where `x` and `y` are normalized to [0,1] relative to screen width and height.

* `"TEXT"`: Input a text string into a selected input field.
  Format: `{ "action": "TEXT", "args": { "content": "string" } }`

* `"SCROLL"`: Perform a swipe gesture.
  Format: `{ "action": "SCROLL", "args": { "start": [x1, y1], "end": [x2, y2] } }` with coordinates normalized to [0,1].

* `"KEY_HOME"`, `"KEY_BACK"`, `"KEY_APPSELECT"`: Simulate hardware key press.
  Format: `{ "action": "KEY_HOME", "args": {} }` (and similar for others)

* `"COMPLETE"`: Task finished successfully.

  * Format: `{ "action": "COMPLETE", "args": {} }`

* `"INCOMPLETE"`: Task cannot be completed.

  * Format: `{ "action": "INCOMPLETE", "args": {} }`

**Your Input:**

* Task description
* History of prior operations
* Current UI screenshot

**Your Output:**

* Reasoning should precede the action to explain your decision-making
* One structured JSON action

Do not skip steps. Always reason before acting."""

register_conv_template(
    Conversation(
        name="internvl2_5_windows_grounding_v1",
        system_template="<|im_start|>system\n{system_message}",
        system_message=grounding_system_prompt.format(
            action_space=build_grounding_space("desktop"),
            # platform="Windows",
        ),
        roles=("<|im_start|>user\n", "<|im_start|>assistant\n"),
        sep_style=SeparatorStyle.MPT,
        sep="<|im_end|>\n",
    )
)

register_conv_template(
    Conversation(
        name="internvl2_5_ubuntu_grounding_v1",
        system_template="<|im_start|>system\n{system_message}",
        system_message=grounding_system_prompt.format(
            action_space=build_grounding_space("desktop"),
            # platform="Ubuntu",
        ),
        roles=("<|im_start|>user\n", "<|im_start|>assistant\n"),
        sep_style=SeparatorStyle.MPT,
        sep="<|im_end|>\n",
    )
)


register_conv_template(
    Conversation(
        name="internvl2_5_macos_grounding_v1",
        system_template="<|im_start|>system\n{system_message}",
        system_message=grounding_system_prompt.format(
            action_space=build_grounding_space("desktop"),
            # platform="macOS",
        ),
        roles=("<|im_start|>user\n", "<|im_start|>assistant\n"),
        sep_style=SeparatorStyle.MPT,
        sep="<|im_end|>\n",
    )
)

register_conv_template(
    Conversation(
        name="internvl2_5_web_grounding_v1",
        system_template="<|im_start|>system\n{system_message}",
        system_message=grounding_system_prompt.format(
            action_space=build_grounding_space("web"),
            # platform="Browser",
        ),
        roles=("<|im_start|>user\n", "<|im_start|>assistant\n"),
        sep_style=SeparatorStyle.MPT,
        sep="<|im_end|>\n",
    )
)

# ---- grounding that may abstain (see build_grounding_refusal_space) ----------------
#
# NAMING IS LOAD-BEARING: preprocess_conversation_format in data_qwen.py dispatches on the substring
# `"_grounding_v" in conv_style` to (a) assert the gpt turn carries a <tool_call> and
# (b) run transform_tool_call_coordinates. A name like
# `internvl2_5_ubuntu_grounding_refusal_v1` does NOT contain `_grounding_v` (it reads
# `_grounding_r`), so the coordinate transform would be skipped -- silently, because with
# coord_norm=True the transform happens to be the identity. It would only break under
# coord_norm=False, where coordinates would stay in [0,1000] instead of being scaled to
# pixels, with no error anywhere. So keep `_grounding_v1` as the tail:
#     internvl2_5_<os>_refusal_grounding_v1        <- correct
#     internvl2_5_<os>_grounding_refusal_v1        <- WRONG, breaks the dispatch
#
# The other two dispatch sites are safe with this name: the system-prompt whitelist is an
# EXACT match (so system_message is not replaced by "You are a helpful assistant."), and
# the <ref>/<box> path keys on startswith(("internvl_grounding", "internvl_referring")),
# which `internvl2_5_*` does not hit (so it will not be routed down the <ref>/<box> path).
#
# Platform argument follows the existing convention of the plain grounding templates:
# ubuntu -> "desktop", web -> "web", android -> "mobile".
#
# desktop / mobile are ALIASES of ubuntu / android: build_grounding_refusal_space takes the
# tool platform, and no `platform=` is passed into the prompt, so the four rendered
# system_messages collapse into two distinct strings (desktop==ubuntu, mobile==android).
# They exist so a trajectory converter can key every output file on its own bucket name
# (desktop / web / mobile) instead of translating desktop->ubuntu and mobile->android when
# writing the data meta file, where a typo would silently fall back to "You are a helpful
# assistant." via the exact-match whitelist in data_qwen.py. Both name sets are in use.
for _refusal_os, _refusal_tool_platform in (
    ("ubuntu", "desktop"),
    ("desktop", "desktop"),
    ("web", "web"),
    ("android", "mobile"),
    ("mobile", "mobile"),
):
    register_conv_template(
        Conversation(
            name=f"internvl2_5_{_refusal_os}_refusal_grounding_v1",
            system_template="<|im_start|>system\n{system_message}",
            # grounding_REFUSAL_system_prompt: plain prompt + GROUNDING_REFUSAL_CLAUSE. The
            # plain `grounding_system_prompt` stays byte-identical for the plain templates.
            system_message=grounding_refusal_system_prompt.format(
                action_space=build_grounding_refusal_space(_refusal_tool_platform),
            ),
            roles=("<|im_start|>user\n", "<|im_start|>assistant\n"),
            sep_style=SeparatorStyle.MPT,
            sep="<|im_end|>\n",
        )
    )
del _refusal_os, _refusal_tool_platform


# Abstain templates. The os aliases are exactly those of the refusal set (ubuntu/desktop and
# android/mobile each render to the same string; they let a converter use its own bucket
# names without translating desktop->ubuntu -- a mistranslation would silently degrade to
# "You are a helpful assistant." via the exact-match whitelist in data_qwen.py).
# The name must keep the `_grounding_v` substring: data_qwen.py uses it to decide whether to
# rescale the coordinates inside <tool_call>, and a miss is **silently skipped** (with
# coord_norm=True the transform happens to be the identity; it only shows up with
# coord_norm=False). `_abstain_grounding_v1` matches.
for _abstain_os, _abstain_tool_platform in (
    ("ubuntu", "desktop"),
    ("desktop", "desktop"),
    ("web", "web"),
    ("android", "mobile"),
    ("mobile", "mobile"),
):
    register_conv_template(
        Conversation(
            name=f"internvl2_5_{_abstain_os}_abstain_grounding_v1",
            system_template="<|im_start|>system\n{system_message}",
            system_message=grounding_abstain_system_prompt.format(
                action_space=build_grounding_abstain_space(_abstain_tool_platform),
            ),
            roles=("<|im_start|>user\n", "<|im_start|>assistant\n"),
            sep_style=SeparatorStyle.MPT,
            sep="<|im_end|>\n",
        )
    )
del _abstain_os, _abstain_tool_platform


register_conv_template(
    Conversation(
        name="internvl2_5_mind2web_grounding_v1",
        system_template="<|im_start|>system\n{system_message}",
        system_message=grounding_system_prompt.format(
            action_space=build_grounding_space("mind2web"),
            platform="Browser",
        ),
        roles=("<|im_start|>user\n", "<|im_start|>assistant\n"),
        sep_style=SeparatorStyle.MPT,
        sep="<|im_end|>\n",
    )
)

register_conv_template(
    Conversation(
        name="internvl2_5_android_grounding_v1",
        system_template="<|im_start|>system\n{system_message}",
        system_message=grounding_system_prompt.format(
            action_space=build_grounding_space("mobile"),
            # platform="Android",
        ),
        roles=("<|im_start|>user\n", "<|im_start|>assistant\n"),
        sep_style=SeparatorStyle.MPT,
        sep="<|im_end|>\n",
    )
)

register_conv_template(
    Conversation(
        name="internvl2_5_iphone_grounding_v1",
        system_template="<|im_start|>system\n{system_message}",
        system_message=grounding_system_prompt.format(
            action_space=build_grounding_space("mobile"),
            # platform="iOS",
        ),
        roles=("<|im_start|>user\n", "<|im_start|>assistant\n"),
        sep_style=SeparatorStyle.MPT,
        sep="<|im_end|>\n",
    )
)

register_conv_template(
    Conversation(
        name="internvl2_5_ipad_grounding_v1",
        system_template="<|im_start|>system\n{system_message}",
        system_message=grounding_system_prompt.format(
            action_space=build_grounding_space("mobile"),
            # platform="iPadOS",
        ),
        roles=("<|im_start|>user\n", "<|im_start|>assistant\n"),
        sep_style=SeparatorStyle.MPT,
        sep="<|im_end|>\n",
    )
)


register_conv_template(
    Conversation(
        name="internvl2_5_mobile_grounding_v1",
        system_template="<|im_start|>system\n{system_message}",
        system_message=grounding_system_prompt.format(
            action_space=build_grounding_space("mobile"),
            # platform="Phone & Tablet",
        ),
        roles=("<|im_start|>user\n", "<|im_start|>assistant\n"),
        sep_style=SeparatorStyle.MPT,
        sep="<|im_end|>\n",
    )
)


register_conv_template(
    Conversation(
        name="internvl2_5_all_grounding_v1",
        system_template="<|im_start|>system\n{system_message}",
        system_message=grounding_system_prompt.format(
            action_space=build_grounding_space("all"),
            # platform="Windows & Ubuntu & Browser & Phone & Tablet",
        ),
        roles=("<|im_start|>user\n", "<|im_start|>assistant\n"),
        sep_style=SeparatorStyle.MPT,
        sep="<|im_end|>\n",
    )
)


register_conv_template(
    Conversation(
        name="internvl2_5_android_navigation_v1",
        system_template="<|im_start|>system\n{system_message}",
        system_message=navigation_system_prompt.format(
            action_space=build_action_space("mobile"),
            platform="Android",
        ),
        roles=("<|im_start|>user\n", "<|im_start|>assistant\n"),
        sep_style=SeparatorStyle.MPT,
        sep="<|im_end|>\n",
    )
)


register_conv_template(
    Conversation(
        name="internvl2_5_android_planning_cot_v1",
        system_template="<|im_start|>system\n{system_message}",
        system_message=planning_system_prompt.format(
            action_space=build_action_space("mobile"),
            platform="Android",
        ),
        roles=("<|im_start|>user\n", "<|im_start|>assistant\n"),
        sep_style=SeparatorStyle.MPT,
        sep="<|im_end|>\n",
    )
)


register_conv_template(
    Conversation(
        name="internvl2_5_android_planning_cot_v2",
        system_template="<|im_start|>system\n{system_message}",
        system_message=planning_system_prompt.format(
            action_space=build_action_space("mobile"),
            platform="Android",
        ),
        roles=("<|im_start|>user\n", "<|im_start|>assistant\n"),
        sep_style=SeparatorStyle.MPT,
        sep="<|im_end|>\n",
    )
)


register_conv_template(
    Conversation(
        name="internvl2_5_mobile_navigation_v1",
        system_template="<|im_start|>system\n{system_message}",
        system_message=navigation_system_prompt.format(
            action_space=build_action_space("mobile"),
            platform="Mobile Phone & Tablet",
        ),
        roles=("<|im_start|>user\n", "<|im_start|>assistant\n"),
        sep_style=SeparatorStyle.MPT,
        sep="<|im_end|>\n",
    )
)

register_conv_template(
    Conversation(
        name="internvl2_5_mobile_planning_cot_v1",
        system_template="<|im_start|>system\n{system_message}",
        system_message=planning_system_prompt.format(
            action_space=build_action_space("mobile"),
            platform="Mobile Phone & Tablet",
        ),
        roles=("<|im_start|>user\n", "<|im_start|>assistant\n"),
        sep_style=SeparatorStyle.MPT,
        sep="<|im_end|>\n",
    )
)

register_conv_template(
    Conversation(
        name="internvl2_5_mobile_planning_cot_v2",
        system_template="<|im_start|>system\n{system_message}",
        system_message=planning_system_prompt.format(
            action_space=build_action_space("mobile"),
            platform="Mobile Phone & Tablet",
        ),
        roles=("<|im_start|>user\n", "<|im_start|>assistant\n"),
        sep_style=SeparatorStyle.MPT,
        sep="<|im_end|>\n",
    )
)

register_conv_template(
    Conversation(
        name="internvl2_5_web_navigation_v1",
        system_template="<|im_start|>system\n{system_message}",
        system_message=navigation_system_prompt.format(
            action_space=build_action_space("web"),
            platform="Web Browser",
        ),
        roles=("<|im_start|>user\n", "<|im_start|>assistant\n"),
        sep_style=SeparatorStyle.MPT,
        sep="<|im_end|>\n",
    )
)


register_conv_template(
    Conversation(
        name="internvl2_5_web_planning_cot_v1",
        system_template="<|im_start|>system\n{system_message}",
        system_message=planning_system_prompt.format(
            action_space=build_action_space("web"),
            platform="Web Browser",
        ),
        roles=("<|im_start|>user\n", "<|im_start|>assistant\n"),
        sep_style=SeparatorStyle.MPT,
        sep="<|im_end|>\n",
    )
)


register_conv_template(
    Conversation(
        name="internvl2_5_web_planning_cot_v2",
        system_template="<|im_start|>system\n{system_message}",
        system_message=planning_system_prompt.format(
            action_space=build_action_space("web"),
            platform="Web Browser",
        ),
        roles=("<|im_start|>user\n", "<|im_start|>assistant\n"),
        sep_style=SeparatorStyle.MPT,
        sep="<|im_end|>\n",
    )
)


register_conv_template(
    Conversation(
        name="internvl2_5_mind2web_navigation_v1",
        system_template="<|im_start|>system\n{system_message}",
        system_message=navigation_system_prompt.format(
            action_space=build_action_space("mind2web"),
            platform="Mind2Web",
        ),
        roles=("<|im_start|>user\n", "<|im_start|>assistant\n"),
        sep_style=SeparatorStyle.MPT,
        sep="<|im_end|>\n",
    )
)

register_conv_template(
    Conversation(
        name="internvl2_5_mind2web_planning_cot_v1",
        system_template="<|im_start|>system\n{system_message}",
        system_message=planning_system_prompt.format(
            action_space=build_action_space("mind2web"),
            platform="Mind2Web",
        ),
        roles=("<|im_start|>user\n", "<|im_start|>assistant\n"),
        sep_style=SeparatorStyle.MPT,
        sep="<|im_end|>\n",
    )
)

register_conv_template(
    Conversation(
        name="internvl2_5_ubuntu_navigation_v1",
        system_template="<|im_start|>system\n{system_message}",
        system_message=navigation_system_prompt.format(
            action_space=build_action_space("desktop"),
            platform="Linux (Ubuntu)",
        ),
        roles=("<|im_start|>user\n", "<|im_start|>assistant\n"),
        sep_style=SeparatorStyle.MPT,
        sep="<|im_end|>\n",
    )
)

register_conv_template(
    Conversation(
        name="internvl2_5_ubuntu_planning_cot_v1",
        system_template="<|im_start|>system\n{system_message}",
        system_message=planning_system_prompt.format(
            action_space=build_action_space("desktop"),
            platform="Linux (Ubuntu)",
        ),
        roles=("<|im_start|>user\n", "<|im_start|>assistant\n"),
        sep_style=SeparatorStyle.MPT,
        sep="<|im_end|>\n",
    )
)

register_conv_template(
    Conversation(
        name="internvl2_5_ubuntu_planning_cot_v2",
        system_template="<|im_start|>system\n{system_message}",
        system_message=planning_system_prompt.format(
            action_space=build_action_space("desktop"),
            platform="Linux (Ubuntu)",
        ),
        roles=("<|im_start|>user\n", "<|im_start|>assistant\n"),
        sep_style=SeparatorStyle.MPT,
        sep="<|im_end|>\n",
    )
)


register_conv_template(
    Conversation(
        name="internvl2_5_windows_navigation_v1",
        system_template="<|im_start|>system\n{system_message}",
        system_message=navigation_system_prompt.format(
            action_space=build_action_space("desktop"),
            platform="Windows",
        ),
        roles=("<|im_start|>user\n", "<|im_start|>assistant\n"),
        sep_style=SeparatorStyle.MPT,
        sep="<|im_end|>\n",
    )
)


register_conv_template(
    Conversation(
        name="internvl2_5_windows_planning_cot_v1",
        system_template="<|im_start|>system\n{system_message}",
        system_message=planning_system_prompt.format(
            action_space=build_action_space("desktop"),
            platform="Windows",
        ),
        roles=("<|im_start|>user\n", "<|im_start|>assistant\n"),
        sep_style=SeparatorStyle.MPT,
        sep="<|im_end|>\n",
    )
)

register_conv_template(
    Conversation(
        name="internvl2_5_windows_planning_cot_v2",
        system_template="<|im_start|>system\n{system_message}",
        system_message=planning_system_prompt.format(
            action_space=build_action_space("desktop"),
            platform="Windows",
        ),
        roles=("<|im_start|>user\n", "<|im_start|>assistant\n"),
        sep_style=SeparatorStyle.MPT,
        sep="<|im_end|>\n",
    )
)

register_conv_template(
    Conversation(
        name="internvl2_5_mac_navigation_v1",
        system_template="<|im_start|>system\n{system_message}",
        system_message=navigation_system_prompt.format(
            action_space=build_action_space("desktop"),
            platform="macOS",
        ),
        roles=("<|im_start|>user\n", "<|im_start|>assistant\n"),
        sep_style=SeparatorStyle.MPT,
        sep="<|im_end|>\n",
    )
)


register_conv_template(
    Conversation(
        name="internvl2_5_mac_planning_cot_v1",
        system_template="<|im_start|>system\n{system_message}",
        system_message=planning_system_prompt.format(
            action_space=build_action_space("desktop"),
            platform="macOS",
        ),
        roles=("<|im_start|>user\n", "<|im_start|>assistant\n"),
        sep_style=SeparatorStyle.MPT,
        sep="<|im_end|>\n",
    )
)


register_conv_template(
    Conversation(
        name="internvl2_5_mac_planning_cot_v2",
        system_template="<|im_start|>system\n{system_message}",
        system_message=planning_system_prompt.format(
            action_space=build_action_space("desktop"),
            platform="macOS",
        ),
        roles=("<|im_start|>user\n", "<|im_start|>assistant\n"),
        sep_style=SeparatorStyle.MPT,
        sep="<|im_end|>\n",
    )
)


register_conv_template(
    Conversation(
        name="internvl2_5_desktop_planning_cot_v1",
        system_template="<|im_start|>system\n{system_message}",
        system_message=planning_system_prompt.format(
            action_space=build_action_space("desktop"),
            platform="desktop (Windows/Linux/macOS)",
        ),
        roles=("<|im_start|>user\n", "<|im_start|>assistant\n"),
        sep_style=SeparatorStyle.MPT,
        sep="<|im_end|>\n",
    )
)


register_conv_template(
    Conversation(
        name="internvl2_5_desktop_planning_cot_v2",
        system_template="<|im_start|>system\n{system_message}",
        system_message=planning_system_prompt.format(
            action_space=build_action_space("desktop"),
            platform="desktop (Windows/Linux/macOS)",
        ),
        roles=("<|im_start|>user\n", "<|im_start|>assistant\n"),
        sep_style=SeparatorStyle.MPT,
        sep="<|im_end|>\n",
    )
)


register_conv_template(
    Conversation(
        name="internvl2_5_desktop_navigation_v1",
        system_template="<|im_start|>system\n{system_message}",
        system_message=navigation_system_prompt.format(
            action_space=build_action_space("desktop"),
            platform="desktop (Windows/Linux/macOS)",
        ),
        roles=("<|im_start|>user\n", "<|im_start|>assistant\n"),
        sep_style=SeparatorStyle.MPT,
        sep="<|im_end|>\n",
    )
)


register_conv_template(
    Conversation(
        name="internvl2_5_mobile_os_genesis",
        system_template="<|im_start|>system\n{system_message}",
        system_message=os_genesis_mobile_system_prompt,
        roles=("<|im_start|>user\n", "<|im_start|>assistant\n"),
        sep_style=SeparatorStyle.MPT,
        sep="<|im_end|>\n",
    )
)


register_conv_template(
    Conversation(
        name="internvl2_5_web_os_genesis",
        system_template="<|im_start|>system\n{system_message}",
        system_message=os_genesis_web_system_prompt,
        roles=("<|im_start|>user\n", "<|im_start|>assistant\n"),
        sep_style=SeparatorStyle.MPT,
        sep="<|im_end|>\n",
    )
)

register_conv_template(
    Conversation(
        name="internvl2_5_mobile_odyssey_plus",
        system_template="<|im_start|>system\n{system_message}",
        system_message=odyssey_plus_systemp_prompt,
        roles=("<|im_start|>user\n", "<|im_start|>assistant\n"),
        sep_style=SeparatorStyle.MPT,
        sep="<|im_end|>\n",
    )
)

# === Reflective planning system prompt ===
reflective_planning_system_prompt = """You are an autonomous GUI agent operating on the **{platform}** platform. Your primary function is to analyze screen captures and perform appropriate UI actions to complete assigned tasks.

{action_space}

## Input Specification
- Screenshot of the current screen + task description + your past interaction history with UI to finish assigned tasks.

## Output Format
Response format for every step:
1) <reflection>...</reflection>: assess whether your previous action achieved its intended effect, based on the current screenshot. For the first step, state that there is no previous action.
2) <think>...</think>: your reasoning process for the next move.
3) <action>...</action>: a short imperative sentence describing what to do in the UI.
4) A single <tool_call></tool_call> block containing one JSON object with the function name and its arguments.

## Note
- Avoid actions that would lead to invalid states.
- The function call must use a function and arguments defined in the tool schema above.
- The imperative sentence describing the action must be enclosed in <action></action> tags.
- Output exactly in the order: the <reflection></reflection> block, the <think></think> block, the <action></action> block, then the <tool_call></tool_call> block."""


# === Failure detection / outcome verification system prompt ===
verification_system_prompt = """You are evaluating the outcome of a GUI action performed on the **{platform}** platform. Your function is to judge, from the screenshots, whether the action achieved its intended effect.

## Input Specification
- The screen before and after an action, together with a description of that action.

## Output Format
```
<reflection>
[Describe what changed on the screen and whether the action achieved its goal]
</reflection>
<result>
[Exactly one of: success / failure / redundant]
</result>
```

## Note
- "success": the action achieved its intended effect.
- "failure": the action did not achieve its intended effect.
- "redundant": the action had no necessary effect (e.g. the target state already held).
- The assessment and verdict should be enclosed within <reflection></reflection> and <result></result> tags, respectively."""


register_conv_template(
    Conversation(
        name="internvl2_5_desktop_reflective_planning_cot_v1",
        system_template="<|im_start|>system\n{system_message}",
        system_message=reflective_planning_system_prompt.format(
            action_space=build_action_space("desktop"),
            platform="desktop (Windows / macOS / Linux)",
        ),
        roles=("<|im_start|>user\n", "<|im_start|>assistant\n"),
        sep_style=SeparatorStyle.MPT,
        sep="<|im_end|>\n",
    )
)


register_conv_template(
    Conversation(
        name="internvl2_5_desktop_verification_v1",
        system_template="<|im_start|>system\n{system_message}",
        system_message=verification_system_prompt.format(
            platform="desktop (Windows / macOS / Linux)",
        ),
        roles=("<|im_start|>user\n", "<|im_start|>assistant\n"),
        sep_style=SeparatorStyle.MPT,
        sep="<|im_end|>\n",
    )
)


# === Forward dynamics (image_t + action -> text description of the next state) ===
forward_dynamics_system_prompt = """You are a world model of a GUI on the **{platform}** platform. Given the current screenshot and an action, predict the resulting screen state described in natural language.

## Input Specification
- The current screenshot + the action that will be performed.

## Output Format
```
<next_state>
[Natural-language description of the screen after the action is performed]
</next_state>
```

## Note
- Describe the resulting state, not the action itself.
- The description should be enclosed within <next_state></next_state> tags."""


# === Forward dynamics v2 ===
# (task + action history + current screen + current action as <action> + <tool_call>
#  -> next_state)
forward_dynamics_v2_system_prompt = """You are a world model of a GUI on the **{platform}** platform. Given the task, the history of past operations, the current screenshot, and the action about to be performed (an <action></action> description and a <tool_call> function call), you predict the change to the next screen that the action produces, described in natural language.

{action_space}

## Input Specification
- The task being pursued and the history of past operations, for context.
- A screenshot of the current screen.
- The next action to be performed, given as an <action></action> description (optional) and a <tool_call> function call from the tool schema above.

## Output Format
```
<state_change>
[Natural-language description of the change to the screen produced by the action]
</state_change>
```

## Note
- Describe only the change the action produces, not the action itself and not the unchanged parts of the screen.
- Ground the action in the current screenshot: locate where the <tool_call> function call operates and reason about its effect on that screen.
- The description must be enclosed within <state_change></state_change> tags."""


register_conv_template(
    Conversation(
        name="internvl2_5_desktop_forward_dynamics_v2",
        system_template="<|im_start|>system\n{system_message}",
        system_message=forward_dynamics_v2_system_prompt.format(
            action_space=build_action_space("desktop"),
            platform="desktop (Windows / macOS / Linux)",
        ),
        roles=("<|im_start|>user\n", "<|im_start|>assistant\n"),
        sep_style=SeparatorStyle.MPT,
        sep="<|im_end|>\n",
    )
)


register_conv_template(
    Conversation(
        name="internvl2_5_web_forward_dynamics_v2",
        system_template="<|im_start|>system\n{system_message}",
        system_message=forward_dynamics_v2_system_prompt.format(
            action_space=build_action_space("web"),
            platform="Web Browser",
        ),
        roles=("<|im_start|>user\n", "<|im_start|>assistant\n"),
        sep_style=SeparatorStyle.MPT,
        sep="<|im_end|>\n",
    )
)


register_conv_template(
    Conversation(
        name="internvl2_5_mobile_forward_dynamics_v2",
        system_template="<|im_start|>system\n{system_message}",
        system_message=forward_dynamics_v2_system_prompt.format(
            action_space=build_action_space("mobile"),
            platform="mobile (Android / iOS)",
        ),
        roles=("<|im_start|>user\n", "<|im_start|>assistant\n"),
        sep_style=SeparatorStyle.MPT,
        sep="<|im_end|>\n",
    )
)


# === Inverse dynamics (before/after frames -> action) ===
inverse_dynamics_system_prompt = """You are an autonomous GUI agent operating on the **{platform}** platform. Given the screen before and after a single action, infer which action caused the transition.

{action_space}

## Input Specification
- The screenshot before the action and the screenshot after the action.

## Output Format
Response format:
1) <action>...</action>: a short imperative sentence describing the inferred operation.
2) A single <tool_call></tool_call> block containing one JSON object with the function name and its arguments.

## Note
- The inferred function call must use a function and arguments defined in the tool schema above.
- The imperative sentence describing the action must be enclosed in <action></action> tags.
- Output exactly in the order: the <action></action> block, then the <tool_call></tool_call> block."""


register_conv_template(
    Conversation(
        name="internvl2_5_desktop_forward_dynamics_v1",
        system_template="<|im_start|>system\n{system_message}",
        system_message=forward_dynamics_system_prompt.format(
            platform="desktop (Windows / macOS / Linux)",
        ),
        roles=("<|im_start|>user\n", "<|im_start|>assistant\n"),
        sep_style=SeparatorStyle.MPT,
        sep="<|im_end|>\n",
    )
)


register_conv_template(
    Conversation(
        name="internvl2_5_desktop_inverse_dynamics_v1",
        system_template="<|im_start|>system\n{system_message}",
        system_message=inverse_dynamics_system_prompt.format(
            action_space=build_action_space("desktop"),
            platform="desktop (Windows / macOS / Linux)",
        ),
        roles=("<|im_start|>user\n", "<|im_start|>assistant\n"),
        sep_style=SeparatorStyle.MPT,
        sep="<|im_end|>\n",
    )
)


# === Task planning decomposition (first screen + instruction -> high-level plan) ===
task_planning_system_prompt = """You are an autonomous GUI agent operating on the **{platform}** platform. Given the current screen and a task, produce a high-level step-by-step plan to accomplish it.

## Input Specification
- The current screenshot + the task description.

## Output Format
```
<plan>
[A high-level, numbered step-by-step plan: step1: ... step2: ...]
</plan>
```

## Note
- Provide a concise high-level plan, not low-level coordinate actions.
- The plan should be enclosed within <plan></plan> tags."""


# --- Forward dynamics, mobile (reuses forward_dynamics_system_prompt) ---
register_conv_template(
    Conversation(
        name="internvl2_5_mobile_forward_dynamics_v1",
        system_template="<|im_start|>system\n{system_message}",
        system_message=forward_dynamics_system_prompt.format(
            platform="mobile (Android / iOS)",
        ),
        roles=("<|im_start|>user\n", "<|im_start|>assistant\n"),
        sep_style=SeparatorStyle.MPT,
        sep="<|im_end|>\n",
    )
)


# --- Inverse dynamics, mobile (inverse_dynamics_system_prompt + mobile action space) ---
register_conv_template(
    Conversation(
        name="internvl2_5_mobile_inverse_dynamics_v1",
        system_template="<|im_start|>system\n{system_message}",
        system_message=inverse_dynamics_system_prompt.format(
            action_space=build_action_space("mobile"),
            platform="mobile (Android / iOS)",
        ),
        roles=("<|im_start|>user\n", "<|im_start|>assistant\n"),
        sep_style=SeparatorStyle.MPT,
        sep="<|im_end|>\n",
    )
)


# --- Task planning, desktop ---
register_conv_template(
    Conversation(
        name="internvl2_5_desktop_task_planning_v1",
        system_template="<|im_start|>system\n{system_message}",
        system_message=task_planning_system_prompt.format(
            platform="desktop (Windows / macOS / Linux)",
        ),
        roles=("<|im_start|>user\n", "<|im_start|>assistant\n"),
        sep_style=SeparatorStyle.MPT,
        sep="<|im_end|>\n",
    )
)


# --- Task planning, mobile ---
register_conv_template(
    Conversation(
        name="internvl2_5_mobile_task_planning_v1",
        system_template="<|im_start|>system\n{system_message}",
        system_message=task_planning_system_prompt.format(
            platform="mobile (Android / iOS)",
        ),
        roles=("<|im_start|>user\n", "<|im_start|>assistant\n"),
        sep_style=SeparatorStyle.MPT,
        sep="<|im_end|>\n",
    )
)


# --- Web variants of forward dynamics / inverse dynamics / task planning (v1) ---
register_conv_template(
    Conversation(
        name="internvl2_5_web_forward_dynamics_v1",
        system_template="<|im_start|>system\n{system_message}",
        system_message=forward_dynamics_system_prompt.format(
            platform="Web Browser",
        ),
        roles=("<|im_start|>user\n", "<|im_start|>assistant\n"),
        sep_style=SeparatorStyle.MPT,
        sep="<|im_end|>\n",
    )
)


register_conv_template(
    Conversation(
        name="internvl2_5_web_inverse_dynamics_v1",
        system_template="<|im_start|>system\n{system_message}",
        system_message=inverse_dynamics_system_prompt.format(
            action_space=build_action_space("web"),
            platform="Web Browser",
        ),
        roles=("<|im_start|>user\n", "<|im_start|>assistant\n"),
        sep_style=SeparatorStyle.MPT,
        sep="<|im_end|>\n",
    )
)


register_conv_template(
    Conversation(
        name="internvl2_5_web_task_planning_v1",
        system_template="<|im_start|>system\n{system_message}",
        system_message=task_planning_system_prompt.format(
            platform="Web Browser",
        ),
        roles=("<|im_start|>user\n", "<|im_start|>assistant\n"),
        sep_style=SeparatorStyle.MPT,
        sep="<|im_end|>\n",
    )
)


register_conv_template(
    Conversation(
        name="internvl2_5_desktop_grounding_v1",
        system_template="<|im_start|>system\n{system_message}",
        system_message=grounding_system_prompt.format(
            action_space=build_grounding_space("desktop"),
        ),
        roles=("<|im_start|>user\n", "<|im_start|>assistant\n"),
        sep_style=SeparatorStyle.MPT,
        sep="<|im_end|>\n",
    )
)


# === Understanding: referring VQA (appearance / function / layout / OCR) ===
# The question refers to an element by bbox; the answer is free text.
# The name starts with internvl_referring, so data_qwen.py (1) uses this template's own
# system_message (it is not in the generic "You are a helpful assistant." whitelist) and
# (2) applies format_grounding_internvl2qwenvl's referring coordinate transform. Answers
# contain no coordinates.
# The coordinate transform depends on coord_norm (data_qwen.py: coord_size):
#   - coord_norm=True  -> coord_size=(1000,1000), identity: the [0,1000] boxes in the
#     question stay [0,1000], consistent with "integers in a 1000x1000 space" in the prompt.
#   - coord_norm=False -> coord_size=resized pixels: the boxes are scaled to absolute pixels,
#     which contradicts the hard-coded "1000x1000" in this prompt. So this conv_style must
#     only be used with coord_norm=True (it is used only in continued-pretraining data).
understanding_system_prompt = """You are a GUI understanding assistant operating on desktops, mobile devices, and web browsers. Given a screenshot and a question that refers to a specific UI element by its bounding box, you describe or read that element as asked.

## Input Specification
- A screenshot of the current screen + a question that refers to an element via its bounding box <box>[[x1, y1, x2, y2]]</box>.
- The bounding box coordinates are integers in a 1000x1000 space (x from the left edge, y from the top edge).

## Note
- Answer only what is asked: the element's appearance, its functionality, its on-screen layout, or the literal text it displays.
- Respond with a concise natural-language answer. Do not output any action, code, or coordinates."""


register_conv_template(
    Conversation(
        name="internvl_referring_understanding_v1",
        system_template="<|im_start|>system\n{system_message}",
        system_message=understanding_system_prompt,
        roles=("<|im_start|>user\n", "<|im_start|>assistant\n"),
        sep_style=SeparatorStyle.MPT,
        sep="<|im_end|>\n",
    )
)


# === Knowledge: four screen knowledge / perception tasks ===
# Coordinates appear only in some answers (multiple boxes for screen_ocr, one box for
# spatial_rel); questions carry no coordinates.
# screen_ocr / spatial_rel start with internvl_grounding: the <ref>text</ref><box>[[...]]</box>
# pairs (per-mille) in the gpt answer are rewritten by format_grounding_internvl2qwenvl /
# find_bbox into a ```json [{"bbox_2d": [rescaled coordinates], "label": text}]``` block;
# not being exact whitelist names, they keep their own system_message.
# existence / negatives are plain text (Yes/No, refusal sentences) and use internvl_knowledge_*
# names, so no coordinate transform is applied.

screen_ocr_system_prompt = """You are a GUI perception assistant operating on desktops, mobile devices, and web browsers. Given a screenshot, read all visible on-screen text and localize each piece with its bounding box.

## Output Format
Return a JSON array; each item is the recognized text and its bounding box [x1, y1, x2, y2]:
```json
[{"bbox_2d": [x1, y1, x2, y2], "label": "<text>"}, ...]
```

## Note
- Cover every legible text element; do not invent text that is not visible.
- All bounding-box coordinates are integers in a 1000x1000 space (x from the left edge, y from the top edge).
- Output only the JSON array."""


spatial_rel_system_prompt = """You are a GUI spatial-reasoning assistant operating on desktops, mobile devices, and web browsers. You answer questions about the relative position of on-screen elements, which are referenced by their text.

## Output Format
- Directional question (left / right / above / below): answer with the single relation word.
- Locate question (e.g. "the element immediately below X"): return the element's text and its bounding box:
```json
[{"bbox_2d": [x1, y1, x2, y2], "label": "<text>"}]
```
- Ordering question (e.g. "arrange the following elements from left to right"): return one entry per named element, sorted along the requested axis:
```json
[{"bbox_2d": [x1, y1, x2, y2], "label": "<text>"}, {"bbox_2d": [x1, y1, x2, y2], "label": "<text>"}, ...]
```

## Note
- Base your answer only on what is visible; add no extra commentary.
- For an ordering question, return every element the question names and no others; the order of the array IS the answer.
- All bounding-box coordinates are integers in a 1000x1000 space (x from the left edge, y from the top edge)."""


existence_system_prompt = """You are a GUI perception assistant operating on desktops, mobile devices, and web browsers. Given a screenshot and a question asking whether a specific element or text is present on the screen, verify its presence.

## Note
- Answer strictly with "Yes" or "No".
- Say "Yes" only if the queried element is actually visible on the screen."""


negatives_system_prompt = """You are a GUI agent operating on desktops, mobile devices, and web browsers. Given a screenshot and an instruction that refers to an element or an ordinal occurrence that may not exist, judge feasibility before acting.

## Note
- If the referenced element or the requested occurrence does not exist on the screen, do not fabricate an action; reply that it cannot be performed and briefly state why (the element is absent, or only N such elements exist).
- Keep the reply to a single concise sentence."""


# Element grounding with a BOX target: given a referring expression in <ref>...</ref>
# (the element's visible text, or a description of it), return that one element's
# bounding box. Used instead of the bare "internvl_grounding" style, whose
# system_message is the Chinese InternVL identity prompt AND which data_qwen.py
# force-overrides to a generic "You are a helpful assistant." -- neither says anything
# about GUI grounding or about the [0,1000] coordinate space the target actually uses.
#
# Naming constraints (both load-bearing, do not rename casually):
#   * MUST start with "internvl_grounding" so preprocess_conversation_format routes the
#     <ref>..</ref><box>[[..]]</box> gpt target through format_grounding_internvl2qwenvl
#     -> find_bbox (which rewrites it to the bbox_2d JSON described below and rescales
#     the coordinates into coord_size).
#   * MUST NOT contain the substring "_grounding_v", or the same function would also take
#     the GUI-action path and assert that the gpt turn carries <tool_call>/<action>.
#   * MUST NOT be added to the exact-match whitelist in data_qwen.py, or this
#     system_message would be discarded in favour of "You are a helpful assistant.".
element_box_system_prompt = """You are a GUI grounding assistant operating on desktops, mobile devices, and web browsers. Given a screenshot and a referring expression that identifies one on-screen element, locate that element and return its bounding box.

## Input Specification
- A screenshot of the current screen + a referring expression wrapped in <ref>...</ref>. The expression is either the element's visible text or a short description of it.

## Output Format
Return a JSON array holding the single referred element and its bounding box [x1, y1, x2, y2]:
```json
[{"bbox_2d": [x1, y1, x2, y2], "label": "<referring expression>"}]
```

## Note
- Return exactly one element: the one the referring expression names. Do not list other elements.
- The box must tightly enclose the element, not the container or row holding it.
- All bounding-box coordinates are integers in a 1000x1000 space (x from the left edge, y from the top edge).
- Output only the JSON array."""


register_conv_template(
    Conversation(
        name="internvl_grounding_element_box_v1",
        system_template="<|im_start|>system\n{system_message}",
        system_message=element_box_system_prompt,
        roles=("<|im_start|>user\n", "<|im_start|>assistant\n"),
        sep_style=SeparatorStyle.MPT,
        sep="<|im_end|>\n",
    )
)


# Spatial-reference grounding with a BOX target. Sibling of the style above, split out
# because the referring expression has a different shape and a different failure mode:
# the expression names an ANCHOR element by its text and asks for the element at a stated
# position relative to it ('the control immediately left of "Home"'), so it mentions TWO
# elements and only one of them is the target.
#
# Why a box target for this task at all: measured on an SFT checkpoint, UI-Vision
# spatial/text puts 23.9% of all samples within half a box-diagonal of the ground truth
# (the highest of any slice) and 71.4% of its failures land within two diagonals -- i.e.
# the model resolves the relation roughly right and then misses the element's extent.
# A point target gives no gradient about where an element ends, so that residual is
# exactly what box supervision can reach. The paired point task is kept (it is what the
# benchmarks actually score); this is the auxiliary half.
#
# Same three naming constraints as internvl_grounding_element_box_v1 -- starts with
# "internvl_grounding", does NOT contain "_grounding_v", NOT in the data_qwen.py
# whitelist. Also distinct from internvl_grounding_spatial_rel_v1 (the Knowledge-set
# spatial-relation QA style), which is a different task.
spatial_box_system_prompt = """You are a GUI grounding assistant operating on desktops, mobile devices, and web browsers. Given a screenshot and a spatial reference that locates one on-screen element relative to another, find the referred element and return its bounding box.

## Input Specification
- A screenshot of the current screen + a spatial reference wrapped in <ref>...</ref>. The reference names an anchor element by its visible text and states where the target sits relative to that anchor (for example: left of, right of, above, below, next to, just before, just after).

## Output Format
Return a JSON array holding the single referred element and its bounding box [x1, y1, x2, y2]:
```json
[{"bbox_2d": [x1, y1, x2, y2], "label": "<spatial reference>"}]
```

## Note
- The reference mentions two elements: the anchor and the target. Return the box of the TARGET, never the anchor.
- The reference may be phrased as an instruction ("click the item below X", "put your cursor left of Y"). Do not perform the action; only return the box of the element it points at.
- Resolve the direction in screen space: left/right along the x axis, above/below along the y axis. "Just before" means the immediately preceding element in reading order, "just after" the immediately following one.
- Return the nearest element satisfying the relation, not any element in that general direction.
- The box must tightly enclose the target element, not the container or row holding it.
- All bounding-box coordinates are integers in a 1000x1000 space (x from the left edge, y from the top edge).
- Output only the JSON array."""


# Element grounding with a POINT target, split by platform. Sibling of
# internvl_grounding_element_box_v1: same referring-expression input, same [0,1000] space,
# but the gpt target is <ref>..</ref><point>[[x, y]]</point> -> find_point -> point_2d JSON.
#
# Why these exist: point-grounding data that ships its target as a <tool_call> click is
# forced down the "_grounding_v" GUI-action path -- the model must emit a tool call and
# therefore learns an ACTION, not a localization. Converting those targets to
# <point> needs a conv_style whose contract
# is "return a point", which none of the existing styles provide: internvl_grounding_* are all
# box/quad, and internvl2_5_*_grounding_v1 are all tool_call.
#
# Naming constraints, identical to internvl_grounding_element_box_v1 and all load-bearing:
#   * MUST start with "internvl_grounding" so preprocess_conversation_format routes the gpt
#     target through format_grounding_internvl2qwenvl -> find_point.
#   * MUST NOT contain the substring "_grounding_v" -- "_grounding_point_" does not, but a
#     rename to e.g. "internvl_grounding_v2_point" WOULD, and would then take the GUI-action
#     path and assert <tool_call> on a <point> target (hard AssertionError on every sample).
#   * MUST NOT be added to the exact-match whitelist in preprocess_conversation_format, or
#     these system messages are discarded in favour of "You are a helpful assistant.".
#
# What differs per platform: only the surface vocabulary and the interaction verb the
# instructions actually use ("click" on desktop/web, "tap" on mobile). There is deliberately
# NO action space here -- the target is a coordinate, not a call, so build_grounding_space()
# has nothing to contribute. ubuntu and desktop share one surface text, mirroring how
# internvl2_5_ubuntu_grounding_v1 and internvl2_5_desktop_grounding_v1 both pass
# build_grounding_space("desktop").
element_point_system_prompt = """You are a GUI grounding assistant operating on {surface}. Given a screenshot and a referring expression that identifies one on-screen element, locate that element and return a single point inside it.

## Input Specification
- A screenshot of the current screen + a referring expression wrapped in <ref>...</ref>. The expression is either the element's visible text, a description of it, or its position relative to another element.

## Output Format
Return a JSON array holding the single referred element and one point [x, y] inside it:
```json
[{{"point_2d": [x, y], "label": "<referring expression>"}}]
```

## Note
- Return exactly one element: the one the referring expression names. Do not list other elements.
- The point must fall inside the element, as close to its centre as you can judge.
- If the expression names an anchor element and states where the target sits relative to it, return the point of the TARGET, never the anchor.
- The expression may be phrased as an instruction ("{verb} the Save button"). Do not perform the action; only return the point of the element it names.
- All coordinates are integers in a 1000x1000 space (x from the left edge, y from the top edge).
- Output only the JSON array."""


# Registered explicitly rather than in a loop over a platform table: a loop builds the
# names with an f-string, so `grep internvl_grounding_point_ubuntu_v1` finds only the
# data meta files that reference it and never the registration itself. The four bodies are
# identical apart from `surface`/`verb`, which is the same trade the rest of this file
# already makes (internvl2_5_{ubuntu,macos,desktop}_grounding_v1 are three near-copies).

register_conv_template(
    Conversation(
        name="internvl_grounding_point_desktop_v1",
        system_template="<|im_start|>system\n{system_message}",
        system_message=element_point_system_prompt.format(
            surface="desktop application windows", verb="click"
        ),
        roles=("<|im_start|>user\n", "<|im_start|>assistant\n"),
        sep_style=SeparatorStyle.MPT,
        sep="<|im_end|>\n",
    )
)

# Same surface text as desktop, mirroring internvl2_5_ubuntu_grounding_v1 and
# internvl2_5_desktop_grounding_v1 both passing build_grounding_space("desktop").
register_conv_template(
    Conversation(
        name="internvl_grounding_point_ubuntu_v1",
        system_template="<|im_start|>system\n{system_message}",
        system_message=element_point_system_prompt.format(
            surface="desktop application windows", verb="click"
        ),
        roles=("<|im_start|>user\n", "<|im_start|>assistant\n"),
        sep_style=SeparatorStyle.MPT,
        sep="<|im_end|>\n",
    )
)

register_conv_template(
    Conversation(
        name="internvl_grounding_point_web_v1",
        system_template="<|im_start|>system\n{system_message}",
        system_message=element_point_system_prompt.format(
            surface="web pages in a browser", verb="click"
        ),
        roles=("<|im_start|>user\n", "<|im_start|>assistant\n"),
        sep_style=SeparatorStyle.MPT,
        sep="<|im_end|>\n",
    )
)

register_conv_template(
    Conversation(
        name="internvl_grounding_point_mobile_v1",
        system_template="<|im_start|>system\n{system_message}",
        system_message=element_point_system_prompt.format(
            surface="mobile phone and tablet screens", verb="tap"
        ),
        roles=("<|im_start|>user\n", "<|im_start|>assistant\n"),
        sep_style=SeparatorStyle.MPT,
        sep="<|im_end|>\n",
    )
)


register_conv_template(
    Conversation(
        name="internvl_grounding_spatial_box_v1",
        system_template="<|im_start|>system\n{system_message}",
        system_message=spatial_box_system_prompt,
        roles=("<|im_start|>user\n", "<|im_start|>assistant\n"),
        sep_style=SeparatorStyle.MPT,
        sep="<|im_end|>\n",
    )
)


register_conv_template(
    Conversation(
        name="internvl_grounding_screen_ocr_v1",
        system_template="<|im_start|>system\n{system_message}",
        system_message=screen_ocr_system_prompt,
        roles=("<|im_start|>user\n", "<|im_start|>assistant\n"),
        sep_style=SeparatorStyle.MPT,
        sep="<|im_end|>\n",
    )
)


register_conv_template(
    Conversation(
        name="internvl_grounding_spatial_rel_v1",
        system_template="<|im_start|>system\n{system_message}",
        system_message=spatial_rel_system_prompt,
        roles=("<|im_start|>user\n", "<|im_start|>assistant\n"),
        sep_style=SeparatorStyle.MPT,
        sep="<|im_end|>\n",
    )
)


register_conv_template(
    Conversation(
        name="internvl_knowledge_existence_v1",
        system_template="<|im_start|>system\n{system_message}",
        system_message=existence_system_prompt,
        roles=("<|im_start|>user\n", "<|im_start|>assistant\n"),
        sep_style=SeparatorStyle.MPT,
        sep="<|im_end|>\n",
    )
)


register_conv_template(
    Conversation(
        name="internvl_knowledge_negatives_v1",
        system_template="<|im_start|>system\n{system_message}",
        system_message=negatives_system_prompt,
        roles=("<|im_start|>user\n", "<|im_start|>assistant\n"),
        sep_style=SeparatorStyle.MPT,
        sep="<|im_end|>\n",
    )
)


# General-purpose OCR (NOT screen-specific): read every text instance in a natural
# image and localize each with a QUADRILATERAL (4 corner points), so rotated/skewed
# text is tightly enclosed instead of by a loose axis-aligned box. Named with the
# "internvl_grounding" prefix so data_qwen.py routes the <ref>text</ref>
# <quad>[[...]]</quad> pairs in the gpt answer through find_quad ([0,1000] -> pixels);
# the "_ocr_quad_v1" suffix keeps it out of the <action>/"_grounding_v" GUI path.
general_ocr_quad_system_prompt = """You are a general-purpose OCR assistant for natural images. Read every piece of text in the image and localize each one with a quadrilateral of four corner points, so rotated or skewed text is tightly enclosed.

## Output Format
Return a JSON array; each item is the recognized text and its quadrilateral [[x1,y1],[x2,y2],[x3,y3],[x4,y4]] (the four corners clockwise from the top-left):
```json
[{"quad": [[x1,y1],[x2,y2],[x3,y3],[x4,y4]], "label": "<text>"}, ...]
```

## Note
- Cover every legible text instance; do not invent text that is not visible.
- All quadrilateral coordinates are integers in a 1000x1000 space (x from the left edge, y from the top edge).
- Output only the JSON array."""


register_conv_template(
    Conversation(
        name="internvl_grounding_ocr_quad_v1",
        system_template="<|im_start|>system\n{system_message}",
        system_message=general_ocr_quad_system_prompt,
        roles=("<|im_start|>user\n", "<|im_start|>assistant\n"),
        sep_style=SeparatorStyle.MPT,
        sep="<|im_end|>\n",
    )
)


GUI_CONV_TEMPLATE = [
    "internvl2_5_all_grounding_v1",
    "internvl2_5_desktop_grounding_v1",
    "internvl2_5_windows_grounding_v1",
    "internvl2_5_ubuntu_grounding_v1",
    "internvl2_5_macos_grounding_v1",
    "internvl2_5_iphone_grounding_v1",
    "internvl2_5_android_grounding_v1",
    "internvl2_5_mobile_grounding_v1",
    "internvl2_5_web_grounding_v1",
    # grounding that may abstain (build_grounding_refusal_space)
    "internvl2_5_ubuntu_refusal_grounding_v1",
    "internvl2_5_desktop_refusal_grounding_v1",
    "internvl2_5_web_refusal_grounding_v1",
    "internvl2_5_android_refusal_grounding_v1",
    "internvl2_5_mobile_refusal_grounding_v1",
    "internvl2_5_android_navigation_v1",
    "internvl2_5_mobile_navigation_v1",
    "internvl2_5_web_navigation_v1",
    "internvl2_5_android_planning_cot_v1",
    "internvl2_5_mobile_planning_cot_v1",
    "internvl2_5_web_planning_cot_v1",
    "internvl2_5_mind2web_navigation_v1",
    "internvl2_5_mind2web_planning_cot_v1",
    "internvl_grounding",
    "internvl_referring",
    "internvl2_5_mobile_odyssey_plus",
    "internvl2_5_web_os_genesis",
    "internvl2_5_mobile_os_genesis",
    "internvl2_5_desktop_navigation_v1",
    "internvl2_5_desktop_planning_cot_v1",
    "internvl2_5_mac_planning_cot_v1",
    "internvl2_5_mac_navigation_v1",
    "internvl2_5_windows_planning_cot_v1",
    "internvl2_5_windows_navigation_v1",
    "internvl2_5_ubuntu_planning_cot_v1",
    "internvl2_5_ubuntu_navigation_v1",
    "internvl2_5_desktop_verification_v1",
    "internvl2_5_desktop_reflective_planning_cot_v1",
    "internvl2_5_desktop_forward_dynamics_v1",
    "internvl2_5_desktop_inverse_dynamics_v1",
    "internvl2_5_mobile_forward_dynamics_v1",
    "internvl2_5_mobile_inverse_dynamics_v1",
    "internvl2_5_desktop_task_planning_v1",
    "internvl2_5_mobile_task_planning_v1",
    "internvl2_5_web_forward_dynamics_v1",
    "internvl2_5_web_inverse_dynamics_v1",
    "internvl2_5_web_task_planning_v1",
    "internvl2_5_desktop_forward_dynamics_v2",
    "internvl2_5_web_forward_dynamics_v2",
    "internvl2_5_mobile_forward_dynamics_v2",
    "internvl2_5_android_planning_cot_v2",
    "internvl2_5_mobile_planning_cot_v2",
    "internvl2_5_web_planning_cot_v2",
    "internvl2_5_ubuntu_planning_cot_v2",
    "internvl2_5_windows_planning_cot_v2",
    "internvl2_5_mac_planning_cot_v2",
    "internvl2_5_desktop_planning_cot_v2",
    # Element/spatial grounding with a BOX target (registered above). Listed so the
    # internvl_chat trainer picks preprocess_gui_conv instead of falling through to the
    # generic preprocess. NOTE: this list is only read by
    # the upstream InternVL-chat trainer (internvl_chat_pretrain.py) -- the qwen-vl-finetune
    # trainer these
    # styles are actually trained with (qwenvl/data/data_qwen.py) never imports it and
    # dispatches on the conv_style string itself, so membership here changes nothing there.
    "internvl_grounding_element_box_v1",
    "internvl_grounding_spatial_box_v1",
    # Element grounding with a POINT target, per platform (registered above). Same
    # rationale as the two box styles for being listed here.
    "internvl_grounding_point_desktop_v1",
    "internvl_grounding_point_ubuntu_v1",
    "internvl_grounding_point_web_v1",
    "internvl_grounding_point_mobile_v1",
    # The rest of the same family, registered above.
    # Same reason: without them the internvl_chat trainer silently falls through to the
    # generic preprocess. internvl_referring_understanding_v1 / internvl_knowledge_* /
    # internvl_grounding_spatial_rel_v1 are in active use by the Understanding and
    # Knowledge sets; ocr_quad / screen_ocr are listed for completeness.
    "internvl_grounding_spatial_rel_v1",
    "internvl_grounding_screen_ocr_v1",
    "internvl_grounding_ocr_quad_v1",
    "internvl_referring_understanding_v1",
    "internvl_knowledge_existence_v1",
    "internvl_knowledge_negatives_v1",
]

if __name__ == "__main__":
    platform = ["Windows", "Ubuntu", "macOS", "Browser", "Android", "iOS", "iPadOS"]
    print(conv_templates["internvl2_5_ubuntu_planning_cot_v1"].system_message)
