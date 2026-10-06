import transformers
from dataclasses import dataclass, field
from typing import Dict, Optional, Sequence, List


@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(default="Qwen/Qwen2.5-VL-3B-Instruct")
    tune_mm_llm: bool = field(default=False)
    tune_mm_mlp: bool = field(default=False)
    tune_mm_vision: bool = field(default=False)


@dataclass
class DataArguments:
    data_path: str = field(default="")
    meta_path: str = field(default="")
    conv_style: Optional[int] = field(default=None)
    video_max_frames: Optional[int] = field(default=8)
    video_min_frames: Optional[int] = field(default=4)
    data_flatten: bool = field(default=False)
    base_interval: int = field(default=2)
    coord_norm: bool = field(default=True)
    group_sampling: bool = field(default=False)
    custom_seed: int = field(default=42)
    training_mode: str = field(
        default="sft",
        metadata={
            "help": "Label masking policy. 'sft' supervises only assistant text "
                    "(default, instruction tuning). 'cpt' supervises every text "
                    "token of the rendered chat string for continued pretraining."
        },
    )
    system_loss_ratio: Optional[float] = field(
        default=None,
        metadata={
            "help": "Per-sample probability of computing loss on the system "
                    "header (the ~constant action-space boilerplate). None uses "
                    "the per-mode default (sft: 0.0 = system masked; cpt: 1.0 = "
                    "system supervised). Set a small value (e.g. 0.1-0.2) to let "
                    "the model internalize the action-space spec without that "
                    "fixed string dominating the loss on every step."
        },
    )
    # Qwen3-VL-8B native image-processor pixel budgets, taken verbatim from the
    # model's preprocessor_config.json (size.shortest_edge / size.longest_edge).
    # Qwen3-VL uses patch_size=16, merge_size=2 (factor 32).
    min_pixels: int = field(default=65536)     # size.shortest_edge
    max_pixels: int = field(default=16777216)  # size.longest_edge
    # Equivalent Qwen2.5-VL (factor 28) values, for reference:
    #   min_pixels=3136 (4*28*28); max_pixels=2109744 (1080p, 69*39*28*28)
    video_max_frame_pixels: int = field(default=32 * 28 * 28)
    video_min_frame_pixels: int = field(default=4 * 28 * 28)


@dataclass
class TrainingArguments(transformers.TrainingArguments):
    cache_dir: Optional[str] = field(default=None)
    optim: str = field(default="adamw_torch")
    model_max_length: int = field(
        default=512,
        metadata={
            "help": "Maximum sequence length. Sequences will be right padded (and possibly truncated)."
        },
    )
    mm_projector_lr: Optional[float] = None
    vision_tower_lr: Optional[float] = None
