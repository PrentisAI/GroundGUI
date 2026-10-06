import os
from re import I
from typing import (
    TYPE_CHECKING,
    Sequence,
    Any,
    Callable,
    Dict,
    List,
    Optional,
    Tuple,
    Type,
    Union,
)

import datasets
import torch
import torch.nn as nn
import torch.distributed as dist
from torch.utils.data import Sampler, BatchSampler, Dataset, DataLoader
import random
import math
from copy import deepcopy
from flash_attn.flash_attn_interface import flash_attn_varlen_func
from torch.utils.data import DataLoader, Sampler
from transformers import Trainer
from transformers.cache_utils import Cache
from transformers.models.qwen2_5_vl.modeling_qwen2_5_vl import (
    Qwen2_5_VisionTransformerPretrainedModel,
    Qwen2_5_VLModel,
)
from transformers.models.qwen2_vl.modeling_qwen2_vl import (
    Qwen2VisionTransformerPretrainedModel,
    Qwen2VLModel,
)
from transformers.trainer import (
    get_parameter_names,
    has_length,
    is_sagemaker_mp_enabled,
)
try:
    from transformers.trainer import ALL_LAYERNORM_LAYERS
except ImportError:
    from transformers.pytorch_utils import ALL_LAYERNORM_LAYERS
from transformers.trainer_utils import seed_worker


def rank0_print(*args):
    if dist.is_initialized():
        if dist.get_rank() == 0:
            print(f"Rank {dist.get_rank()}: ", *args)
    else:
        print(*args)


def _flash_attention_forward(
    query_states: torch.Tensor,
    key_states: torch.Tensor,
    value_states: torch.Tensor,
    attention_mask: torch.Tensor,
    query_length: int,
    is_causal: bool,
    dropout: float = 0.0,
    position_ids: Optional[torch.Tensor] = None,
    softmax_scale: Optional[float] = None,
    sliding_window: Optional[int] = None,
    use_top_left_mask: bool = False,
    softcap: Optional[float] = None,
    deterministic: bool = None,
    cu_seq_lens_q: Optional[torch.LongTensor] = None,
    cu_seq_lens_k: Optional[torch.LongTensor] = None,
    max_length_q: Optional[int] = None,
    max_length_k: Optional[int] = None,
    target_dtype: Optional[torch.dtype] = None,
    **kwargs,
):
    """
    Calls the forward method of Flash Attention - if the input hidden states contain at least one padding token
    first unpad the input, then computes the attention scores and pad the final attention scores.

    Args:
        query_states (`torch.Tensor`):
            Input query states to be passed to Flash Attention API
        key_states (`torch.Tensor`):
            Input key states to be passed to Flash Attention API
        value_states (`torch.Tensor`):
            Input value states to be passed to Flash Attention API
        attention_mask (`torch.Tensor`):
            The padding mask - corresponds to a tensor of size `(batch_size, seq_len)` where 0 stands for the
            position of padding tokens and 1 for the position of non-padding tokens.
        dropout (`float`):
            Attention dropout
        softmax_scale (`float`, *optional*):
            The scaling of QK^T before applying softmax. Default to 1 / sqrt(head_dim)
        use_top_left_mask (`bool`, defaults to `False`):
            flash_attn<2.1 generates top-left aligned causal mask, while what is needed here is bottom-right alignement, that was made default for flash_attn>=2.1. This attribute is used to handle this difference.
        softcap (`float`, *optional*):
            Softcap for the attention logits, used e.g. in gemma2.
        deterministic (`bool`, *optional*):
            Determines if the deterministic option introduced in flash_attn>=2.4.1 is enabled.
    """
    assert query_states.size(0) == key_states.size(0) == value_states.size(0) == 1
    query_states = query_states.squeeze(0)
    key_states = key_states.squeeze(0)
    value_states = value_states.squeeze(0)
    cu_seqlens = attention_mask

    with torch.no_grad():
        max_seqlen = max(
            [
                cu_seqlens[idx + 1] - cu_seqlens[idx]
                for idx in range(cu_seqlens.size(0) - 1)
            ]
        ).item()

    if not use_top_left_mask:
        causal = is_causal
    else:
        # TODO: Remove the `query_length != 1` check once Flash Attention for RoCm is bumped to 2.1.
        causal = is_causal and query_length != 1

    # Assuming 4D tensors, key_states.shape[1] is the key/value sequence length (source length).
    flash_kwargs = {}

    if softcap is not None:
        flash_kwargs["softcap"] = softcap

    attn_output = flash_attn_varlen_func(
        query_states,
        key_states,
        value_states,
        cu_seqlens_q=cu_seqlens,
        cu_seqlens_k=cu_seqlens,
        max_seqlen_q=max_seqlen,
        max_seqlen_k=max_seqlen,
        dropout_p=dropout,
        softmax_scale=softmax_scale,
        causal=causal,
        **flash_kwargs,
    )

    attn_output = attn_output.unsqueeze(0)
    query_states = query_states.unsqueeze(0)
    key_states = key_states.unsqueeze(0)
    value_states = value_states.unsqueeze(0)

    return attn_output


def _update_causal_mask(
    self,
    attention_mask: torch.Tensor,
    input_tensor: torch.Tensor,
    cache_position: torch.Tensor,
    past_key_values: Cache,
    output_attentions: bool,
):
    return attention_mask


def replace_qwen2_vl_attention_class():
    import transformers
    import transformers.modeling_flash_attention_utils

    transformers.models.qwen2_vl.modeling_qwen2_vl._flash_attention_forward = (
        _flash_attention_forward
    )
    transformers.models.qwen2_vl.modeling_qwen2_vl.Qwen2VLModel._update_causal_mask = (
        _update_causal_mask
    )
    transformers.models.qwen2_5_vl.modeling_qwen2_5_vl._flash_attention_forward = (
        _flash_attention_forward
    )
    transformers.models.qwen2_5_vl.modeling_qwen2_5_vl.Qwen2_5_VLModel._update_causal_mask = (
        _update_causal_mask
    )


def print_trainable_parameters_visual(self) -> None:
    """
    Prints the trainable status of all vision components including attention blocks and merger module.
    Outputs the indices of trainable/non-trainable blocks and the merger module status.
    """
    trainable_blocks = []
    non_trainable_blocks = []

    # Check trainable status of vision attention blocks
    for block_idx, block in enumerate(self.blocks):
        is_trainable = all(param.requires_grad for param in block.parameters())
        if is_trainable:
            trainable_blocks.append(block_idx)
        else:
            non_trainable_blocks.append(block_idx)

    # Check trainable status of merger module
    is_merger_trainable = any(param.requires_grad for param in self.merger.parameters())

    # Print results
    print("Vision Module - Attention Blocks:")
    print(
        f"Trainable Block Indices: {trainable_blocks if trainable_blocks else 'None'}"
    )
    print(
        f"Non-Trainable Block Indices: {non_trainable_blocks if non_trainable_blocks else 'None'}"
    )
    print(f"Merger Module Trainable: {is_merger_trainable}")


def print_trainable_parameters(self) -> None:
    """
    Prints the trainable status of all LLM components including embeddings, layers, and normalization.
    Outputs the indices of trainable/non-trainable layers and other module statuses.
    """
    # transformers >=4.50 wraps the text submodules under .language_model
    text_module = getattr(self, "language_model", self)
    # Check embed_tokens
    is_embed_trainable = any(
        param.requires_grad for param in text_module.embed_tokens.parameters()
    )
    print(f"LLM Module - Embed Tokens Trainable: {is_embed_trainable}")

    # Check each decoder layer
    trainable_layers = []
    non_trainable_layers = []

    for layer_idx, layer in enumerate(text_module.layers):
        is_trainable = any(param.requires_grad for param in layer.parameters())
        if is_trainable:
            trainable_layers.append(layer_idx)
        else:
            non_trainable_layers.append(layer_idx)

    # Print layer status
    print(
        f"LLM Module - Trainable Layer Indices: {trainable_layers if trainable_layers else 'None'}"
    )
    print(
        f"LLM Module - Non-Trainable Layer Indices: {non_trainable_layers if non_trainable_layers else 'None'}"
    )


def create_optimizer(self):

    opt_model = self.model

    if self.optimizer is None:
        decay_parameters = get_parameter_names(opt_model, ALL_LAYERNORM_LAYERS)
        decay_parameters = [name for name in decay_parameters if "bias" not in name]
        if self.args.mm_projector_lr is not None and self.args.mm_projector_lr != 0:
            projector_parameters = [
                name for name, _ in opt_model.named_parameters() if "merger" in name
            ]
            if self.args.vision_tower_lr is not None and self.args.vision_tower_lr != 0:
                vision_tower_parameters = [
                    name for name, _ in opt_model.named_parameters() if "visual" in name
                ]
                optimizer_grouped_parameters = [
                    {
                        "params": [
                            p
                            for n, p in opt_model.named_parameters()
                            if (
                                n in decay_parameters
                                and n not in projector_parameters
                                and n not in vision_tower_parameters
                                and p.requires_grad
                            )
                        ],
                        "weight_decay": self.args.weight_decay,
                    },
                    {
                        "params": [
                            p
                            for n, p in opt_model.named_parameters()
                            if (
                                n in decay_parameters
                                and n not in projector_parameters
                                and n in vision_tower_parameters
                                and p.requires_grad
                            )
                        ],
                        "weight_decay": self.args.weight_decay,
                        "lr": self.args.vision_tower_lr,
                    },
                    {
                        "params": [
                            p
                            for n, p in opt_model.named_parameters()
                            if (
                                n not in decay_parameters
                                and n not in projector_parameters
                                and n not in vision_tower_parameters
                                and p.requires_grad
                            )
                        ],
                        "weight_decay": 0.0,
                    },
                    {
                        "params": [
                            p
                            for n, p in opt_model.named_parameters()
                            if (
                                n not in decay_parameters
                                and n not in projector_parameters
                                and n in vision_tower_parameters
                                and p.requires_grad
                            )
                        ],
                        "weight_decay": 0.0,
                        "lr": self.args.vision_tower_lr,
                    },
                    {
                        "params": [
                            p
                            for n, p in opt_model.named_parameters()
                            if (
                                n in decay_parameters
                                and n in projector_parameters
                                and p.requires_grad
                            )
                        ],
                        "weight_decay": self.args.weight_decay,
                        "lr": self.args.mm_projector_lr,
                    },
                    {
                        "params": [
                            p
                            for n, p in opt_model.named_parameters()
                            if (
                                n not in decay_parameters
                                and n in projector_parameters
                                and p.requires_grad
                            )
                        ],
                        "weight_decay": 0.0,
                        "lr": self.args.mm_projector_lr,
                    },
                ]
            else:
                optimizer_grouped_parameters = [
                    {
                        "params": [
                            p
                            for n, p in opt_model.named_parameters()
                            if (
                                n in decay_parameters
                                and n not in projector_parameters
                                and p.requires_grad
                            )
                        ],
                        "weight_decay": self.args.weight_decay,
                    },
                    {
                        "params": [
                            p
                            for n, p in opt_model.named_parameters()
                            if (
                                n not in decay_parameters
                                and n not in projector_parameters
                                and p.requires_grad
                            )
                        ],
                        "weight_decay": 0.0,
                    },
                    {
                        "params": [
                            p
                            for n, p in opt_model.named_parameters()
                            if (
                                n in decay_parameters
                                and n in projector_parameters
                                and p.requires_grad
                            )
                        ],
                        "weight_decay": self.args.weight_decay,
                        "lr": self.args.mm_projector_lr,
                    },
                    {
                        "params": [
                            p
                            for n, p in opt_model.named_parameters()
                            if (
                                n not in decay_parameters
                                and n in projector_parameters
                                and p.requires_grad
                            )
                        ],
                        "weight_decay": 0.0,
                        "lr": self.args.mm_projector_lr,
                    },
                ]
        else:
            optimizer_grouped_parameters = [
                {
                    "params": [
                        p
                        for n, p in opt_model.named_parameters()
                        if (n in decay_parameters and p.requires_grad)
                    ],
                    "weight_decay": self.args.weight_decay,
                },
                {
                    "params": [
                        p
                        for n, p in opt_model.named_parameters()
                        if (n not in decay_parameters and p.requires_grad)
                    ],
                    "weight_decay": 0.0,
                },
            ]

        optimizer_cls, optimizer_kwargs = Trainer.get_optimizer_cls_and_kwargs(
            self.args
        )
        self.optimizer = optimizer_cls(optimizer_grouped_parameters, **optimizer_kwargs)

    return self.optimizer


class GroupedSampler(Sampler):
    def __init__(
        self, dataset: Dataset, batch_size: int, shuffle: bool = True, seed: int = 10086
    ):
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.seed = seed
        self.rng = random.Random(self.seed)

        try:
            if dist.is_initialized():
                self.rank = dist.get_rank()
                self.world_size = dist.get_world_size()
            else:
                self.rank, self.world_size = 0, 1
        except ImportError:
            self.rank, self.world_size = 0, 1

        total_batch_size = self.world_size * self.batch_size
        self.group_true = dataset.mm_samples
        self.group_true = self.group_true[
            : (len(self.group_true) // total_batch_size) * total_batch_size
        ]
        self.group_false = dataset.text_samples
        self.group_false = self.group_false[
            : (len(self.group_false) // total_batch_size) * total_batch_size
        ]

        rank0_print(
            f"Length of multimodal samples: {len(self.group_true)}, "
            f"pure textual samples: {len(self.group_false)}"
        )

    def make_data(self):
        def _shard(groups, cnt):
            if self.shuffle:
                self.rng.shuffle(groups)
            return [groups[i] for i in range(self.rank, len(groups), self.world_size)]

        shard_true = _shard(deepcopy(self.group_true), 0)
        shard_false = _shard(deepcopy(self.group_false), 1)
        return shard_true, shard_false

    def __iter__(self):
        # Radomly shuffle the indices if required
        shard_true, shard_false = self.make_data()
        idx_true, idx_false = 0, 0
        ratio_true = max(
            len(shard_true) / (len(shard_true) + len(shard_false)) - 0.1, 0.5
        )
        p = 0
        for idx in range(len(shard_true) + len(shard_false)):
            if idx % self.batch_size == 0:
                p = self.rng.random()
            if idx_false < len(shard_false) and idx_true < len(shard_true):
                if p > ratio_true:
                    item = shard_false[idx_false]
                    idx_false += 1
                else:
                    item = shard_true[idx_true]
                    idx_true += 1
            else:
                if idx_false < len(shard_false):
                    item = shard_false[idx_false]
                    idx_false += 1
                elif idx_true < len(shard_true):
                    item = shard_true[idx_true]
                    idx_true += 1

            yield item

    def __len__(self):
        total = (len(self.group_true) + len(self.group_false)) // self.world_size
        return total


def get_train_dataloader(self):
    total_train_batch_size = (
        self._train_batch_size
        * self.args.gradient_accumulation_steps
        * self.args.world_size
    )
    sampler = GroupedSampler(self.train_dataset, self._train_batch_size, shuffle=True)

    return DataLoader(
        self.train_dataset,
        batch_size=self._train_batch_size,
        sampler=sampler,
        # batch_sampler=sampler,
        collate_fn=self.data_collator,
        drop_last=self.args.dataloader_drop_last,
        num_workers=self.args.dataloader_num_workers,
        pin_memory=self.args.dataloader_pin_memory,
    )


"""
class CustomTrainer(Trainer):
    # def save_model(self, output_dir: Optional[str] = None, _internal_call: bool = False):
    #     processing_class = self.processing_class
    #     self.processing_class = None
    #     super().save_model(output_dir, _internal_call)
    #     self.processing_class = processing_class
    #     # save tokenizer and image_processor
    #     tokenizer = self.processing_class.tokenizer
    #     self.processing_class.tokenizer = None
    #     tokenizer.save_pretrained(output_dir)
    #     self.processing_class.save_pretrained(output_dir)
    #     self.processing_class.tokenizer = tokenizer
"""


# Apply monkey patches
# Trainer.create_optimizer = create_optimizer
Trainer.create_optimizer = create_optimizer
# Trainer.get_train_dataloader = get_train_dataloader

Qwen2VisionTransformerPretrainedModel.print_trainable_parameters = (
    print_trainable_parameters_visual
)
Qwen2VLModel.print_trainable_parameters = print_trainable_parameters
Qwen2_5_VisionTransformerPretrainedModel.print_trainable_parameters = (
    print_trainable_parameters_visual
)
Qwen2_5_VLModel.print_trainable_parameters = print_trainable_parameters

# Qwen3-VL (transformers >= 4.57): attach the same trainable-param reporters to
# the new module classes. print_trainable_parameters_visual reads self.blocks /
# self.merger (present on Qwen3VLVisionModel); print_trainable_parameters reads
# self.layers / self.embed_tokens (present on Qwen3VLTextModel). Guarded so the
# file still imports on transformers versions without Qwen3-VL.
try:
    from transformers.models.qwen3_vl.modeling_qwen3_vl import (
        Qwen3VLVisionModel,
        Qwen3VLTextModel,
    )

    Qwen3VLVisionModel.print_trainable_parameters = print_trainable_parameters_visual
    Qwen3VLTextModel.print_trainable_parameters = print_trainable_parameters
except Exception as _e:  # noqa: BLE001
    pass


# ---------------------------------------------------------------------------
# Diagnosis and skipping of large-gradient steps. Both switches are off by default; with
# neither set, no patch is installed and the training path is unchanged.
#
#   GRAD_NORM_PROBE=1        print only, no behaviour change. Each optimizer step logs
#                            grad_norm, sequence length, per-rank token counts and their
#                            imbalance, to locate where spikes come from.
#   GRAD_NORM_SKIP_MULT=5    skip the whole step when grad_norm > 5x (median of the last
#                            200 steps).
#   GRAD_NORM_SKIP_WARMUP=50 number of steps of history to collect before skipping
#                            (default 50).
#
# Why skip instead of tuning --max_grad_norm further: clipping only changes the gradient
# magnitude, and Adam is invariant to a global rescaling of the gradient, so when every step
# gets clipped (e.g. a threshold of 1 or 10 here) clipping does not distinguish spikes from
# normal steps at all. Skipping changes *which* steps contribute to the update, which Adam
# cannot normalise away, so it is a separate, genuinely effective lever. The first choice is
# still to move --max_grad_norm inside the grad_norm distribution (see
# scripts/sft_local.sh), which loses no data; skipping is the next step if that is not
# enough.
#
# Diagnose before skipping: --group_sampling True makes batches non-i.i.d. (samples of
# similar length are grouped together), so spikes likely correspond to specific data groups
# rather than random noise; blind skipping would systematically drop the hardest data.
#
# Implementation reuses DeepSpeed's own skip exit: when
# `_overflow_check_and_loss_scale_update` returns True, stage3.step() takes the fp16-overflow
# branch -- `_overflow_clean_up` clears the gradients and discards averaged_gradients, then
# returns without calling optimizer.step(). That path is well tested and much safer than a
# hand-rolled skip. Note that "zero the gradients and step as usual" is NOT equivalent: with
# zero gradients Adam still updates parameters through momentum, and the bias-correction t is
# incremented, so that is not a skipped step.
#
# Cost: on non-skipped steps _get_norm_groups() is computed one extra time (one all-reduce,
# tens of milliseconds; negligible relative to ~18 s/it).
# ---------------------------------------------------------------------------
_GN_PROBE = os.environ.get("GRAD_NORM_PROBE") == "1"
_GN_SKIP_MULT = float(os.environ.get("GRAD_NORM_SKIP_MULT", "0") or 0)
_GN_SKIP_WARMUP = int(os.environ.get("GRAD_NORM_SKIP_WARMUP", "50") or 50)

if _GN_PROBE or _GN_SKIP_MULT > 0:
    try:
        from deepspeed.runtime.zero.stage3 import DeepSpeedZeroOptimizer_Stage3
        from transformers import Trainer as _HFTrainer

        import collections as _gn_collections

        _gn_hist: List[float] = []
        _gn_batch: Dict[str, Any] = {}
        _gn_step = [0]

        if _GN_PROBE:
            _gn_orig_training_step = _HFTrainer.training_step

            def _gn_training_step(self, model, inputs, *args, **kwargs):
                # With gradient_accumulation_steps>1 this runs several times per optimizer
                # step, so accumulate rather than overwrite; the optimizer patch clears it
                # after printing. Looking only at the last micro-batch would miss samples and
                # break spike attribution.
                if isinstance(inputs, dict):
                    # Must pop: inputs is later expanded as model(**inputs), and an extra key raises.
                    srcs = inputs.pop("_gn_src", None)
                    ids = inputs.get("input_ids")
                    if ids is not None:
                        _gn_batch["tokens"] = _gn_batch.get("tokens", 0) + int(ids.numel())
                        _gn_batch["maxlen"] = max(
                            _gn_batch.get("maxlen", 0), int(ids.shape[-1])
                        )
                    if srcs:
                        _gn_batch.setdefault("srcs", []).extend(
                            "text"
                            if s is None
                            else "/".join(str(s).rstrip("/").split("/")[-2:])
                            for s in srcs
                        )
                return _gn_orig_training_step(self, model, inputs, *args, **kwargs)

            _HFTrainer.training_step = _gn_training_step

        _gn_orig_ovf = DeepSpeedZeroOptimizer_Stage3._overflow_check_and_loss_scale_update

        def _gn_overflow_check(self):
            # Run the original fp16-overflow logic first; on a real overflow keep its behaviour.
            if _gn_orig_ovf(self):
                return True

            # averaged_gradients is ready at this point; the norm is an all-reduced global
            # value, identical on every rank, so the skip decision below is the same on all
            # ranks and cannot desynchronise collectives.
            gnorm = float(
                torch.linalg.vector_norm(torch.stack(self._get_norm_groups()))
                / self.loss_scale
            )
            self._global_grad_norm = gnorm

            if _GN_PROBE:
                _gn_step[0] += 1
                world = dist.get_world_size()
                tok = torch.tensor(
                    [_gn_batch.get("tokens", 0)],
                    device=torch.cuda.current_device(),
                    dtype=torch.long,
                )
                gathered = [torch.zeros_like(tok) for _ in range(world)]
                dist.all_gather(gathered, tok)
                # Per-rank data sources need all_gather_object (strings, not tensors). This
                # sync cost is only paid when the probe is on.
                src_buf: List[Any] = [None] * world
                dist.all_gather_object(src_buf, _gn_batch.get("srcs", []))
                if dist.get_rank() == 0:
                    per_rank = [int(x.item()) for x in gathered]
                    lo = max(1, min(per_rank))
                    flat = [s for lst in src_buf if lst for s in lst]
                    # No top-N truncation: source-enrichment analysis needs full counts, and
                    # truncating could hide the real culprit in the long tail. Under
                    # group_sampling a single step has few sources anyway.
                    top = _gn_collections.Counter(flat).most_common()
                    src_str = ",".join(f"{k}:{v}" for k, v in top) or "-"
                    # Single key=value line, easy to grep and parse later for correlation analysis.
                    print(
                        f"[gn-probe] step={_gn_step[0]} grad_norm={gnorm:.2f} "
                        f"maxlen={_gn_batch.get('maxlen', -1)} "
                        f"tokens={sum(per_rank)} spread={max(per_rank) / lo:.3f} "
                        f"nsample={len(flat)} src={src_str}",
                        flush=True,
                    )
                # Clear on every rank, otherwise non-zero ranks keep accumulating.
                _gn_batch.clear()

            if _GN_SKIP_MULT > 0 and len(_gn_hist) >= _GN_SKIP_WARMUP:
                med = sorted(_gn_hist)[len(_gn_hist) // 2]
                if gnorm > _GN_SKIP_MULT * med:
                    # Reuse DeepSpeed's overflow clean-up: zero_grad(set_to_none=True) +
                    # discard averaged_gradients; step() then returns immediately.
                    self._overflow_clean_up(self.loss_scale)
                    if dist.get_rank() == 0:
                        print(
                            f"[gn-skip] grad_norm={gnorm:.2f} > "
                            f"{_GN_SKIP_MULT}x median({med:.2f}) -> step skipped",
                            flush=True,
                        )
                    return True

            # Only non-skipped steps enter the history; otherwise spikes inflate the median and
            # skipping becomes progressively harder to trigger.
            _gn_hist.append(gnorm)
            if len(_gn_hist) > 200:
                del _gn_hist[0]
            return False

        DeepSpeedZeroOptimizer_Stage3._overflow_check_and_loss_scale_update = (
            _gn_overflow_check
        )
        print(
            f"[grad-norm] patch installed: probe={_GN_PROBE} "
            f"skip_mult={_GN_SKIP_MULT} warmup={_GN_SKIP_WARMUP}",
            flush=True,
        )
    except Exception as _gn_e:  # noqa: BLE001
        # Fail loudly if the patch cannot be installed; otherwise one would believe skipping is
        # active when it is not.
        print(f"[grad-norm] FAILED to install patch: {_gn_e!r}", flush=True)
        raise
