# Copyright (c) ModelScope Contributors. All rights reserved.
import math
import os
import platform
from dataclasses import dataclass, field
from typing import List, Literal, Optional, Union

from transformers.training_args import TrainingArguments as HfTrainingArguments
from transformers.training_args_seq2seq import Seq2SeqTrainingArguments as HfSeq2SeqTrainingArguments

from swift.loss import loss_map
from swift.utils import get_dist_setting, get_logger, is_liger_available, is_mp, json_parse_to_dict

logger = get_logger()


@dataclass
class TrainArgumentsMixin:
    """A dataclass mixin for configuring model training parameters.

    Args:
        per_device_train_batch_size (int): The batch size per GPU/TPU core for training. Defaults to 1.
        per_device_eval_batch_size (int): The batch size per GPU/TPU core for evaluation. Defaults to 1.
        gradient_accumulation_steps (Optional[int]): The number of update steps to accumulate gradients for before
            performing an optimizer step.
        tuner_backend (Optional[str]): The backend to use for parameter-efficient fine-tuning (e.g., 'peft'). Defaults
            to None.
        gradient_checkpointing (bool): If True, use gradient checkpointing to save memory at the cost of a slower
            backward pass. Defaults to True.
        vit_gradient_checkpointing (Optional[bool]): A specific gradient checkpointing setting for the Vision
            Transformer part of the model. Defaults to None.
        gradient_checkpointing_kwargs (Optional[Union[dict, str]]): Keyword arguments for
            `torch.utils.checkpoint.checkpoint`. Defaults to None.
        logging_first_step (bool): Whether to log the first global step. Defaults to True.
        logging_steps (int): Log every `logging_steps` global steps. Defaults to 5.
        router_aux_loss_coef (float): The coefficient for the router auxiliary loss in Mixture-of-Experts models.
            Defaults to 0.0.
        enable_dft_loss (bool): Whether to enable Diversity-from-Diversity (DFD) loss.
            See https://arxiv.org/abs/2508.05629. Defaults to False.
        enable_channel_loss (bool): Whether to enable channel loss. Defaults to False.
        weight_decay (float): The weight decay to apply (if not zero) to all layers except bias and LayerNorm weights.
            Defaults to 0.1.
        adam_beta2 (float): The beta2 hyperparameter for the AdamW optimizer. Defaults to 0.95.
        lr_scheduler_type (str): The learning rate scheduler type to use. Defaults to 'cosine'.
        lr_scheduler_kwargs (Optional[Union[dict, str]]): Additional keyword arguments for the learning rate scheduler,
            passed as a JSON string or a dictionary. Defaults to None.
        report_to (List[str]): The list of integrations to report results to (e.g., 'tensorboard', 'wandb'). Defaults
            to ['tensorboard']. If you specify `--report_to wandb`, you can set the project name through `WANDB_PROJECT`
            and specify the API KEY corresponding to your account through `WANDB_API_KEY`.
        dataloader_num_workers (Optional[int]): The number of subprocesses to use for data loading. Defaults to None.
        dataloader_persistent_workers (bool): If True, the data loader workers will not be shut down after a dataset
            has been consumed once. Defaults to False.
        dataloader_prefetch_factor (Optional[int]): The number of batches loaded in advance by each worker. Defaults
            to None.
        use_liger_kernel (bool): Whether to use the Liger kernel for optimization. Defaults to False.
        check_model (bool): If True, checks local model files for corruption or modification and provides a warning.
            Should be set to False in an offline environment. Defaults to True.
        acc_strategy (Literal['token', 'seq']): The strategy for calculating accuracy during training and validation.
            Can be 'token' for token-level accuracy or 'seq' for sequence-level accuracy. Defaults to 'token'.
        train_dataloader_shuffle (bool): Whether to shuffle the training data. Defaults to True.
        max_epochs (Optional[int]): The total number of training epochs to perform. Overrides `num_train_epochs`.
            Defaults to None.
        aligner_lr (Optional[float]): A specific learning rate for the aligner part of the model. Defaults to None.
        vit_lr (Optional[float]): A specific learning rate for the Vision Transformer part of the model. Defaults to
            None.
        use_logits_to_keep (Optional[bool]): If enabled, reduces VRAM usage and speeds up training by calculating and
            storing only the necessary logits based on the labels during the forward pass. If None, the behavior is
            automatically determined. Defaults to None.
        ds3_gather_for_generation (bool): In DeepSpeed ZeRO-3, whether to gather model parameters for generation.
            Defaults to True.
        resume_only_model (bool): When resuming from a checkpoint, whether to load only the model weights and not the
            optimizer/scheduler states. Defaults to False.

        optimizer (Optional[str]):The optimizer plugin to use (takes priority over `--optim`), default is None.
            Available optimizers can be found in `optimizers/mapping.py`
        loss_type (Optional[str]): Custom loss_type name. Default is None, uses the model's built-in loss function.
            Available loss options can be found in `loss/mapping.py`
        metric (Optional[str]): Custom eval metric name. Default is None. Available eval_metric options can be found
            in `eval_metric/mapping.py`.
        callbacks (List[str]): Custom trainer callbacks, default is `[]`. Available callbacks can be found
            in `callbacks/mapping.py`.
        early_stop_interval (Optional[int]): The interval for early stopping. Training will be terminated if the
            `best_metric` does not improve for `early_stop_interval` evaluation periods (based on `save_steps`). It is
            recommended to set `eval_steps` and `save_steps` to the same value. The implementation can be found in the
            callback plugin. For more complex requirements, you can directly override the implementation in
            `callback.py`. Defaults to None.

        eval_use_evalscope (bool): Whether to use EvalScope for evaluation during training. Must be set to `True` to
            enable it. Refer to examples for usage details. Defaults to False.
        eval_dataset (List[str]): A list of evaluation dataset names. Multiple datasets can be specified, separated
            by spaces.
        eval_dataset_args (Optional[Union[str, dict]]): Arguments for the evaluation dataset(s), provided as a JSON
            string or a dictionary.
        eval_limit (Optional[int]): The maximum number of samples to use from the evaluation dataset. Defaults to None.
        eval_generation_config (Optional[Union[str, dict]]): Model inference configuration for evaluation, provided as
            a JSON string or a dictionary, e.g., `{'max_tokens': 512}`. Defaults to None.
        extra_eval_args (Optional[Union[str, dict]]): Extra arguments for evaluation, provided as a JSON string or a
            dictionary.

        use_galore (bool): Flag to indicate if Galore is used. Default is False.
        galore_target_modules (Optional[List[str]]): List of target modules for Galore. Default is None.
        galore_rank (int): Rank for Galore. Default is 128.
        galore_update_proj_gap (int): Update projection gap for Galore. Default is 50.
        galore_scale (float): Scaling factor for Galore. Default is 1.0.
        galore_proj_type (str): Projection type for Galore. Default is 'std'.
        galore_optim_per_parameter (bool): Flag to indicate if optimization is per parameter for Galore.
            Default is False.
        galore_with_embedding (bool): Flag to indicate if embedding is used with Galore. Default is False.
        galore_quantization (bool): Flag to indicate if use Q-Galore. Default is False.
        galore_proj_quant (bool): Flag to indicate if projection quantization is used for Galore. Default is False.
        galore_proj_bits (int): Number of bits for projection quantization. Default is 4.
        galore_proj_group_size (int): Group size for projection quantization. Default is 256.
        galore_cos_threshold (float): Cosine threshold for projection quantization. Default is 0.4.
        galore_gamma_proj (int): Gamma for projection quantization. Default is 2.
        galore_queue_size (int): Queue size for projection quantization. Default is 5.
        lisa_activated_layers (int): Number of activated layers for LISA. Default is 0.
        lisa_step_interval (int): Step interval for LISA activation. Default is 20.

        use_flash_ckpt (bool): Whether to enable DLRover Flash Checkpoint. When enabled, weights are first saved to
            shared memory and then asynchronously persisted to disk. Currently does not support the safetensors format.
            It is recommended to use this with `PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"` to prevent CUDA OOM
            errors during training. Defaults to False.
    """
    per_device_train_batch_size: int = 1
    per_device_eval_batch_size: int = 1
    gradient_accumulation_steps: Optional[int] = None
    tuner_backend: Optional[str] = None

    gradient_checkpointing: bool = True
    vit_gradient_checkpointing: Optional[bool] = None
    gradient_checkpointing_kwargs: Optional[Union[dict, str]] = None
    logging_first_step: bool = True
    logging_steps: int = 5
    router_aux_loss_coef: float = 0.
    enable_dft_loss: bool = False  # https://arxiv.org/abs/2508.05629
    enable_channel_loss: bool = False

    weight_decay: float = 0.1
    adam_beta2: float = 0.95
    lr_scheduler_type: str = 'cosine'
    lr_scheduler_kwargs: Optional[Union[dict, str]] = None
    report_to: List[str] = field(default_factory=lambda: ['tensorboard'])
    dataloader_num_workers: Optional[int] = None
    dataloader_persistent_workers: bool = False
    dataloader_prefetch_factor: Optional[int] = None
    use_liger_kernel: bool = False

    # extra
    check_model: bool = True
    acc_strategy: Literal['token', 'seq'] = 'token'
    train_dataloader_shuffle: bool = True
    max_epochs: Optional[int] = None
    aligner_lr: Optional[float] = None
    vit_lr: Optional[float] = None
    use_logits_to_keep: Optional[bool] = None
    ds3_gather_for_generation: bool = True
    resume_only_model: bool = False

    # plugins
    optimizer: Optional[str] = None
    loss_type: Optional[str] = field(default=None, metadata={'help': f'loss_func choices: {list(loss_map.keys())}'})
    eval_metric: Optional[str] = None
    callbacks: List[str] = field(default_factory=list)
    # early_step
    early_stop_interval: Optional[int] = None

    # train-eval loop args
    eval_use_evalscope: bool = False
    eval_dataset: List[str] = field(default_factory=list)
    eval_dataset_args: Optional[Union[str, dict]] = None
    eval_limit: Optional[int] = None
    eval_generation_config: Optional[Union[str, dict]] = None
    extra_eval_args: Optional[Union[str, dict]] = None

    # Value copied from SftArguments
    tuner_type: Optional[str] = None

    # galore
    use_galore: bool = False
    galore_target_modules: Optional[List[str]] = None
    galore_rank: int = 128
    galore_update_proj_gap: int = 50
    galore_scale: float = 1.0
    galore_proj_type: str = 'std'
    galore_optim_per_parameter: bool = False
    galore_with_embedding: bool = False
    galore_quantization: bool = False
    galore_proj_quant: bool = False
    galore_proj_bits: int = 4
    galore_proj_group_size: int = 256
    galore_cos_threshold: float = 0.4
    galore_gamma_proj: int = 2
    galore_queue_size: int = 5
    # lisa
    lisa_activated_layers: int = 0
    lisa_step_interval: int = 20

    # dlrover flash_checkpoint
    use_flash_ckpt: bool = False

    # add
    opsd_mask_dir: str = "cache/opd_cache"
    opsd_mask_mode: str = "zoom_in"  # teacher view: "zoom_in" fixed center crop; "adaptive" bbox-scaled crop with a minimum area; "gaussian" Gaussian falloff blur; "original" full image + drawn bbox; "no_mask" full image, no bbox
    opsd_zoom_ratio: float = 2.0  # adaptive: crop = bbox expanded by this factor
    opsd_min_area_frac: float = 0.1  # adaptive: minimum visible area as a fraction of the image
    opsd_gaussian_sigma_ratio: float = 1.5  # gaussian: sigma = max(bbox_w, bbox_h) * sigma_ratio
    opsd_token_weight_mode: str = "linear"  # "uniform": all 1.0; "linear": digit weight N-k (hundreds 3, tens 2, units 1); "uniform-entropy": 1.0 x teacher confidence; "linear-entropy": N-k x teacher confidence
    opsd_non_digit_weight: float = 1.0  # weight of non-digit tokens
    opsd_max_digit_len: int = 0  # linear: cap on the effective length of a digit run (longer runs are truncated). 0 = no cap; 3 suits norm-1000 coordinates (Qwen3-VL); 4 suits absolute pixels (Qwen2.5-VL)
    opsd_weighted_mean: bool = False  # False: loss = sum(w*jsd) / n_tokens (scale drifts with the mean weight); True: loss = sum(w*jsd) / sum(w), comparable across weight modes
    opsd_hint_mode: str = "hint"  # "none": teacher input identical to the student's; "hint": green box + hint text; "gt": hint states the normalized GT center; "spotlight": masked image only, no hint text (with opsd_draw_marker=false: marker-free privilege); "hint_minimal": green box + shortest prohibitive hint (see below)
    opsd_draw_marker: bool = True  # False: do not draw the green GT box on the teacher image. With description-style outputs the marker vocabulary can leak from the teacher distribution into the student; turning the marker off removes the cause
    # ---- Two independent guards against marker leakage; both are no-ops by default ----------
    # Guard 1 = opsd_hint_mode='hint_minimal' (cause side): keep the green box, use the shortest
    #   prohibitive hint. It cuts leakage sharply while keeping most of the teacher's advantage;
    #   'spotlight' removes leakage almost entirely but also most of the privilege.
    #   The hint is given only to the teacher, which never generates, so a longer hint has no format cost.
    # Guard 2 = opsd_marker_words (effect side): zero the weight at positions where a marker token
    #   was sampled, breaking the loop "student mentions the green box -> teacher (which does see one)
    #   agrees -> reverse KL sharpens it".
    #   Comma-separated; empty string (default) skips the whole block.
    #   Choose words by their frequency ratio between contaminated and clean descriptions, not by intuition:
    #     safe to zero: rectangle, surrounded, highlighted, enclosed
    #     keep: box, marked, rectangular, outline, marker, frame (ordinary UI vocabulary)
    #     'green' alone is not diagnostic (legitimate color descriptions use it just as often);
    #     no stemming: 'highlighted' is safe but 'highlight' is not.
    opsd_marker_words: str = ''
    # Sequence-level fallback: if marker tokens exceed this fraction of a row's response tokens, zero
    # the whole row's weights. Rows are not dropped: dropping can deadlock on empty per-rank batches,
    # and teacher/student use two independent boolean gathers that would fall out of alignment.
    # 1.0 (default) with a strict > comparison never triggers.
    opsd_marker_seq_max: float = 1.0
    opsd_ema_decay: float = 0.0  # EMA decay for the teacher model, updated once per optimizer step (independent of GAS). 0 = disabled (fixed teacher), 0.999 = slow, 0.99 = fast

    # TtRT (arXiv:2606.18101), see swift/rlhf_trainers/ttrt_trainer.py. Only active with --use_ttrt true,
    # and requires --opsd_token_weight_mode uniform (TtRT weights are absolute; the base weight must be 1.0).
    # Do not put '_token' in new argument names: swift/utils/utils.py writes such keys as null in args.json.
    ttrt_lambda: float = 3.0  # global coefficient lambda for coordinate-digit weights (paper: 3); w = lambda * g_t * p_t, other tokens 1
    ttrt_alpha: float = 0.5  # soft-gate floor alpha (paper: 0.5): g_t = alpha + (1 - alpha) * h_t; alpha = 0 is the paper's hard gating
    ttrt_free_length: bool = False  # False (faithful): reachability uses only the student's remaining digits for this number; True: allow the number to end early

    # OPSD+GRPO / RSTG (arXiv:2608.00782), see swift/rlhf_trainers/opsd_grpo_trainer.py.
    # Only active with --use_opsd_grpo true; rlhf_type stays 'grpo'. NZG routing + an SFT term on the
    # GT tool call, plus an optional OPD term (nzg_opd_*). Do not put '_token' in these names (see above).
    nzg_reward_name: str = 'GroundAcc'  # binary reward column that decides "whole group failed"; a weighted total mixing in g2acc is continuous and never has exactly zero variance
    nzg_sft_beta_peak: float = 0.0  # SFT-term coefficient peak; 0 = off (plain GRPO). RSTG beta_init = 5e-3
    nzg_sft_beta_valley: float = 0.0  # final annealed value; RSTG beta_min = 1e-3
    nzg_sft_warmup_steps: int = 0  # with decay = 0 as well, beta stays at peak (no annealing)
    nzg_sft_decay_steps: int = 0  # cosine annealing via mu_schedule_function
    nzg_sft_all_rollouts: bool = False  # False: supervise one rollout per NZG group; True: supervise all K
    nzg_allow_mixed_reward: bool = False  # by default a second non-zero-weight reward is rejected: if GroundAcc is all 0 but g2acc varies, A_GRPO is non-zero and the group is not an NZG group
    # --- OPD term. Off by default; with nzg_opd_beta_peak <= 0 opsd_trainer is not even imported,
    # so other GRPO/OPSD/TtRT scripts are unaffected. Requires --teacher_model.
    nzg_opd_beta_peak: float = 0.0  # OPD-term coefficient peak; 0 = off
    nzg_opd_beta_valley: float = 0.0
    nzg_opd_warmup_steps: int = 0
    nzg_opd_decay_steps: int = 0
    nzg_opd_all_rollouts: bool = False  # False: distill only the group's owner rollout; True: distill all K (K times the cost)
    allow_single_generation: bool = False  # for K=1 pure OPSD: lift GRPO's num_generations>=2 check. Default False = unchanged behaviour
    nzg_opd_weight_mode: str = 'position'  # OPD token weights: 'uniform' (all 1.0) | 'position' (coordinate digits beta * k_t, k_t = units 1 / tens 2 / hundreds 3; other tokens 1.0)
    nzg_opd_pos_beta: float = 1.0  # beta for --nzg_opd_weight_mode position; ignored otherwise
    nzg_route_all: bool = False  # pure-OPSD ablation: route EVERY group to the teacher, not just all-miss groups; with --reward_weights 0 the GRPO term is off. Default False = unchanged behaviour
    nzg_log_routing: bool = False  # write one line per group per rollout batch to <output_dir>/nzg_routing.jsonl, for post-hoc attribution of which groups were fixed

    @staticmethod
    def _patch_liger_kernel():
        # fix logits_to_keep
        from liger_kernel.transformers.model import loss_utils
        origin_LigerForCausalLMLoss = loss_utils.LigerForCausalLMLoss

        def LigerForCausalLMLoss(hidden_states, *args, **kwargs):
            hidden_states = hidden_states.contiguous()
            for key in ['cu_seq_lens_q', 'cu_seq_lens_k', 'max_length_q', 'max_length_k']:
                kwargs.pop(key, None)
            return origin_LigerForCausalLMLoss(hidden_states, *args, **kwargs)

        loss_utils.LigerForCausalLMLoss = LigerForCausalLMLoss

    def _init_liger(self):
        if self.use_liger_kernel:
            assert is_liger_available(), 'use_liger_kernel requires liger_kernels, try `pip install liger-kernel`'
            try:
                self._patch_liger_kernel()
            except Exception:
                logger.warning('Failed to patch liger_kernel')

    def _init_callbacks(self):
        if self.lisa_activated_layers > 0:
            self.callbacks.append('lisa')
        if self.tuner_type == 'adalora':
            self.callbacks.append('adalora')
        if self.early_stop_interval is not None and self.early_stop_interval > 0:
            self.callbacks.append('early_stop')
        fsdp_config = getattr(self, 'fsdp_config', {})
        if isinstance(fsdp_config, dict) and fsdp_config.get('activation_cpu_offload', False):
            self.callbacks.append('activation_cpu_offload')

    def __post_init__(self):
        if hasattr(self, 'output_dir'):
            self.output_dir = os.path.abspath(os.path.expanduser(self.output_dir))
        if is_mp() and self.use_liger_kernel:
            raise ValueError('liger_kernel does not support device_map. '
                             'Please use DDP/DeepSpeed for multi-GPU training.')

        if self.optimizer is None and (self.vit_lr is not None or self.aligner_lr is not None):
            self.optimizer = 'multimodal'
        self._init_callbacks()
        if self.gradient_accumulation_steps is None:
            world_size = get_dist_setting()[2]
            self.gradient_accumulation_steps = max(1, math.ceil(16 / self.per_device_train_batch_size / world_size))
            logger.info(f'Setting args.gradient_accumulation_steps: {self.gradient_accumulation_steps}')
        if self.lr_scheduler_kwargs:
            self.lr_scheduler_kwargs = json_parse_to_dict(self.lr_scheduler_kwargs)
        if 'wandb' in self.report_to:
            os.environ.setdefault('WANDB_PROJECT', 'ms-swift')
        if self.vit_gradient_checkpointing is None:
            self.vit_gradient_checkpointing = self.gradient_checkpointing
        if self.gradient_checkpointing_kwargs:
            self.gradient_checkpointing_kwargs = json_parse_to_dict(self.gradient_checkpointing_kwargs)
        self._init_liger()
        if self.dataloader_num_workers is None:
            if platform.system() == 'Windows':
                self.dataloader_num_workers = 0
            else:
                self.dataloader_num_workers = 1
            logger.info(f'Setting args.dataloader_num_workers: {self.dataloader_num_workers}')
        if self.dataloader_prefetch_factor is None and self.dataloader_num_workers > 0:
            self.dataloader_prefetch_factor = 2
        if self.eval_use_evalscope:
            try:
                import evalscope
            except ImportError:
                raise ImportError('evalscope is not installed, please install it by `pip install evalscope`')
            self.eval_dataset_args = json_parse_to_dict(self.eval_dataset_args)
            self.eval_generation_config = json_parse_to_dict(self.eval_generation_config)
            self.extra_eval_args = json_parse_to_dict(self.extra_eval_args)


@dataclass
class TrainingArguments(TrainArgumentsMixin, HfTrainingArguments):

    def __post_init__(self):
        TrainArgumentsMixin.__post_init__(self)
        HfTrainingArguments.__post_init__(self)


@dataclass
class Seq2SeqTrainingArguments(TrainArgumentsMixin, HfSeq2SeqTrainingArguments):

    def __post_init__(self):
        TrainArgumentsMixin.__post_init__(self)
        HfSeq2SeqTrainingArguments.__post_init__(self)
