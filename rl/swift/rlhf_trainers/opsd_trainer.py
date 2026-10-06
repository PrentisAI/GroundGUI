# Copyright (c) ModelScope Contributors. All rights reserved.
import inspect
import os
import random
from collections import defaultdict, deque
from contextlib import contextmanager, nullcontext
from copy import deepcopy
from enum import Enum
from typing import Dict, Optional, Union
import json 

import torch
import torch.nn as nn
import torch.nn.functional as F
import trl
from accelerate.utils import gather_object, is_peft_model
from packaging import version
from transformers import PreTrainedModel
from trl import SFTTrainer as HFSFTTrainer

try:  # trl>=0.29 relocated GKD under trl.experimental
    from trl import GKDTrainer as HFGKDTrainer
except ImportError:
    from trl.experimental.gkd import GKDTrainer as HFGKDTrainer

from swift.template import TemplateInputs
from swift.trainers import SwiftMixin, disable_gradient_checkpointing
from swift.utils import (JsonlWriter, get_logger, is_swanlab_available, is_wandb_available, remove_response, to_device,
                         unwrap_model_for_generation)
from .rollout_mixin import DataType, RolloutTrainerMixin
from .utils import (get_gather_if_zero3_context, identity_data_collator, prepare_deepspeed, profiling_context, 
                    profiling_decorator, swanlab_has_run)
import math
import numpy as np
from PIL import Image, ImageDraw, ImageFilter

try:
    from liger_kernel.chunked_loss import LigerFusedLinearJSDLoss
    _liger_kernel_available = True
except ImportError:
    _liger_kernel_available = False

# trl>=0.28 splits GKD into a public GKDTrainer plus a trl.experimental base, and
# BOTH define __init__. swift deliberately bypasses trl's __init__ chain (SwiftMixin
# routes to transformers.Trainer), so the __init__ must be dropped at every GKD level:
# leaving the experimental one alive makes it forward `peft_config` into
# transformers.Trainer.__init__, which rejects it. Guarded so importing the gkd and
# opsd trainer modules in the same process stays idempotent.
for _gkd_cls in HFGKDTrainer.__mro__:
    if _gkd_cls.__name__ == 'GKDTrainer' and '__init__' in _gkd_cls.__dict__:
        del _gkd_cls.__init__
if '__init__' in HFSFTTrainer.__dict__:
    del HFSFTTrainer.__init__

logger = get_logger()
if is_wandb_available():
    import wandb
if is_swanlab_available():
    import swanlab


class DataSource(str, Enum):
    STUDENT = 'student'  # On-policy: student model generates responses
    TEACHER = 'teacher'  # Sequential KD: teacher model generates responses
    DATASET = 'dataset'  # Off-policy: use dataset responses


class OPSDTrainer(RolloutTrainerMixin, SwiftMixin, HFGKDTrainer):

    def __init__(self, model: Optional[Union[PreTrainedModel, nn.Module, str]] = None, *_args, **kwargs):
        teacher_model = kwargs.pop('teacher_model')
        teacher_deepspeed_config = kwargs.pop('teacher_deepspeed_config', None)
        self.vllm_client = kwargs.pop('vllm_client', None)
        super().__init__(model, None, *_args, **kwargs)
        args = kwargs['args']
        self.lmbda = args.lmbda
        self.temperature = args.temperature
        self.seq_kd = args.seq_kd
        self.generation_config = model.generation_config
        self._metrics = {'train': defaultdict(list), 'eval': defaultdict(list)}
        self._total_train_tokens = 0

        # Initialize logging components
        self._prepare_logging()

        # Initialize liger loss
        self._prepare_liger_loss()

        self.teacher_ds3_gather_for_generation = args.ds3_gather_for_generation
        self.is_teacher_ds3 = None
        # Initialize teacher model
        if self.is_deepspeed_enabled:
            if teacher_deepspeed_config is not None:
                self.is_teacher_ds3 = teacher_deepspeed_config.get('zero_optimization', {}).get('stage') == 3
                if not self.is_teacher_ds3:
                    self.teacher_ds3_gather_for_generation = False
                self.teacher_model = prepare_deepspeed(
                    teacher_model, self.accelerator, deepspeed_config=teacher_deepspeed_config, training_args=args)
            else:
                self.teacher_model = prepare_deepspeed(teacher_model, self.accelerator)
        elif self.is_fsdp_enabled:
            from .utils import prepare_fsdp
            self.teacher_model = prepare_fsdp(teacher_model, self.accelerator)
        else:
            self.teacher_model = self.accelerator.prepare_model(teacher_model, evaluation_mode=True)
        self.teacher_model.eval()
        if self.args.offload_teacher_model:
            self.offload_model(self.accelerator.unwrap_model(self.teacher_model))

        # Initialize EMA for teacher model
        self.ema_decay = getattr(args, 'opsd_ema_decay', 0.0)
        self._last_ema_step = 0
        if self.ema_decay > 0:
            self._check_ema_params()
            logger.info(f"EMA enabled for teacher model with decay={self.ema_decay} per optimizer step")

        # Initialize rollout infrastructure for vLLM support
        self.prepare_rollout()

        # Initialize activation offloading context
        args.activation_offloading = False  # TODO: remove
        if args.activation_offloading:
            from trl.models import get_act_offloading_ctx_manager
            self.maybe_activation_offload_context = get_act_offloading_ctx_manager(model=self.model)
        else:
            self.maybe_activation_offload_context = nullcontext()
        self._trl_version_gte_0_24 = version.parse(trl.__version__) >= version.parse('0.24')

        # Initialize resample data iterator for truncation_strategy 'raise'('delete')
        if self.template.truncation_strategy == 'raise':
            self._prepare_resample_data_iterator()

    def _get_data_collator(self, args, template):
        return identity_data_collator

    # Code borrowed from huggingface/trl
    def generate_on_policy_outputs(self, model, inputs, generation_config, pad_token_id=None):
        """Generate on-policy outputs using the model.

        When encode_prompt_only=True, inputs['input_ids'] already contains only the prompt part.
        """
        assert not self.template.padding_free, 'generate not support padding_free/packing.'
        prompt_input_ids = inputs['input_ids']
        model_inputs = {k: v for k, v in inputs.items() if k != 'labels'}
        model_inputs.pop('position_ids', None)
        model_inputs.pop('text_position_ids', None)
        kwargs = {}
        base_model = self.template.get_base_model(model)
        parameters = inspect.signature(base_model.generate).parameters
        if 'use_model_defaults' in parameters:
            kwargs['use_model_defaults'] = False
        with self.template.generate_context():
            if self.model.model_meta.is_multimodal:
                _, model_inputs = self.template.pre_forward_hook(model, None, model_inputs)
            generated_outputs = model.generate(
                **model_inputs, generation_config=generation_config, return_dict_in_generate=True, **kwargs)
        # Get the generated token IDs
        generated_tokens = generated_outputs.sequences
        if not self.template.skip_prompt:
            generated_tokens = torch.concat([prompt_input_ids, generated_tokens], dim=1)
        # Calculate new attention mask
        new_attention_mask = torch.ones_like(generated_tokens)
        new_labels = generated_tokens.clone()
        new_labels[:, :prompt_input_ids.shape[1]] = -100

        # If there's pad_token_id, set attention mask to 0 for padding tokens
        if pad_token_id is not None:
            new_labels[new_labels == pad_token_id] = -100
            new_attention_mask[generated_tokens == pad_token_id] = 0

        new_position_ids = new_attention_mask.cumsum(dim=1) - 1
        new_position_ids[new_position_ids < 0] = 0
        inputs['position_ids'] = new_position_ids
        return generated_tokens, new_attention_mask, new_labels


    @profiling_decorator
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        # Student and teacher are encoded separately (see _prepare_batch_inputs)
        student_inputs = inputs["student_inputs"]
        teacher_inputs = inputs["teacher_inputs"]

        data_source = inputs.pop('_data_source', DataSource.DATASET)  # set on the outer dict by training_step
        
        # Model forward kwargs (drop non-model keys)
        student_model_inputs = {k: v for k, v in student_inputs.items() if k not in {'prompt', 'labels'}}
        teacher_model_inputs = {k: v for k, v in teacher_inputs.items() if k not in {'prompt', 'labels'}}

        use_logits_to_keep = self.get_use_logits_to_keep(True)
        if use_logits_to_keep and not self.use_liger_gkd_loss:
            self.prepare_logits_to_keep(student_inputs)
            student_model_inputs['logits_to_keep'] = student_inputs['logits_to_keep']
            # logits_to_keep is applied to the student forward only

        if self.use_liger_gkd_loss:
            assert False 
        else:
            # Standard loss computation
            if self.args.sft_alpha > 0:
                student_model_inputs['labels'] = student_inputs['labels']
            
            # Each model gets its own inputs
            outputs_student = model(**student_model_inputs)
            student_model_inputs.pop('labels', None)

            load_context = self.load_teacher_model_context() if self.args.offload_teacher_model else nullcontext()
            with torch.no_grad(), load_context, disable_gradient_checkpointing(self.teacher_model, self.args.gradient_checkpointing_kwargs):
                outputs_teacher = self.teacher_model(**teacher_model_inputs)

            # Logit alignment: teacher and student prompts differ (hint text, privileged image),
            # so their prefix lengths differ and raw logits do not line up. The labels == -100
            # masks drop each side's prompt and keep only the response positions for the KL.
            shifted_student_labels = torch.roll(student_inputs['labels'], shifts=-1, dims=1)
            shifted_teacher_labels = torch.roll(teacher_inputs['labels'], shifts=-1, dims=1)
            
            mask_student = shifted_student_labels != -100
            mask_teacher = shifted_teacher_labels != -100

            # =========================================================
            # Token weights: two orthogonal factors combined
            #   base:    "linear"  -> digit at position k of an N-digit run gets N-k,
            #                         non-digits get non_digit_w
            #            "uniform" -> all 1.0
            #   entropy: mode contains "entropy" -> multiply by (1 - H/H_max) of the teacher
            #
            # Modes:
            #   "uniform"          - all 1.0
            #   "linear"           - digits N-k, non-digits non_digit_w
            #   "uniform-entropy"  - 1.0 x teacher confidence
            #   "linear-entropy"   - digits N-k x teacher confidence
            # =========================================================
            if not hasattr(self, '_digit_tokens'):
                self._digit_tokens = set(self.template.tokenizer.encode(str(i), add_special_tokens=False)[-1] for i in range(10))
            
            weight_mode = getattr(self.args, 'opsd_token_weight_mode', 'linear')
            non_digit_w = getattr(self.args, 'opsd_non_digit_weight', 1.0)
            max_digit_len = getattr(self.args, 'opsd_max_digit_len', 0)  # 0 = no cap
            
            # Step 1: position-aware base weights
            if 'linear' in weight_mode:
                weights = torch.full_like(shifted_student_labels, non_digit_w, dtype=torch.float32)
                seqs = shifted_student_labels.tolist()
                for b, seq in enumerate(seqs):
                    i = 0
                    while i < len(seq):
                        if seq[i] in self._digit_tokens:
                            j = i
                            while j < len(seq) and seq[j] in self._digit_tokens:
                                j += 1
                            N = j - i
                            # max_digit_len caps the weight, e.g. with max_digit_len=3 the
                            # 4-digit "1920" gets [3,3,2,1] instead of [4,3,2,1]. This keeps the
                            # weight profile consistent across coordinate ranges (norm1000 vs pixels).
                            effective_N = min(N, max_digit_len) if max_digit_len > 0 else N
                            for k in range(N):
                                # Counted from the end: last digit = 1, second-to-last = 2, ...
                                # higher positions are capped at effective_N
                                pos_from_end = N - k  # N, N-1, ..., 2, 1
                                weights[b, i+k] = float(min(pos_from_end, effective_N))
                            i = j
                        else:
                            i += 1
            else:
                # uniform
                weights = torch.ones_like(shifted_student_labels, dtype=torch.float32)

            # =========================================================
            # Step 1b: marker-word veto. Skipped entirely when opsd_marker_words is empty (default).
            #
            # Why: the teacher's image carries a green box, so words like "green rectangle" are
            # *correct* under the teacher. Once the student samples them, reverse KL sharpens
            # them, and a lagging EMA teacher confirms the drift instead of correcting it; rollout
            # descriptions can degenerate into loops like "The green rectangle, located within the
            # green rectangle, ...". Zeroing the weight at positions where a marker token was
            # sampled breaks this feedback loop: no gradient there, so it is never reinforced.
            #
            # Placement: after both base-weight branches (the 'linear' N-k ladder is already
            # written; this only zeroes, never adds) and before the weights[mask_student] gather
            # below (so the 'entropy' confidence multiply inherits the zeros: 0 * conf = 0).
            # =========================================================
            _marker_words = getattr(self.args, 'opsd_marker_words', '') or ''
            if _marker_words.strip():
                if not hasattr(self, '_marker_tokens'):
                    _tk = self.template.tokenizer
                    _ids, _dropped = set(), []
                    for _w in _marker_words.split(','):
                        _w = _w.strip()
                        if not _w:
                            continue
                        # Collect case / leading-space variants: " green" and "green" are different ids.
                        for _v in (_w, ' ' + _w, _w.capitalize(), ' ' + _w.capitalize(),
                                   _w.upper(), ' ' + _w.upper()):
                            _enc = _tk.encode(_v, add_special_tokens=False)
                            if len(_enc) == 1:
                                _ids.add(_enc[0])
                            else:
                                # Drop multi-token variants entirely; never take the first token:
                                # 'highlighted' -> ['highlight', 'ed'], and zeroing 'highlight'
                                # would also suppress legitimate uses of that word.
                                _dropped.append(_v)
                    # Digit tokens must be excluded, otherwise a coordinate digit that happens to
                    # match would silently weaken the coordinate target.
                    _ids -= self._digit_tokens
                    self._marker_tokens = _ids
                    logger.info(f'[opsd marker veto] words={_marker_words!r} -> {len(_ids)} single-token ids: '
                                f'{sorted(_ids)}; dropped {len(_dropped)} multi-token variants: {_dropped}')
                if self._marker_tokens:
                    _mk = torch.zeros_like(shifted_student_labels, dtype=torch.bool)
                    for _tid in self._marker_tokens:
                        _mk |= (shifted_student_labels == _tid)
                    _mk &= mask_student          # response positions only, never prompt / -100
                    weights = weights.masked_fill(_mk, 0.0)
                    # Sequence-level fallback: if the fraction of marker tokens in a response exceeds
                    # opsd_marker_seq_max, zero the whole row (a degenerate loop's coordinates are
                    # unreliable too).
                    _seq_max = float(getattr(self.args, 'opsd_marker_seq_max', 1.0))
                    if _seq_max < 1.0:
                        _n_mk = _mk.sum(dim=1).float()
                        _n_resp = mask_student.sum(dim=1).float().clamp_min(1.0)
                        _kill = (_n_mk / _n_resp) > _seq_max        # strict >, so 1.0 never fires
                        weights = weights.masked_fill(_kill[:, None] & mask_student, 0.0)

            # Weights at valid positions (aligned with mask_student)
            valid_weights = weights[mask_student]
            
            # Step 2: entropy modes multiply by teacher confidence (normalized to [0, 1]).
            # Teacher and student sequence lengths differ, so extract the teacher's valid
            # positions first and compute entropy on those.
            if 'entropy'  in weight_mode:
                # Teacher logits at valid response positions [N_valid, V]
                valid_teacher_logits = outputs_teacher.logits[mask_teacher]
                teacher_probs = F.softmax(valid_teacher_logits, dim=-1)  # [N_valid, V]
                token_entropy = -(teacher_probs * torch.log(teacher_probs + 1e-8)).sum(dim=-1)  # [N_valid]
                del teacher_probs, valid_teacher_logits
                # Normalized confidence: 1 = teacher fully certain, 0 = uniform distribution
                vocab_size = outputs_teacher.logits.size(-1)
                H_max = torch.log(torch.tensor(vocab_size, dtype=token_entropy.dtype, device=token_entropy.device))
                confidence = 1.0 - token_entropy / H_max  # [N_valid], in [0, 1]
                valid_weights = valid_weights * confidence
            # =========================================================

            # Valid response logits (student and teacher responses must be token-identical,
            # otherwise the shapes do not match)
            shifted_student_logits = outputs_student.logits[mask_student][None]
            shifted_teacher_logits = outputs_teacher.logits[mask_teacher][None]

            # Generalized JSD loss
            loss = self.generalized_jsd_loss(
                student_logits=shifted_student_logits,
                teacher_logits=shifted_teacher_logits,
                beta=self.beta,
                weights=valid_weights,
                weighted_mean=getattr(self.args, 'opsd_weighted_mean', False),
            )
            # Add SFT loss if enabled (skip for student-generated responses)
            if self.args.sft_alpha > 0 and data_source != DataSource.STUDENT:
                loss = loss + self.args.sft_alpha * outputs_student.loss

        # Return loss
        if return_outputs:
            if self.use_liger_gkd_loss:
                # outputs has been released in liger loss computation to reduce peak memory
                outputs_student = None
            return (loss, outputs_student)
        else:
            return loss

    def mask_processor(self, data):
        if isinstance(data['additional_paras'], str):
            data['additional_paras'] = json.loads(data['additional_paras'])

        image_path = data['images'][0]['path']
        x1, y1, x2, y2 = map(int, data['solution']['arguments']['coordinate'])
        
        save_path = os.path.join(self.args.opsd_mask_dir, f"{data['sample_id']}_{os.path.basename(image_path)}")
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        
        img = Image.open(image_path).convert("RGB")
        W, H = img.size
        
        mask_mode = getattr(self.args, 'opsd_mask_mode', 'zoom_in')
        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
        bbox_w, bbox_h = x2 - x1, y2 - y1
        
        if mask_mode == 'zoom_in':
            # Keep a fixed W/2 x H/2 window around the bbox center, black elsewhere
            res = Image.new("RGB", (W, H), "black") 
            crop_x1 = max(0, cx - W // 4)
            crop_y1 = max(0, cy - H // 4)
            crop_x2 = min(W, cx + W // 4)
            crop_y2 = min(H, cy + H // 4)
            if crop_x1 < crop_x2 and crop_y1 < crop_y2:
                crop_box = (crop_x1, crop_y1, crop_x2, crop_y2)
                res.paste(img.crop(crop_box), (crop_x1, crop_y1))

        elif mask_mode == 'adaptive':
            # Window scaled from the bbox size, with a minimum exposed area
            zoom_ratio = getattr(self.args, 'opsd_zoom_ratio', 2.0)
            min_area_frac = getattr(self.args, 'opsd_min_area_frac', 0.1)
            
            # Expand outward relative to the bbox size
            pad_w = int(bbox_w * zoom_ratio)
            pad_h = int(bbox_h * zoom_ratio)
            
            # Expose at least min_area_frac of the original image area
            min_half_w = int(W * math.sqrt(min_area_frac) / 2)
            min_half_h = int(H * math.sqrt(min_area_frac) / 2)
            pad_w = max(pad_w, min_half_w)
            pad_h = max(pad_h, min_half_h)
            
            crop_x1 = max(0, cx - pad_w)
            crop_y1 = max(0, cy - pad_h)
            crop_x2 = min(W, cx + pad_w)
            crop_y2 = min(H, cy + pad_h)
            
            res = Image.new("RGB", (W, H), "black")
            if crop_x1 < crop_x2 and crop_y1 < crop_y2:
                crop_box = (crop_x1, crop_y1, crop_x2, crop_y2)
                res.paste(img.crop(crop_box), (crop_x1, crop_y1))

        elif mask_mode == 'gaussian':
            # Soft Gaussian mask: pixels darken with distance from the bbox; the bbox
            # interior is unchanged
            sigma_ratio = getattr(self.args, 'opsd_gaussian_sigma_ratio', 1.5)
            min_area_frac = getattr(self.args, 'opsd_min_area_frac', 0.1)
            sigma = max(bbox_w, bbox_h) * sigma_ratio
            # Floor sigma at short_side * sqrt(min_area_frac) so small targets do not decay too fast
            min_sigma = min(W, H) * math.sqrt(min_area_frac)
            sigma = max(sigma, min_sigma)
            
            img_arr = np.array(img, dtype=np.float32)
            
            # Distance from each pixel to the bbox (0 inside the bbox)
            xs = np.arange(W)[None, :]  # (1, W)
            ys = np.arange(H)[:, None]  # (H, 1)
            
            dx = np.maximum(x1 - xs, 0) + np.maximum(xs - x2, 0)  # (H, W)
            dy = np.maximum(y1 - ys, 0) + np.maximum(ys - y2, 0)  # (H, W)
            dist = np.sqrt(dx.astype(np.float64)**2 + dy.astype(np.float64)**2)
            
            # Gaussian falloff: alpha = 1 inside the bbox, decaying with distance outside
            alpha = np.exp(-dist**2 / (2 * sigma**2)).astype(np.float32)
            alpha = alpha[:, :, None]  # (H, W, 1), broadcast over RGB
            
            res_arr = (img_arr * alpha).astype(np.uint8)
            res = Image.fromarray(res_arr)

        elif mask_mode == 'no_mask':
            # no_mask: original image, no bbox drawn
            res = img.copy()

        else:
            # original: original image with the bbox drawn
            res = img.copy()
        
        # Draw the green gt bbox (except in no_mask mode; opsd_draw_marker=false disables it
        # globally, since marker words can leak through the teacher distribution into the
        # student's description tokens; see that field in arguments.py)
        if mask_mode != 'no_mask' and getattr(self.args, 'opsd_draw_marker', True):
            line_width = 5
            draw_x1 = x1 - line_width
            draw_y1 = y1 - line_width
            draw_x2 = x2 + line_width
            draw_y2 = y2 + line_width
            ImageDraw.Draw(res).rectangle(
                [draw_x1, draw_y1, draw_x2, draw_y2], 
                outline="green", 
                width=line_width
            )
        
        res.save(save_path)
        
        return save_path

 
    def _prepare_batch_inputs(self, inputs: list, encode_prompt_only: bool = False) -> Dict[str, torch.Tensor]:
        from .utils import replace_assistant_response_with_ids

        template = self.template
        student_encoded_inputs = []
        teacher_encoded_inputs = []

        mode = 'transformers' if encode_prompt_only else 'train'
        with self._template_context(template, mode=mode):
            for data in inputs:
                # ------ Student inputs ------
                student_data = deepcopy(data)
                if 'response_token_ids' in student_data and student_data['response_token_ids']:
                    student_data['messages'] = replace_assistant_response_with_ids(student_data['messages'], student_data['response_token_ids'])
                if encode_prompt_only and student_data.get('messages') and student_data['messages'][-1].get('role') == 'assistant':
                    student_data['messages'][-1]['content'] = None
                student_encoded = template.encode(student_data, return_length=True)
                student_encoded_inputs.append(student_encoded)

                # ------ Teacher inputs (privileged) ------
                teacher_data = deepcopy(data)
                hint_mode = getattr(self.args, 'opsd_hint_mode', 'hint')
                if 'messages' in teacher_data:
                    assert teacher_data['messages'][1]['role'] == "user"

                    if hint_mode == 'none':
                        # No hint: teacher input is identical to the student's
                        pass
                    elif hint_mode == 'hint':
                        # Green-box masked image + hint text
                        teacher_data['messages'][1]['content'] += " Hint: The answer is located within the green rectangle."
                        mask_img_path = self.mask_processor(teacher_data)
                        teacher_data['images'][0]['path'] = mask_img_path
                    elif hint_mode == 'gt':
                        # Green-box masked image + the gt normalized center coordinate
                        gt_bbox = teacher_data['solution']['arguments']['coordinate']
                        additional = teacher_data.get('additional_paras', '{}')
                        if isinstance(additional, str):
                            additional = json.loads(additional)
                        image_size = additional['image_size']
                        # Normalized center, consistent with data construction: bbox -> norm1000 -> center
                        norm_bbox = [
                            int(gt_bbox[0] / image_size[0] * 1000),
                            int(gt_bbox[1] / image_size[1] * 1000),
                            int(gt_bbox[2] / image_size[0] * 1000),
                            int(gt_bbox[3] / image_size[1] * 1000),
                        ]
                        norm_cx = (norm_bbox[0] + norm_bbox[2]) // 2
                        norm_cy = (norm_bbox[1] + norm_bbox[3]) // 2
                        gt_tool_call = f'<tool_call>\n{{"name": "computer_use", "arguments": {{"action": "left_click", "coordinate": [{norm_cx}, {norm_cy}]}}}}\n</tool_call>'
                        user_text = teacher_data['messages'][1]['content']
                        teacher_data['messages'][1]['content'] = (
                            f"{user_text}\n\n"
                            f"The correct answer: {gt_tool_call}"
                        )

                        mask_img_path = self.mask_processor(teacher_data)
                        teacher_data['images'][0]['path'] = mask_img_path
                    elif hint_mode == 'hint_minimal':
                        # Green box kept + the shortest possible prohibition. Putting the prohibition
                        # first and naming the rectangle only once keeps most of the teacher's
                        # privileged gain while leaking far fewer marker words into the student than
                        # 'hint'; longer, more verbose guard wording leaked more, so keep this
                        # sentence verbatim.
                        teacher_data['messages'][1]['content'] += (
                            " Hint: Never mention the green rectangle."
                            " The answer is the element inside it.")
                        mask_img_path = self.mask_processor(teacher_data)
                        teacher_data['images'][0]['path'] = mask_img_path
                    elif hint_mode == 'spotlight':
                        # Privileged image only, no hint text. Combined with --opsd_draw_marker false,
                        # both leakage channels (hint sentence, visible marker) are closed and the
                        # teacher's privilege comes only from the masked image's focus.
                        mask_img_path = self.mask_processor(teacher_data)
                        teacher_data['images'][0]['path'] = mask_img_path
                    else:
                        raise ValueError(f"Unknown opsd_hint_mode: {hint_mode}, expected 'none', 'hint', 'gt', 'hint_minimal', or 'spotlight'")

                if 'response_token_ids' in teacher_data and teacher_data['response_token_ids']:
                    teacher_data['messages'] = replace_assistant_response_with_ids(teacher_data['messages'], teacher_data['response_token_ids'])
                if encode_prompt_only and teacher_data.get('messages') and teacher_data['messages'][-1].get('role') == 'assistant':
                    teacher_data['messages'][-1]['content'] = None
                teacher_encoded = template.encode(teacher_data, return_length=True)
                teacher_encoded_inputs.append(teacher_encoded)

            # Pad and collate each side separately
            student_batch = to_device(template.data_collator(student_encoded_inputs), self.model.device)
            teacher_batch = to_device(template.data_collator(teacher_encoded_inputs), self.model.device)

        # Return both for compute_loss
        return {
            "student_inputs": student_batch,
            "teacher_inputs": teacher_batch
        }


    # Code borrowed from huggingface/trl
    @profiling_decorator
    def training_step(self,
                      model: nn.Module,
                      inputs: DataType,
                      num_items_in_batch: Optional[int] = None) -> torch.Tensor:
        """
        Perform a training step for the Generalized Knowledge Distillation (GKD) model.

        This method implements the on-policy learning approach described in the GKD paper.
        With probability `self.lmbda`, it generates new responses using the student model,
        which are then used for training instead of the original inputs.

        When use_vllm is enabled, vLLM engine is used for faster generation.
        """
        args = self.args
        # EMA the teacher once per OPTIMIZER step. training_step runs per micro-batch and the
        # student is frozen inside an accumulation window, so an unguarded call would apply
        # decay**gradient_accumulation_steps per step (0.95 -> 0.44 at GAS=16).
        if self.ema_decay > 0 and self.state.global_step != self._last_ema_step:
            self._last_ema_step = self.state.global_step
            self._ema_update_teacher()

        with profiling_context(self, 'get_completions'):
            if self._get_random_num() <= self.lmbda:
                # On-policy: student model generates responses
                data_source = DataSource.STUDENT
                # Resample inputs that fail encoding when truncation_strategy is 'raise'('delete')
                if self.template.truncation_strategy == 'raise':
                    inputs = self.resample_encode_failed_inputs(inputs)
                if args.use_vllm:
                    processed_inputs = self._preprocess_inputs(inputs)
                    generated_inputs = self._fast_infer(processed_inputs)
                    if self.log_completions:
                        messages = [inp['messages'][:-1] for inp in generated_inputs]
                        completions = [deepcopy(inp['messages'][-1]['content']) for inp in generated_inputs]
                        valid_messages = gather_object(messages)
                        valid_completions = gather_object(completions)
                        self._logs['prompt'].extend(self._apply_chat_template_to_messages_list(valid_messages))
                        self._logs['completion'].extend(valid_completions)
                    with self._template_context(self.template):
                        # vLLM already generated response, encode full messages
                        encoded_inputs = self._prepare_batch_inputs(generated_inputs, encode_prompt_only=False)
                else:
                    # Need prompt-only encoding for on-policy generation
                    encoded_inputs = self._prepare_batch_inputs(inputs, encode_prompt_only=True)
                    with unwrap_model_for_generation(
                            model, self.accelerator,
                            gather_deepspeed3_params=args.ds3_gather_for_generation) as unwrapped_model:
                        unwrapped_model.eval()
                        new_input_ids, new_attention_mask, new_labels = self.generate_on_policy_outputs(
                            unwrapped_model, encoded_inputs, self.generation_config, self.processing_class.pad_token_id)
                        unwrapped_model.train()
                    # override with generated inputs
                    encoded_inputs['input_ids'] = new_input_ids
                    encoded_inputs['attention_mask'] = new_attention_mask
                    encoded_inputs['labels'] = new_labels

            elif self.seq_kd:
                # Sequential KD: teacher model generates responses
                data_source = DataSource.TEACHER

                # Resample inputs that fail encoding when truncation_strategy is 'raise'('delete')
                if self.template.truncation_strategy == 'raise':
                    inputs = self.resample_encode_failed_inputs(inputs)
                # Need prompt-only encoding for teacher generation
                encoded_inputs = self._prepare_batch_inputs(inputs, encode_prompt_only=True)
                load_context = self.load_teacher_model_context() if self.args.offload_teacher_model else nullcontext()
                with load_context, unwrap_model_for_generation(
                        self.teacher_model,
                        self.accelerator,
                        gather_deepspeed3_params=self.teacher_ds3_gather_for_generation) as unwrapped_model:
                    unwrapped_model.eval()
                    new_input_ids, new_attention_mask, new_labels = self.generate_on_policy_outputs(
                        unwrapped_model, encoded_inputs, self.generation_config, self.processing_class.pad_token_id)
                # override with generated inputs
                encoded_inputs['input_ids'] = new_input_ids
                encoded_inputs['attention_mask'] = new_attention_mask
                encoded_inputs['labels'] = new_labels

            else:
                # Off-policy: use dataset responses, encode full messages
                data_source = DataSource.DATASET
                total_length = self.template.max_length + self.max_completion_length
                with self._template_context(self.template, max_length=total_length):
                    encoded_inputs = self._prepare_batch_inputs(inputs, encode_prompt_only=False)

            # Mark data source for downstream processing (e.g., conditional SFT loss)
            encoded_inputs['_data_source'] = data_source

        with self.template.forward_context(self.model, encoded_inputs):
            loss = HFSFTTrainer.training_step(self, model, encoded_inputs, num_items_in_batch)

        return loss

    def _check_ema_params(self):
        """Fail loudly if teacher/student parameter names do not line up.

        The EMA loop looks student parameters up by teacher name; under PEFT every student
        name is prefixed ('base_model.model.') so nothing matches and the EMA would be a
        silent no-op.
        """
        student = {n for n, _ in self.accelerator.unwrap_model(self.model).named_parameters()}
        missing = [
            n for n, _ in self.accelerator.unwrap_model(self.teacher_model).named_parameters() if n not in student
        ]
        if missing:
            raise ValueError(f'opsd_ema_decay={self.ema_decay} but {len(missing)} teacher parameters have no '
                             f'student counterpart (e.g. {missing[:3]}). EMA requires train_type=full; '
                             f'pass --opsd_ema_decay 0 to use a fixed teacher.')

    @torch.no_grad()
    def _ema_update_teacher(self):
        """Update teacher model parameters using EMA of student parameters.

        θ_teacher = decay * θ_teacher + (1 - decay) * θ_student

        Called once per optimizer step from training_step (see the guard there).

        Handles:
        - offload_teacher_model: load teacher to GPU before update, offload after
        - DeepSpeed ZeRO3: gather sharded parameters before update
        """
        decay = self.ema_decay

        # If teacher is offloaded to CPU, load it back to GPU first
        load_context = self.load_teacher_model_context() if self.args.offload_teacher_model else nullcontext()

        # If teacher uses DeepSpeed ZeRO3, need to gather full parameters
        if self.is_teacher_ds3:
            import deepspeed
            gather_context = lambda params: deepspeed.zero.GatheredParameters(params, modifier_rank=0)
        else:
            gather_context = lambda params: nullcontext()

        with load_context:
            student_model = self.accelerator.unwrap_model(self.model)
            teacher_model = self.accelerator.unwrap_model(self.teacher_model)

            student_params = dict(student_model.named_parameters())

            for name, teacher_param in teacher_model.named_parameters():
                student_param = student_params[name]  # KeyError beats skipping silently, see _check_ema_params
                with gather_context([teacher_param]):
                    # Only rank 0 modifies under ZeRO3; for ZeRO2/non-DS all ranks update
                    if not self.is_teacher_ds3 or self.accelerator.is_main_process:
                        teacher_param.data.mul_(decay).add_(student_param.data.to(teacher_param.device), alpha=1.0 - decay)

    def prediction_step(self, model, inputs, *args, **kwargs):
        # Prediction uses full messages
        encoded_inputs = self._prepare_batch_inputs(inputs, encode_prompt_only=False)
        with self.template.forward_context(self.model, encoded_inputs):
            return super().prediction_step(model, encoded_inputs, *args, **kwargs)

    @contextmanager
    def offload_context(self):
        """Context manager for offloading model and optimizer during vLLM inference

        This offloads:
        - Student model (self.model)
        - Optimizer states

        to CPU to free up GPU memory for vLLM engine.
        """
        if self.args.offload_model:
            self.offload_model(self.accelerator.unwrap_model(self.model))
        if getattr(self, 'optimizer', None) and self.args.offload_optimizer:
            self.offload_optimizer()

        try:
            yield
        finally:
            # reload (load back) model when exiting context
            if self.args.offload_model:
                self.load_model(self.accelerator.unwrap_model(self.model))
            if getattr(self, 'optimizer', None) and self.args.offload_optimizer:
                self.load_optimizer()

    def _get_random_num(self) -> float:
        """
        Generate a deterministic random number.

        Uses an isolated Random instance to avoid interfering with the global
        random state, ensuring thread-safety and consistent behavior across processes.

        Returns:
            float: A random number in the range [0.0, 1.0).
        """
        seed = int(getattr(self.args, 'seed', 0))
        seed += int(self.state.global_step)
        rng = random.Random(seed)
        return rng.random()

    @contextmanager
    def load_teacher_model_context(self):
        """
        Context manager to load and offload the teacher model with memory and timing profiling.
        """
        if not self.args.offload_teacher_model:
            yield
            return

        self.load_model(self.accelerator.unwrap_model(self.teacher_model))
        yield
        self.offload_model(self.accelerator.unwrap_model(self.teacher_model))

    def _prepare_liger_loss(self):
        """Initialize liger loss if enabled."""
        args = self.args
        self.use_liger_gkd_loss = False
        if getattr(args, 'use_liger_kernel', False):
            if not _liger_kernel_available:
                raise ImportError(
                    'Liger kernel is not installed. Please install liger-kernel by running: pip install liger-kernel')
            assert self.args.sft_alpha == 0, 'SFT loss is not supported with liger loss'

            self.liger_jsd_loss = LigerFusedLinearJSDLoss(
                beta=self.beta,
                ignore_index=-100,
                temperature=self.temperature,
                compiled=False,
            )
            self.use_liger_gkd_loss = True


    @staticmethod
    def generalized_jsd_loss(
        student_logits,
        teacher_logits,
        labels=None,
        beta=0.5,
        temperature=1.0,
        chunk_size=512,
        weights=None,  # per-token weights over valid positions
        weighted_mean=False,  # normalize by sum(weights) instead of token count
    ):
        # Apply temperature scaling
        student_logits = student_logits / temperature
        teacher_logits = teacher_logits / temperature

        # Apply masking if labels provided
        if labels is not None:
            mask = labels != -100
            student_logits = student_logits[mask]
            teacher_logits = teacher_logits[mask]
            num_valid = mask.sum()
            if weights is not None:
                weights = weights[mask]
        else:
            # Flatten to [num_tokens, vocab_size]
            student_logits = student_logits.view(-1, student_logits.size(-1))
            teacher_logits = teacher_logits.view(-1, teacher_logits.size(-1))
            num_valid = student_logits.size(0)
            if weights is not None:
                weights = weights.view(-1)

        if num_valid == 0:
            return student_logits.new_zeros(())

        num_valid_int = num_valid if isinstance(num_valid, int) else num_valid.item()
        total_loss = student_logits.new_zeros(())

        # Precompute beta tensor once if needed
        if beta != 0 and beta != 1:
            beta_t = torch.tensor(beta, dtype=student_logits.dtype, device=student_logits.device)
            log_beta = torch.log(beta_t)
            log_1_minus_beta = torch.log1p(-beta_t)
        else:
            beta_t = log_beta = log_1_minus_beta = None

        # Process in chunks to reduce peak memory
        for start_idx in range(0, num_valid_int, chunk_size):
            end_idx = min(start_idx + chunk_size, num_valid_int)
            s_chunk = student_logits[start_idx:end_idx]
            t_chunk = teacher_logits[start_idx:end_idx]

            s_log_probs = F.log_softmax(s_chunk, dim=-1)
            t_log_probs = F.log_softmax(t_chunk, dim=-1)
            del s_chunk, t_chunk

            if beta == 0:
                jsd_chunk = F.kl_div(s_log_probs, t_log_probs, reduction='none', log_target=True)
            elif beta == 1:
                jsd_chunk = F.kl_div(t_log_probs, s_log_probs, reduction='none', log_target=True)
            else:
                mixture_log_probs = torch.logsumexp(
                    torch.stack([s_log_probs + log_1_minus_beta, t_log_probs + log_beta]),
                    dim=0,
                )

                kl_teacher = F.kl_div(mixture_log_probs, t_log_probs, reduction='none', log_target=True)
                kl_student = F.kl_div(mixture_log_probs, s_log_probs, reduction='none', log_target=True)
                del mixture_log_probs

                jsd_chunk = beta_t * kl_teacher + (1 - beta_t) * kl_student
                del kl_teacher, kl_student

            # Apply per-token weights
            if weights is not None:
                jsd_chunk = jsd_chunk * weights[start_idx:end_idx].unsqueeze(1)

            total_loss = total_loss + jsd_chunk.sum()
            del jsd_chunk, s_log_probs, t_log_probs

        # Weighted-mean branch (opsd_weighted_mean=True).
        # The default total_loss / num_valid is a plain mean of weighted terms, not a weighted
        # average: the denominator is the token count rather than the weight sum, so the overall
        # gain drifts with the mean weight and loss / gradient scales are not comparable across
        # weight modes. Dividing by sum(w) gives a true weighted average that is invariant to
        # scaling all weights by a constant. With all weights 1.0, sum(w) == num_valid, so the
        # result is identical to the default path.
        if weighted_mean and weights is not None:
            return total_loss / weights.sum().clamp_min(1e-8)
        return total_loss / num_valid


    def _prepare_logging(self):
        """Initialize logging components for on-policy rollout tracking."""
        args = self.args
        self.log_completions = args.log_completions
        self.wandb_log_unique_prompts = getattr(args, 'wandb_log_unique_prompts', False)
        self.jsonl_writer = JsonlWriter(os.path.join(self.args.output_dir, 'completions.jsonl'))

        # Initialize logs deque for storing rollout data (aligned with GRPO)
        self._logs = {
            'prompt': deque(),
            'completion': deque(),
        }

    def _apply_chat_template_to_messages_list(self, messages_list: DataType):
        """Convert messages list to prompt text list using template (aligned with GRPO)."""
        prompts_text = []
        for messages in messages_list:
            remove_response(messages)
            template_inputs = TemplateInputs.from_dict({'messages': messages})
            res = self.template.encode(template_inputs)
            prompts_text.append(self.template.safe_decode(res['input_ids']))
        return prompts_text

    def log(self, logs: Dict[str, float], start_time: Optional[float] = None) -> None:
        """Override log method to include completion table logging (aligned with GRPO)."""
        # Call parent log method
        import transformers
        from packaging import version
        if version.parse(transformers.__version__) >= version.parse('4.47.0.dev0'):
            super().log(logs, start_time)
        else:
            super().log(logs)

        # Log completions table if we have data (only for on-policy generations)
        if self.accelerator.is_main_process and self.log_completions and len(self._logs['prompt']) > 0:
            seen_nums = len(self._logs['prompt'])
            table = {
                'step': [str(self.state.global_step)] * seen_nums,
                'prompt': list(self._logs['prompt'])[:seen_nums],
                'completion': list(self._logs['completion'])[:seen_nums],
            }

            # Write to jsonl
            self.jsonl_writer.append(table)

            self._logs['prompt'].clear()
            self._logs['completion'].clear()
            # Log to wandb if enabled
            report_to_wandb = self.args.report_to and 'wandb' in self.args.report_to and wandb.run is not None
            if report_to_wandb:
                wandb_table = table.copy()
                import pandas as pd
                df = pd.DataFrame(wandb_table)
                if self.wandb_log_unique_prompts:
                    df = df.drop_duplicates(subset=['prompt'])
                wandb.log({'completions': wandb.Table(dataframe=df)})

            # Log to swanlab if enabled
            report_to_swanlab = self.args.report_to and 'swanlab' in self.args.report_to and swanlab_has_run()
            if report_to_swanlab:
                headers = list(table.keys())
                rows = []
                for i in range(len(table['step'])):
                    row = [table[header][i] for header in headers]
                    rows.append(row)
                swanlab.log({'completions': swanlab.echarts.Table().add(headers, rows)})
