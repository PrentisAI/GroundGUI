# Copyright (c) ModelScope Contributors. All rights reserved.
"""OPSD+GRPO -- GRPO with zero-variance-group routing to supervised terms.

Method
------
Follows the routing of RSTG (arXiv:2608.00782, "Distill Where You Fail"). A
*negative zero-variance group* (NZG) is a prompt whose G rollouts all score 0, so
``A_GRPO = (r - mean)/std`` is identically zero and the group contributes no
gradient. Routed groups get supervised terms instead::

    J = J_GRPO(theta; A_GRPO) + beta_sft * J_SFT + beta_opd * J_OPSD

where ``J_SFT`` and ``J_OPSD`` are computed on the routed rows only:

* ``J_SFT`` (``--nzg_sft_beta_*``): cross entropy on the ground-truth tool call of
  the routed prompt.
* ``J_OPSD`` (``--nzg_opd_beta_*``, needs ``--teacher_model``): on-policy
  self-distillation, the reverse KL from a privileged teacher that sees the target box (the same masking / hint machinery as
  ``opsd_trainer.py``), on the group's owner rollout (all K with
  ``--nzg_opd_all_rollouts``), weighted per token by ``_nzg_opd_token_weights``.
* ``--nzg_route_all true`` routes every group, not only NZG groups. With
  ``--reward_weights 0`` the GRPO term vanishes and the run is on-policy
  self-distillation (OPSD): the ``opsd`` arm of
  ``Grounding_scripts/opsd+grpo/submit.sh``.

Both betas default to 0, in which case the trainer reduces to plain GRPO.

Isolation
---------
``rlhf_type`` stays ``'grpo'`` -- same reasoning as the TtRT comment in
``swift/trainers/trainer_factory.py``: that is what keeps ``_init_grpo``,
``_check_grpo``, the ``loss_type``/``gradient_accumulation_steps`` defaults,
``padding_side``, the vLLM/padding_free/sequence_parallel whitelists, the template
mode, ``vllm_client`` and ``reward_funcs`` wiring identical to a plain GRPO run.
Selected by ``--use_opsd_grpo true``, dispatched in ``TrainerFactory.get_cls``.
This trainer needs no change to ``grpo_trainer.py`` or ``opsd_trainer.py``: every
override below hooks a method the parent dispatches through ``self``.

Hook points (all verified ``self.``-dispatched in grpo_trainer.py)
------------------------------------------------------------------
* ``_score_completions``     (called at grpo_trainer.py:231) -- flag NZG groups.
* ``_prepare_batch_inputs``  (:238, *before* ``_compute_advantages`` at :240)
                             -- build the GT tool-call SFT batch.
* ``_compute_loss_and_metrics`` (:1089) -- add ``beta * J_SFT``.
* ``_prepare_model_inputs``  (:2603) -- keep our extra keys out of the forward.

On the loss form
----------------
``J_SFT`` is added as a **separate additive term**, not injected as a per-token
advantage into the clipped GRPO objective.  This is a deliberate, *approximately*
equivalent choice, not an identity:

* The parent always snapshots ``old_per_token_logps`` (grpo_trainer.py:933) with
  the weights that are then frozen for the whole accumulation window, so
  ``coef_1 = exp(logp - old_logp)`` is numerically ~1 and the PPO clip/min is
  inert.  ``old_policy()`` is never called in this fork.
* But the GRPO branch also divides logits by ``--temperature`` and can apply
  ``entropy_mask``, ``rollout_is_weights``, ``delta`` clipping and the off-policy
  sequence mask.  A plain cross entropy passes through none of those.  That is
  intended for a supervised term, and it is why this is called "separate", not
  "equivalent".
* ``--loss_type`` is pinned to ``grpo`` in ``__init__``: bnpo and dr_grpo use
  different normalizers (dr_grpo divides by ``max_completion_length``, ~4x for a
  30-token answer), which would silently rescale beta against the GRPO term.

Scale of the SFT term (this is the part that is easy to get wrong)
-----------------------------------------------------------------
transformers divides every micro-batch loss by ``gradient_accumulation_steps``
(``model_accepts_loss_kwargs`` is False for GRPO, and DeepSpeed sets
``scale_wrt_gas=False`` so it is divided exactly once), and DeepSpeed then averages
gradients over the ``W`` data-parallel ranks.  So a term ``X`` added on one rank in
one micro-batch reaches the optimizer with weight ``X / (GAS * W)``.  Attaching one
owner's raw loss would therefore make the effective coefficient ``beta/(GAS*W)``
-- e.g. ``beta/40`` for GAS=5, W=8 -- and would drift with GAS, world
size and how owners happen to pack into chunks.  The term is instead rescaled by
``GAS * W / n_owners_global`` so that ``beta`` multiplies exactly the **mean SFT
loss over the NZG prompts of this optimizer step**, which is what ``beta * J_SFT``
means in Eq. 14 and makes ``beta`` directly comparable to the paper's 5e-3.
Note what that implies: per *prompt*, the supervised term carries weight
``beta / n_nzg`` against GRPO's ``1 / n_prompts``, i.e. beta is amplified by
``n_prompts / n_nzg`` (e.g. 5x when 20% of groups are NZG).  ``nzg/sft_scale``
logs the factor every step so the effective magnitude is never a guess.

Distributed safety -- and why the SFT term is backwarded separately
-------------------------------------------------------------------
Summing the two terms into one loss does NOT work under ZeRO-2: training dies on
the first optimizer step with

    AssertionError: The parameter model.language_model.layers.35.mlp.down_proj.weight
    has already been reduced. Gradient computed twice for this partition.

One backward over ``grpo_loss + beta*sft_loss`` traverses two independent subgraphs
that share every parameter, so ZeRO-2's per-parameter reduction hook
(``stage_1_and_2.py:1091``) fires twice for the same parameter and trips the
single-reduction assert.  Which module object the forward went through is
irrelevant -- this is autograd, not the wrapper -- and no DeepSpeed config avoids
it: the assert sits above the ``contiguous_gradients`` branch.

DeepSpeed's own escape hatch for exactly this shape is three lines above that
assert (``stage_1_and_2.py:1104``): a parameter carrying
``ds_grad_is_ready = False`` makes the hook return before reducing, so the
gradient merely accumulates into ``param.grad``.  Since
``get_gradient_for_reduction`` (:1034) returns ``param.grad``, the later hook
buckets whatever total is sitting there.  So:

1. ``_nzg_backward_sft`` sets ``ds_grad_is_ready = False`` on every trainable
   parameter, runs a plain ``torch.autograd`` backward on the SFT term (NOT
   ``engine.backward`` -- no micro-step bookkeeping, no epilogue), and restores the
   flag.  Hooks suppressed means no collective, so ranks without an owner row
   skipping this entirely is safe.
2. The parent's GRPO loss is returned untouched; transformers divides it by
   ``gradient_accumulation_steps`` and calls ``engine.backward`` once, hooks live,
   and each parameter's bucket receives ``grpo_grad + sft_grad``.

The ORDER is load-bearing and is the reason this runs before ``super()``: when the
hook does fire it rebinds ``param.grad.data`` into the IPG bucket buffer
(:1118-1124), so accumulating into ``param.grad`` *after* a reduction would write
into an already-reduced bucket.  SFT first, GRPO second -- never the reverse.
Because transformers applies the ``/gradient_accumulation_steps`` division only to
the loss it is handed, ``_nzg_backward_sft`` applies that division itself.

Still refused: DeepSpeed ZeRO-3 (its forward all-gathers parameters, so an
asymmetric forward count across ranks is a guaranteed hang) and plain DDP with
W>1 (``DistributedDataParallel.forward`` arms the reducer once per forward).
"""
import re
from copy import deepcopy
from typing import Any, Dict, List

import torch
from accelerate.utils import gather_object

from swift.trainers import per_token_loss_func
from swift.utils import get_logger, to_device
from .grpo_trainer import GRPOTrainer
from .utils import get_even_process_data, mu_schedule_function

logger = get_logger()

# Byte-for-byte the string OPSDTrainer builds for ``--opsd_hint_mode gt``
# (opsd_trainer.py:506).  Do not "improve" it: it must match the tool-call shape
# the policy already emits, or the SFT term teaches a second output format.
GT_TOOL_CALL = ('<tool_call>\n{{"name": "computer_use", "arguments": '
                '{{"action": "left_click", "coordinate": [{cx}, {cy}]}}}}\n</tool_call>')

_FORMAT_PROBE = '"name": "computer_use"'
# Same pattern the offline rollout analysis uses to parse coordinates, so the
# routing log and the offline groups are parsed identically.
_COORD_RE = re.compile(r'"coordinate"\s*:\s*\[\s*(-?\d+)\s*,\s*(-?\d+)')


class OPSDGRPOTrainer(GRPOTrainer):
    """GRPO plus a ground-truth-supervised term on negative zero-variance groups."""

    def __init__(self, *args, **kwargs):
        # Popped before super(): HFGRPOTrainer.__init__ rejects unknown kwargs.
        # Unused at this stage -- kept so an A_OPD branch needs no further change
        # to swift/pipelines/train/rlhf.py.
        self._nzg_teacher_model = kwargs.pop('teacher_model', None)
        self._nzg_teacher_deepspeed_config = kwargs.pop('teacher_deepspeed_config', None)
        super().__init__(*args, **kwargs)
        a = self.args

        self.nzg_reward_name = getattr(a, 'nzg_reward_name', 'GroundAcc')
        self.nzg_sft_beta_peak = float(getattr(a, 'nzg_sft_beta_peak', 0.0))
        self.nzg_sft_beta_valley = float(getattr(a, 'nzg_sft_beta_valley', 0.0))
        self.nzg_sft_warmup_steps = int(getattr(a, 'nzg_sft_warmup_steps', 0))
        self.nzg_sft_decay_steps = int(getattr(a, 'nzg_sft_decay_steps', 0))
        self.nzg_sft_all_rollouts = bool(getattr(a, 'nzg_sft_all_rollouts', False))
        self.nzg_allow_mixed_reward = bool(getattr(a, 'nzg_allow_mixed_reward', False))
        self.nzg_log_routing = bool(getattr(a, 'nzg_log_routing', False))
        self.nzg_opd_beta_peak = float(getattr(a, 'nzg_opd_beta_peak', 0.0))
        self.nzg_opd_beta_valley = float(getattr(a, 'nzg_opd_beta_valley', 0.0))
        self.nzg_opd_warmup_steps = int(getattr(a, 'nzg_opd_warmup_steps', 0))
        self.nzg_opd_decay_steps = int(getattr(a, 'nzg_opd_decay_steps', 0))
        self.nzg_opd_all_rollouts = bool(getattr(a, 'nzg_opd_all_rollouts', False))
        self.nzg_route_all = bool(getattr(a, 'nzg_route_all', False))
        self.teacher_model = None
        self._nzg_opd_total = 0

        self._check_preconditions()
        if self.nzg_opd_beta_peak > 0:
            self._check_hint_parity()
            self._prepare_opd_teacher()

        self._nzg_frac_groups = None   # share of all-miss groups, last rollout batch (instantaneous)
        self._nzg_fmt_match = None     # format drift AMONG owner completions that emitted a coordinate
        self._nzg_unparsed = None      # share of owner completions with no coordinate at all (an A1 failure,
                                       # which is a legitimate reason for a group to be all-miss)
        self._nzg_gen_calls = 0        # _score_completions calls since the last optimizer step
        self._nzg_gen_step = -1
        self._nzg_owner_total = 0      # owner count of the MOST RECENT rollout batch; only ever
                                       # read inside the same _generate_and_score_completions call
                                       # (copied into the batch there) -- never from the loss, see
                                       # the note on _nzg_sft_scale
        logger.info(f'[opsd+grpo] NZG routing on reward column {self.nzg_reward_name!r}; '
                    f'SFT beta peak={self.nzg_sft_beta_peak} valley={self.nzg_sft_beta_valley} '
                    f'warmup={self.nzg_sft_warmup_steps} decay={self.nzg_sft_decay_steps}; '
                    + ('ALL K rollouts supervised' if self.nzg_sft_all_rollouts else
                       'one SFT sequence per NZG prompt'))

    def _check_preconditions(self):
        """Every one of these, if violated, changes the method silently or hangs."""
        a = self.args

        # --- the branch is only well defined on a BINARY column ---------------
        if self.nzg_reward_name not in self.reward_func_names:
            raise ValueError(
                f'--use_opsd_grpo needs a binary in-box reward column named {self.nzg_reward_name!r} to define '
                f'"all rollouts fail"; got reward_func_names={self.reward_func_names}. Pass --reward_funcs '
                'ground-acc [...] or set --nzg_reward_name.')
        col = self.reward_func_names.index(self.nzg_reward_name)
        own_w = float(self.reward_weights[col])
        if own_w < 0:
            raise ValueError(
                f'--use_opsd_grpo needs a POSITIVE weight on {self.nzg_reward_name!r} (got {own_w}). At 0 the GRPO '
                'advantage is identically zero while the NZG branch still fires, and at a negative weight the '
                'advantage is inverted -- either way the run is not the routed objective.')
        if own_w == 0:
            logger.warning('weight 0 on %s: the GRPO advantage is identically zero, so this run is PURE DISTILLATION '
                           '(teacher KL only, no policy-gradient term). Teacher routing: %s.', self.nzg_reward_name,
                           'ALL groups (nzg_route_all)' if self.nzg_route_all else 'all-miss groups only')
        other = [(n, float(w)) for i, (n, w) in enumerate(zip(self.reward_func_names, self.reward_weights))
                 if i != col and float(w) != 0.0]
        if other and not self.nzg_allow_mixed_reward:
            raise ValueError(
                f'--use_opsd_grpo with a second non-zero-weight reward {other} is NOT the routed objective. A group '
                f'that misses on {self.nzg_reward_name} can still have non-zero variance in a continuous reward '
                '(g2acc), so the parent hands it a non-zero A_GRPO and you get GRPO(A_GRPO) + beta*SFT, not the '
                'RSTG branch. Use --reward_funcs ground-acc --reward_weights 1.0, or pass '
                '--nzg_allow_mixed_reward true if that hybrid is what you actually want.')

        # --- keep the two loss terms on a known relative scale ---------------
        if a.gradient_accumulation_steps != a.steps_per_generation:
            raise ValueError(
                f'--use_opsd_grpo requires gradient_accumulation_steps == steps_per_generation (got '
                f'{a.gradient_accumulation_steps} vs {a.steps_per_generation}) so that one rollout batch is exactly '
                'one optimizer step; the SFT rescaling counts owners per rollout batch.')
        if a.num_iterations != 1:
            raise ValueError(f'--use_opsd_grpo requires num_iterations == 1 (got {a.num_iterations}); with more, '
                             'coef_1 != 1, the PPO clip goes live and the two terms stop being comparable.')
        if getattr(a, 'loss_type', 'grpo') != 'grpo':
            raise ValueError(f'--use_opsd_grpo pins --loss_type grpo (got {a.loss_type!r}); bnpo and dr_grpo use '
                             'different normalizers, which silently rescales the SFT beta against the GRPO term.')
        if getattr(self, 'use_liger_loss', False):
            raise ValueError('--use_opsd_grpo is incompatible with --use_liger_kernel true: compute_loss routes to '
                             'compute_liger_loss (grpo_trainer.py:1063) and never reaches '
                             '_compute_loss_and_metrics, which would DELETE the SFT term without a warning.')

        # --- these change what the branch even selects -----------------------
        if getattr(a, 'dynamic_sample', False):
            raise ValueError('--use_opsd_grpo is incompatible with --dynamic_sample true: dynamic sampling is the '
                             'DAPO-style *discard* of zero-variance groups, so the NZG branch would never fire. '
                             'They are two mutually exclusive answers to the same problem.')
        if getattr(self, 'dynamic_num_samples', False) or getattr(self, 'enable_server_multi_turn', False):
            # dynamic_num_samples is False at __init__ and only flips at runtime
            # (rollout_mixin.py:1016) when server multi-turn returns a different
            # number of rollouts than requested, so the flag alone is not enough --
            # refuse the mode that can set it, and re-check inside _score_completions.
            raise ValueError('--use_opsd_grpo does not support dynamic_num_samples / server multi-turn: a variable '
                             'rollout count per prompt breaks the consecutive-K group reshape, the K-block owner '
                             'election and the gather/get_even_process_data round trip.')
        autotp = getattr(a, 'deepspeed_autotp_size', None) or 1
        ds_cfg = getattr(a, 'deepspeed', None)
        if isinstance(ds_cfg, dict):
            autotp = max(int(autotp), int(ds_cfg.get('tensor_parallel', {}).get('autotp_size', 1) or 1))
        if int(autotp) > 1:
            raise ValueError(f'--use_opsd_grpo is incompatible with DeepSpeed AutoTP (autotp_size={autotp}): the '
                             'extra SFT forward runs on some ranks only and a tensor-parallel forward issues '
                             'collectives, so the ranks would disagree and hang. It also makes num_processes stop '
                             'being the data-parallel width the SFT rescaling assumes.')
        if self.template.padding_free or getattr(self.template, 'sequence_parallel_size', 1) > 1:
            raise ValueError('--use_opsd_grpo has not been validated with padding_free/packing or '
                             'sequence_parallel; the SFT batch is encoded and collated separately and the rmpad '
                             'round trip is untested.')

        # --- asymmetric extra forward: must perform no collectives -----------
        world = self.accelerator.num_processes
        if world > 1:
            if not self.is_deepspeed_enabled:
                raise ValueError('--use_opsd_grpo with world_size > 1 requires DeepSpeed. Under plain DDP the extra '
                                 'SFT forward arms the reducer a second time on some ranks only, which deadlocks or '
                                 'trips "marked ready twice".')
            stage = None
            plugin = getattr(self.accelerator.state, 'deepspeed_plugin', None)
            if plugin is not None:
                stage = plugin.deepspeed_config.get('zero_optimization', {}).get('stage')
            if stage == 3:
                raise ValueError('--use_opsd_grpo is incompatible with DeepSpeed ZeRO-3 for the policy: a ZeRO-3 '
                                 'forward all-gathers parameters, and the SFT forward runs on some ranks only, so '
                                 'the ranks would disagree on the number of collectives and hang. Use zero2.')

        if self.nzg_opd_beta_peak > 0:
            if self._nzg_teacher_model is None:
                raise ValueError('--nzg_opd_beta_peak > 0 needs a teacher: pass --teacher_model (and '
                                 '--teacher_deepspeed zero2, NOT zero3 -- see the stage-3 refusal below). Without '
                                 'a teacher there is no privileged distribution to distil from and the branch '
                                 'would be a silent no-op. Note prepare_deepspeed rewrites any non-3 stage to 0 '
                                 '(utils.py:339), so the teacher is REPLICATED: budget ~16.3 GiB per GPU.')
            cfg = self._nzg_teacher_deepspeed_config or {}
            if cfg.get('zero_optimization', {}).get('stage') == 3:
                raise ValueError(
                    'the OPD branch refuses a ZeRO-3 TEACHER: its forward all-gathers parameters, and the teacher '
                    'forward runs only on ranks that hold an NZG row, so the ranks would issue different numbers of '
                    'collectives and hang. Use --teacher_deepspeed zero2 (or none) until every rank performs a '
                    'teacher forward unconditionally.')
            hm = getattr(a, 'opsd_hint_mode', 'hint')
            if hm not in ('none', 'hint', 'hint_minimal', 'spotlight'):
                raise ValueError(f"the OPD branch supports --opsd_hint_mode in {{none,hint,hint_minimal,spotlight}}; "
                                 f"got {hm!r}. 'gt' puts the answer coordinate in the teacher PROMPT, which is a "
                                 'different experiment (and makes the teacher trivially right).')
        if self.nzg_sft_beta_peak <= 0 and self.nzg_opd_beta_peak <= 0:
            logger.warning('--use_opsd_grpo is on but both nzg_sft_beta_peak and nzg_opd_beta_peak <= 0: this run '
                           'is identical to plain GRPO except for the NZG bookkeeping metrics.')

    def _prepare_opd_teacher(self):
        """Bring up the privileged teacher, mirroring OPSDTrainer.__init__.

        ``prepare_deepspeed`` is imported LAZILY here, and ``opsd_trainer`` is never
        imported at all unless the branch is on: importing that module also runs its
        module-level trl MRO patch (``del _gkd_cls.__init__``), and a run with OPD off
        must not pay that side effect.
        """
        from .utils import prepare_deepspeed
        cfg = self._nzg_teacher_deepspeed_config
        if self.is_deepspeed_enabled:
            self.is_teacher_ds3 = (cfg or {}).get('zero_optimization', {}).get('stage') == 3
            self.teacher_model = (prepare_deepspeed(self._nzg_teacher_model, self.accelerator,
                                                    deepspeed_config=cfg, training_args=self.args)
                                  if cfg is not None else
                                  prepare_deepspeed(self._nzg_teacher_model, self.accelerator))
        else:
            self.teacher_model = self.accelerator.prepare_model(self._nzg_teacher_model, evaluation_mode=True)
        self.teacher_model.eval()
        self.teacher_model.requires_grad_(False)
        logger.info(f'[opsd+grpo] OPD branch ON: teacher up (deepspeed={self.is_deepspeed_enabled}, '
                    f'teacher_ds3={getattr(self, "is_teacher_ds3", None)}), '
                    f'hint_mode={getattr(self.args, "opsd_hint_mode", "hint")!r} '
                    f'mask_mode={getattr(self.args, "opsd_mask_mode", "zoom_in")!r} '
                    f'beta peak={self.nzg_opd_beta_peak} valley={self.nzg_opd_beta_valley}; '
                    + ('ALL K rollouts distilled' if self.nzg_opd_all_rollouts else 'owner row only'))

    # ------------------------------------------------------------------ hook 1
    def _score_completions(self, inputs: List[Dict[str, Any]]) -> torch.Tensor:
        """Flag negative zero-variance groups and elect one SFT owner per group.

        Runs on the *gathered* reward tensor, the same way ``_compute_advantages``
        does (``rewards.view(-1, num_generations)``), then hands the per-row flags
        back through ``get_even_process_data`` -- the identical round trip the
        parent uses for ``advantages`` at grpo_trainer.py:242-245.

        The owner is the K-block's offset 0, computed from the *global row index*.
        It is deliberately NOT elected by ``prompt_id``: that id hashes
        ``messages`` only (rollout_mixin.py:975) and carries no image, and grounding
        corpora contain rows that share their messages with another row while showing
        a different screenshot (generic instructions such as "Type here to search"
        recur across many screenshots), so a prompt_id election would silently leave
        colliding NZG groups ownerless.
        """
        total_rewards_per_func = super()._score_completions(inputs)
        for inp in inputs:
            inp['_nzg'] = False
            inp['_nzg_sft_owner'] = False
        self._nzg_owner_total = 0
        self._nzg_fmt_match = None
        self._nzg_unparsed = None
        self._nzg_frac_groups = None
        if not self.model.training:
            return total_rewards_per_func
        if getattr(self, 'dynamic_num_samples', False):
            raise RuntimeError('dynamic_num_samples flipped to True at runtime (server multi-turn returned a '
                               'different rollout count); the K-block owner election is no longer valid.')

        k = self.num_generations
        col = self.reward_func_names.index(self.nzg_reward_name)
        hits = total_rewards_per_func[:, col]
        if hits.numel() % k != 0:
            raise RuntimeError(f'gathered rollout count {hits.numel()} is not a multiple of num_generations={k}; '
                               'cannot form groups.')
        # A NaN means the reward func declined to score that row. Treat it as "not a
        # miss" so an unscored row can never manufacture an NZG group.
        hits = torch.nan_to_num(hits, nan=1.0)
        group_all_miss = (hits.view(-1, k) == 0).all(dim=1)            # [n_groups]
        if self.nzg_route_all:
            # pure-OPSD ablation: every group goes to the teacher regardless of hits.
            group_all_miss = torch.ones_like(group_all_miss)
        n_groups = int(group_all_miss.numel())
        row_nzg = group_all_miss.repeat_interleave(k).tolist()         # [n_rows]

        # Exactly one supervised sequence per NZG group unless asked otherwise:
        # without this the same GT string is fitted once per rollout, i.e. K times.
        row_owner = [False] * len(row_nzg)
        for g in range(n_groups):
            if not bool(group_all_miss[g]):
                continue
            if self.nzg_sft_all_rollouts:
                for j in range(k):
                    row_owner[g * k + j] = True
            else:
                row_owner[g * k] = True
        self._nzg_owner_total = sum(row_owner)

        # One canonical owner per NZG group, ALWAYS (K-block offset 0), regardless of
        # nzg_sft_all_rollouts. The OPD branch selects on this.
        row_group_owner = [False] * len(row_nzg)
        for g in range(n_groups):
            if bool(group_all_miss[g]):
                row_group_owner[g * k] = True
        self._nzg_opd_total = (sum(row_nzg) if self.nzg_opd_all_rollouts else sum(row_group_owner))
        local_nzg = get_even_process_data(self, row_nzg)
        local_owner = get_even_process_data(self, row_owner)
        local_group_owner = get_even_process_data(self, row_group_owner)
        assert len(local_nzg) == len(inputs) == len(local_owner), \
            f'{len(local_nzg)} / {len(inputs)} / {len(local_owner)}'
        for inp, is_nzg, own in zip(inputs, local_nzg, local_owner):
            inp['_nzg'] = bool(is_nzg)
            inp['_nzg_sft_owner'] = bool(own)
        for inp, is_nzg, own in zip(inputs, local_nzg, local_group_owner):
            inp['_nzg_group_owner'] = bool(own)

        self._nzg_frac_groups = (int(group_all_miss.sum().item()) / n_groups) if n_groups else None
        # How many rollout batches feed one optimizer step? It need not be 1, and it
        # multiplies the SFT term's per-step weight, so measure it instead of assuming.
        gs = int(self.state.global_step)
        if gs != self._nzg_gen_step:
            self._nzg_gen_step, self._nzg_gen_calls = gs, 0
        self._nzg_gen_calls += 1
        if self.nzg_log_routing:
            self._write_routing_log(inputs, hits, row_nzg, row_owner, k)
        return total_rewards_per_func

    def _write_routing_log(self, inputs, hits, row_nzg, row_owner, k):
        """One JSON line per group, appended to ``<output_dir>/nzg_routing.jsonl``.

        This is the only record of WHICH prompts the branch fired on, and it cannot
        be reconstructed afterwards: the rollouts are not checkpointed.  It is what
        makes the post-hoc question answerable -- of the eval items that got fixed,
        were they the GROSS ones the NZG route is supposed to reach?  One record per
        group, so per-group analyses can be run on it after training.
        """
        import json as _json
        import os as _os
        local = []
        for inp in inputs:
            rec = {'pred': None, 'gt_px': None, 'wh': None, 'ui': None, 'plat': None}
            try:
                sol = inp['solution']
                if isinstance(sol, str):
                    sol = _json.loads(sol.replace("'", '"'))
                rec['gt_px'] = list(sol['arguments']['coordinate'])
                extra = inp.get('additional_paras', {})
                if isinstance(extra, str):
                    extra = _json.loads(extra)
                rec['wh'] = list(extra['image_size'])
                rec['ui'] = extra.get('ui_type')
                rec['plat'] = extra.get('platform')
            except Exception:
                pass
            text = self._completion_text(inp)
            if text:
                m = _COORD_RE.search(text)
                if m:
                    rec['pred'] = [int(m.group(1)), int(m.group(2))]
            local.append(rec)
        allrec = gather_object(local)
        if not self.accelerator.is_main_process:
            return
        if not (len(allrec) == len(row_nzg) == hits.numel()):
            logger.warning(f'[opsd+grpo] routing log skipped: gather mismatch '
                           f'{len(allrec)}/{len(row_nzg)}/{hits.numel()}')
            return
        hl = hits.tolist()
        path = _os.path.join(self.args.output_dir, 'nzg_routing.jsonl')
        _os.makedirs(_os.path.dirname(path) or '.', exist_ok=True)
        with open(path, 'a', encoding='utf-8') as f:
            for g in range(len(allrec) // k):
                sl = slice(g * k, (g + 1) * k)
                base = allrec[g * k]
                gt_px, wh = base['gt_px'], base['wh']
                gt_norm = None
                if gt_px and wh:
                    gt_norm = [int(gt_px[0] / wh[0] * 1000), int(gt_px[1] / wh[1] * 1000),
                               int(gt_px[2] / wh[0] * 1000), int(gt_px[3] / wh[1] * 1000)]
                f.write(_json.dumps({
                    'step': int(self.state.global_step),
                    'group': g,
                    'gt': gt_norm,
                    'gt_px': gt_px,
                    'wh': wh,
                    'ui': base['ui'],
                    'plat': base['plat'],
                    'pts': [r['pred'] for r in allrec[sl]],
                    'ga': [float(x) for x in hl[sl]],
                    'nzg': bool(row_nzg[g * k]),
                    'owner': [bool(x) for x in row_owner[sl]],
                }, ensure_ascii=False) + '\n')

    # ------------------------------------------------------------------ hook 2
    def _prepare_batch_inputs(self, inputs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Attach a GT tool-call SFT sub-batch to each gradient-accumulation chunk.

        ``split_by_mini_batches`` is deterministic and side-effect free (it slices
        the list and calls ``to_device``), so calling it again here reproduces the
        chunking the parent used -- the parent re-derives it the same way at
        grpo_trainer.py:250.  ``to_device`` rebuilds dicts, but the owner flag was
        set on the originals before this runs, so the copies carry it.
        """
        batches = super()._prepare_batch_inputs(inputs)
        for batch in batches:
            batch['nzg_sft_inputs'] = None
            batch['nzg_rows'] = 0
            batch['nzg_opd_inputs'] = None
            batch['nzg_opd_rows'] = 0
            # Snapshot the owner count of THIS rollout batch. It must travel with the
            # batch: the trainer can score more than one rollout batch per optimizer
            # step (visible in the routing log), so a micro-batch served out of
            # _buffered_inputs can be consumed AFTER a later _score_completions has
            # already overwritten the instance field -- which would silently scale
            # the SFT term by another batch's owner count.
            batch['nzg_owner_total'] = int(self._nzg_owner_total)
            # MEASURED micro-batches this rollout batch feeds. Needed because
            # gradient_accumulation_steps == steps_per_generation does NOT imply one
            # rollout batch per optimizer step: a run can still do B > 1 generations
            # per step (the routing log then has every (step, group) pair B times, same
            # GT box, different sampled points). Scaling by GAS instead of m would
            # inflate the effective beta by exactly that B.
            batch['nzg_micro_per_batch'] = len(batches)
        # NOT `sft_peak <= 0` alone: that would return before the OPD pairs are built,
        # so an OPD-only ablation would load the teacher, log "OPD branch ON" and then
        # produce no OPD gradient at all -- a silent no-op.
        if not self.model.training or (self.nzg_sft_beta_peak <= 0 and self.nzg_opd_beta_peak <= 0):
            return batches

        chunks = self.split_by_mini_batches(inputs)
        if len(chunks) != len(batches):
            raise RuntimeError(f'chunk/batch mismatch: {len(chunks)} vs {len(batches)}')
        if self.nzg_opd_beta_peak > 0:
            for chunk, batch in zip(chunks, batches):
                # `_nzg_group_owner` is the canonical one-per-group marker, independent
                # of nzg_sft_all_rollouts -- reusing `_nzg_sft_owner` would couple the
                # two flags (turning SFT's all-rollouts on would silently turn OPD's on
                # too).
                rows = [d for d in chunk
                        if (d.get('_nzg') if self.nzg_opd_all_rollouts else d.get('_nzg_group_owner'))]
                batch['nzg_opd_rows'] = len(rows)
                # OPD gets its OWN denominator: with all_rollouts it selects K times as
                # many rows as the SFT owner count, so sharing the SFT scale would
                # over-weight the term by K.
                batch['nzg_opd_total'] = int(self._nzg_opd_total)
                batch['nzg_opd_inputs'] = self._encode_opd_pair(rows) if rows else None
        fmt_ok = fmt_n = unparsed = seen = 0
        # Gated on the SFT beta: an OPD-only ablation must not pay for -- or be aborted
        # by -- GT-target encoding it will never use.
        for chunk, batch in (zip(chunks, batches) if self.nzg_sft_beta_peak > 0 else ()):
            owners = [d for d in chunk if d.get('_nzg_sft_owner')]
            batch['nzg_rows'] = len(owners)
            if not owners:
                continue
            batch['nzg_sft_inputs'] = self._encode_gt_sft(owners)
            for d in owners:
                text = self._completion_text(d)
                seen += 1
                if text is None:
                    continue
                # Two different failures, kept apart on purpose. A rollout with NO
                # parseable coordinate scores GroundAcc=0 by construction, so a group
                # of format failures IS an all-miss group -- and SFT on the GT tool
                # call is precisely the right fix for it, not a warning sign. Mixing
                # it into the format-drift probe makes the probe fire when nothing is
                # wrong, because many rollouts inside all-miss groups simply have no
                # coordinate at all.
                if _COORD_RE.search(text) is None:
                    unparsed += 1
                    continue
                fmt_n += 1
                fmt_ok += int(_FORMAT_PROBE in text)
        if seen:
            self._nzg_unparsed = unparsed / seen
        if fmt_n:
            # Drift among rollouts that DID emit a coordinate: this is the one that
            # says "the policy no longer writes the shape the SFT target assumes".
            self._nzg_fmt_match = fmt_ok / fmt_n
            if self._nzg_fmt_match < 0.9:
                logger.warning(f'[opsd+grpo] only {self._nzg_fmt_match:.1%} of supervised NZG completions that DO '
                               f'emit a coordinate still contain {_FORMAT_PROBE!r}: the policy has drifted off the '
                               'tool-call shape the SFT target assumes, so this term is teaching a second format.')
        return batches

    # ------------------------------------------------------------------ hook 3
    def _compute_loss_and_metrics(self, model, inputs):
        mode = 'train' if self.model.training else 'eval'
        beta = self._nzg_beta()
        sft_inputs = inputs.get('nzg_sft_inputs')
        scale = self._nzg_sft_scale(inputs)
        sft_loss_val = 0.0
        # The SFT term is backwarded HERE, before the parent's forward, and is NOT
        # added to the returned loss. See _nzg_backward_sft for why.
        probe = (0.0, 0.0)
        rows = float(inputs.get('nzg_rows', 0))
        opd_beta = self._nzg_beta(opd=True)
        opd_inputs = inputs.get('nzg_opd_inputs')
        opd_loss_val = 0.0
        opd_probe = (0.0, 0.0)
        opd_scale = self._nzg_scale_from(inputs, 'nzg_opd_total')
        if mode == 'train' and opd_beta > 0 and opd_inputs is not None and opd_scale > 0:
            # OPD before SFT before GRPO. Same suppressed-hook mechanism, same ordering
            # rule: every extra term must land in param.grad BEFORE the reduction hook
            # fires on the GRPO backward.
            opd_loss_val, opd_probe = self._nzg_backward_opd(opd_inputs, opd_beta * opd_scale)
        if mode == 'train' and beta > 0 and sft_inputs is not None and scale > 0:
            sft_loss_val, probe = self._nzg_backward_sft(sft_inputs, beta * scale)
        # Cross-rank reduce so the diagnostics do not depend on WHICH rank happened to
        # own the SFT row. With only a few owners per optimizer step spread over W
        # ranks, rank 0 often holds none, so rank-local numbers would log 0 while
        # other ranks are training -- the gradient probe would be unobservable
        # exactly when it matters. Called UNCONDITIONALLY by every rank
        # (zeros when idle), so it stays a symmetric collective.
        if mode == 'train' and self.nzg_log_routing:
            sft_loss_val, rows, probe = self._nzg_reduce_diag(sft_loss_val, rows, probe)
        loss, metrics = super()._compute_loss_and_metrics(model, inputs)
        m = self._metrics[mode]
        # These are rank-local by construction (only the owner's rank pays the SFT
        # forward), so read them as "somebody's rank did this", not as a mean.
        m['nzg/beta'].append(beta)
        m['nzg/sft_scale'].append(scale)
        m['nzg/sft_rows'].append(rows)
        m['nzg/sft_loss'].append(sft_loss_val)
        if self.nzg_opd_beta_peak > 0:
            opd_rows = float(inputs.get('nzg_opd_rows', 0))
            if mode == 'train' and self.nzg_log_routing:
                # Same reason as the SFT diagnostics: only the rank holding an NZG row
                # runs the OPD backward, so rank-local numbers log 0 while other ranks
                # train. Called by EVERY rank (zeros when idle) to stay symmetric.
                opd_loss_val, opd_rows, opd_probe = self._nzg_reduce_diag(opd_loss_val, opd_rows, opd_probe)
            m['nzg/opd_beta'].append(opd_beta)
            m['nzg/opd_scale'].append(opd_scale)
            m['nzg/opd_rows'].append(opd_rows)
            # SUM of per-row token-mean KLs (same unit as nzg/sft_loss), NOT a mean:
            # divide by nzg/opd_rows to read a per-rollout KL off the curve.
            m['nzg/opd_loss'].append(opd_loss_val)
            if mode == 'train' and self.nzg_log_routing:
                m['nzg/opd_gradnorm_before'].append(opd_probe[0])
                m['nzg/opd_gradnorm_after'].append(opd_probe[1])
        if mode == 'train' and self.nzg_log_routing:
            m['nzg/gradnorm_before'].append(probe[0])
            m['nzg/gradnorm_after'].append(probe[1])
        m['nzg/owner_total'].append(float(inputs.get('nzg_owner_total', 0) or 0))
        m['nzg/gen_per_step'].append(float(self._nzg_gen_calls))
        m['nzg/micro_per_batch'].append(float(inputs.get('nzg_micro_per_batch', 0) or 0))
        if self._nzg_frac_groups is not None:
            m['nzg/frac_groups'].append(self._nzg_frac_groups)
        if self._nzg_fmt_match is not None:
            m['nzg/format_match'].append(self._nzg_fmt_match)
        if self._nzg_unparsed is not None:
            m['nzg/unparsed'].append(self._nzg_unparsed)
        return loss, metrics

    # ------------------------------------------------------------------ hook 4
    def _prepare_model_inputs(self, inputs) -> Dict[str, Any]:
        """Strip our extra keys before ``model(**model_inputs)``.

        The parent filters by a denylist, so any key we add to the batch would
        otherwise be forwarded into the model and raise.
        """
        return {k: v for k, v in super()._prepare_model_inputs(inputs).items() if not k.startswith('nzg_')}

    # ---------------------------------------------------------------- internals
    def _nzg_beta(self, opd: bool = False) -> float:
        peak, valley, warm, dec = ((self.nzg_opd_beta_peak, self.nzg_opd_beta_valley,
                                    self.nzg_opd_warmup_steps, self.nzg_opd_decay_steps) if opd else
                                   (self.nzg_sft_beta_peak, self.nzg_sft_beta_valley,
                                    self.nzg_sft_warmup_steps, self.nzg_sft_decay_steps))
        if peak <= 0:
            return 0.0
        if warm <= 0 and dec <= 0:
            return peak
        return mu_schedule_function(self.state.global_step, max(warm, 0), max(dec, 0), peak, valley)

    def _nzg_scale_from(self, inputs, key: str) -> float:
        """``m * W / n`` where n is the count under ``key`` for THIS rollout batch."""
        n = int(inputs.get(key, 0) or 0)
        if n <= 0:
            return 0.0
        m = int(inputs.get('nzg_micro_per_batch', 0) or 0) or self.args.gradient_accumulation_steps
        return (m * self.accelerator.num_processes) / float(n)

    def _nzg_sft_scale(self, inputs) -> float:
        """``m * W / n_owners`` for the rollout batch THIS micro-batch came from.

        Both counts are read off the batch, never off ``self``: see the note in
        ``_prepare_batch_inputs``.  ``m`` is the measured micro-batches per rollout
        batch, NOT ``gradient_accumulation_steps`` -- with ``B = GAS/m`` rollout
        batches per optimizer step, each contributing ``beta * mean_b``, using GAS
        would make the effective coefficient ``beta * B``.
        """
        n = int(inputs.get('nzg_owner_total', 0) or 0)
        if n <= 0:
            return 0.0
        m = int(inputs.get('nzg_micro_per_batch', 0) or 0) or self.args.gradient_accumulation_steps
        world = self.accelerator.num_processes
        return (m * world) / float(n)

    def _completion_text(self, data: Dict[str, Any]):
        """The rollout's completion as text.

        The vLLM path stores it as token ids, not a string (the parent decodes it
        the same way when logging, grpo_trainer.py:262-270), so a naive
        ``isinstance(content, str)`` probe would silently never fire.
        """
        content = data['messages'][-1].get('content')
        if isinstance(content, str):
            return content
        if isinstance(content, dict):
            content = content.get('token_ids')
        if isinstance(content, list):
            # An EMPTY id list is a real, decodable completion (it just contains no
            # tool call). Returning None would drop it from the denominator and hide
            # exactly the degeneration the probe exists to catch.
            if all(isinstance(t, int) for t in content):
                return self.processing_class.decode(content) if content else ''
        return None

    @staticmethod
    def _gt_center_norm1000(data: Dict[str, Any]):
        """GT box centre in the 0-1000 normalised space Qwen3-VL emits.

        Mirrors ``bboxreal2norm`` then the centre, exactly as opsd_trainer.py:511
        and swift/custom_utils/ground_func.py do.  For qwen3_vl the model's
        coordinate is ALREADY normalised -- emitting pixels here would be a silent
        off-by-a-lot.
        """
        import json as _json
        sol = data['solution']
        if isinstance(sol, str):
            sol = _json.loads(sol.replace("'", '"'))
        box = sol['arguments']['coordinate']
        if len(box) != 4:
            raise ValueError(f'expected a 4-value GT bbox, got {box!r}')
        extra = data.get('additional_paras', {})
        if isinstance(extra, str):
            extra = _json.loads(extra)
        w, h = extra['image_size']
        nx1, nx2 = int(box[0] / w * 1000), int(box[2] / w * 1000)
        ny1, ny2 = int(box[1] / h * 1000), int(box[3] / h * 1000)
        return (nx1 + nx2) // 2, (ny1 + ny2) // 2

    def _encode_gt_sft(self, rows: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        """Encode ``prompt + GT tool call`` for the elected NZG rows.

        The prompt is the student's own -- same image, no spotlight, no green box,
        no hint.  So this term opens no privileged-marker leak channel at all
        (contrast the privileged teacher prompt, whose hint TEXT can leak marker
        phrases such as "green rectangle" into the student's outputs).
        """
        datas = []
        for d in rows:
            dd = dict(d)
            dd['messages'] = deepcopy(d['messages'][:-1])
            cx, cy = self._gt_center_norm1000(d)
            dd['messages'].append({'role': 'assistant', 'content': GT_TOOL_CALL.format(cx=cx, cy=cy)})
            if 'images' in dd:
                dd['images'] = deepcopy(d['images'])
            # Otherwise the rollout's token ids would win over our text.
            dd.pop('response_token_ids', None)
            dd.pop('response_loss_mask', None)
            datas.append(dd)

        with self._template_context(self.template):
            encoded = [self.template.encode(x, return_length=True) for x in datas]
            batch = to_device(self.template.data_collator(encoded), self.model.device)
        if 'labels' not in batch:
            raise RuntimeError('the SFT batch has no labels; template.encode ran in a prompt-only mode')
        return batch

    def _nzg_backward_sft(self, sft_inputs: Dict[str, torch.Tensor], coef: float) -> float:
        """Backward the SFT term on its own, with ZeRO-2's reduction hooks suppressed.

        Returns ``(unscaled SFT loss, (||g|| before, ||g|| after))``.  See the module docstring's
        distributed-safety section -- the suppression, the plain autograd backward
        and the SFT-before-GRPO ordering are all load-bearing.
        """
        gas = getattr(self, 'current_gradient_accumulation_steps', None) or self.args.gradient_accumulation_steps
        params = [p for p in self.model.parameters() if p.requires_grad]
        sft_loss = self._nzg_sft_loss(self.model, sft_inputs)
        term = (coef / float(gas)) * sft_loss
        # Direct proof that the suppressed-hook backward actually deposits gradient:
        # if before == after, the term contributed nothing and the whole branch is a
        # no-op that no aggregate metric would reveal. Only under the diagnostics
        # flag -- it is a reduction over every parameter.
        want = self.nzg_log_routing
        before = self._nzg_grad_norm(params) if want else 0.0
        prior = [getattr(p, 'ds_grad_is_ready', True) for p in params]
        for p in params:
            p.ds_grad_is_ready = False
        try:
            term.backward()
        finally:
            # restore the PRIOR value, not a blanket True: a nested/other suppression
            # would otherwise be silently cleared.
            for p, v in zip(params, prior):
                p.ds_grad_is_ready = v
        after = self._nzg_grad_norm(params) if want else 0.0
        # UNCHANGED, not "did not increase". A true no-op leaves param.grad byte-identical,
        # so the norm is exactly equal. A DECREASE means this term's gradient partially
        # cancelled what was already accumulated -- with A_OPD enabled the OPD backward
        # runs first, so `before` is non-zero and a decrease is both expected and
        # informative (the two terms disagree on this step). A `<=` test would report
        # those decreases as silent no-ops, which is exactly backwards.
        if want and after == before:
            logger.warning(f'[opsd+grpo] the SFT backward deposited NO gradient '
                           f'(||g|| unchanged at {before:.4e}): the branch is a silent no-op.')
        elif want and after < before:
            logger.info(f'[opsd+grpo] SFT gradient partially CANCELLED the accumulated '
                        f'gradient (||g|| {before:.4e} -> {after:.4e}); the terms disagree on '
                        'this step. Not a no-op.')
        return float(sft_loss.detach().item()), (before, after)

    # ------------------------------------------------------------ A_OPD branch
    # Verbatim copies of the hint strings in opsd_trainer.py. Duplicated rather than
    # imported; _check_hint_parity() re-reads that file at startup and refuses to run if
    # either literal has drifted, turning a silent divergence into a loud one. Keep the
    # hint_minimal wording verbatim.
    _HINT_FULL = " Hint: The answer is located within the green rectangle."
    _HINT_MINIMAL = (" Hint: Never mention the green rectangle."
                     " The answer is the element inside it.")

    def _check_hint_parity(self):
        """Refuse to run if our copy of the hint text drifted from opsd_trainer.py."""
        import os as _os
        src = _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), 'opsd_trainer.py')
        try:
            with open(src, encoding='utf-8') as f:
                text = f.read()
        except OSError as e:
            logger.warning(f'[opsd+grpo] could not read {src} for hint parity ({e}); check skipped')
            return
        # Checked FRAGMENT by fragment, not as one joined string: opsd_trainer.py
        # writes hint_minimal as a two-line implicit concatenation, so the joined text
        # never appears as a contiguous substring of that file. Searching for the join
        # is a guaranteed false positive.
        frags = [('hint', self._HINT_FULL),
                 ('hint_minimal', ' Hint: Never mention the green rectangle.'),
                 ('hint_minimal', ' The answer is the element inside it.')]
        for name, lit in frags:
            if lit not in text:
                raise ValueError(
                    f'the {name!r} hint text here no longer appears verbatim in {src}. The two copies must stay '
                    f'identical or the OPD branch distils from a different teacher prompt than OPSD does. '
                    f'Missing fragment: {lit!r}')
        joined = ''.join(f for n, f in frags if n == 'hint_minimal')
        if joined != self._HINT_MINIMAL:
            raise ValueError(f'_HINT_MINIMAL ({self._HINT_MINIMAL!r}) is not the concatenation of the fragments '
                             f'checked against opsd_trainer.py ({joined!r}); fix one of the two.')

    def _privileged_teacher_data(self, d: Dict[str, Any]) -> Dict[str, Any]:
        """The teacher's view of one row: spotlight/green-box image plus the hint text.

        The student's row is left completely alone -- original image, original prompt --
        so the privilege lives only on the teacher side, exactly as in OPSD.
        """
        from .opsd_trainer import OPSDTrainer  # lazy: see _prepare_opd_teacher
        td = dict(d)
        td['messages'] = deepcopy(d['messages'])
        td['images'] = deepcopy(d.get('images'))
        if not td.get('messages') or td['messages'][1].get('role') != 'user':
            raise ValueError('expected messages[1] to be the user turn when building the privileged prompt')
        hm = getattr(self.args, 'opsd_hint_mode', 'hint')
        if hm == 'hint':
            td['messages'][1]['content'] += self._HINT_FULL
        elif hm == 'hint_minimal':
            td['messages'][1]['content'] += self._HINT_MINIMAL
        # 'spotlight' and 'none' add no text at all.
        if hm != 'none':
            img0 = td['images'][0]
            if isinstance(img0, str):
                # A bare path string: mask_processor indexes ['path'], so normalise
                # to the dict form it and the template both accept.
                img0 = {'path': img0, 'bytes': None}
                td['images'][0] = img0
            if not img0.get('path'):
                raise RuntimeError('the privileged teacher needs a file PATH for the source screenshot, but this '
                                   f'rollout row carries only bytes ({sorted(img0)}). mask_processor reads '
                                   "images[0]['path'] to open the original; materialise the dataset with paths.")
            # Unbound call: mask_processor only reads self.args.opsd_*, every one of
            # which GRPOConfig already carries via TrainArgumentsMixin, so
            # opsd_trainer.py stays unmodified.
            mask_path = OPSDTrainer.mask_processor(self, td)
            img0['path'] = mask_path
            # CLEARING bytes is load-bearing, not tidiness. template/base.py:209 does
            #     image = image['bytes'] or image['path']
            # so bytes WIN: with bytes present, assigning path alone is a silent no-op
            # and the teacher reads the ORIGINAL unmasked screenshot while the OPD
            # loss, the gradient probe and the hint-parity check all still look
            # healthy. That would void the entire privileged-teacher premise
            # invisibly. Falsy bytes fall through to path by the same line.
            img0['bytes'] = None
        return td

    def _encode_opd_pair(self, rows: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Encode (student view, teacher view) of the SAME rollout tokens.

        Both sides carry byte-identical response token ids -- that is what makes the
        two logit tensors alignable after masking on ``labels != -100`` even though the
        teacher's prompt is longer (hint text, different image).
        """
        from .utils import replace_assistant_response_with_ids
        st, te = [], []
        for d in rows:
            sd = dict(d)
            sd['messages'] = deepcopy(d['messages'])
            sd['images'] = deepcopy(d.get('images'))
            td = self._privileged_teacher_data(d)
            ids = d.get('response_token_ids')
            if ids:
                lm = d.get('response_loss_mask') or None
                sd['messages'] = replace_assistant_response_with_ids(sd['messages'], ids, lm)
                td['messages'] = replace_assistant_response_with_ids(td['messages'], ids, lm)
            st.append(sd)
            te.append(td)
        with self._template_context(self.template):
            se = [self.template.encode(x, return_length=True) for x in st]
            te = [self.template.encode(x, return_length=True) for x in te]
            sb = to_device(self.template.data_collator(se), self.model.device)
            tb = to_device(self.template.data_collator(te), self.model.device)
        # Per ROW, not just the batch total: two rows whose lengths compensate
        # (student 40/60, teacher 60/40) pass a total-count check and are then
        # cross-paired into the KL -- a wrong number, not a crash. Checked here, at
        # encode time, so it costs nothing and fires before the two forwards.
        n_s = (sb['labels'] != -100).sum(dim=1).tolist()
        n_t = (tb['labels'] != -100).sum(dim=1).tolist()
        if n_s != n_t:
            raise RuntimeError(f'OPD pair misaligned per row: student {n_s} supervised tokens, teacher {n_t}. The '
                               'response token ids must be identical on both sides; a truncated teacher prompt '
                               '(bigger image + hint) is the usual cause -- raise --max_length.')
        if min(n_s, default=0) <= 0:
            raise RuntimeError(f'OPD pair has a row with zero supervised tokens: {n_s}. Every distilled rollout '
                               'must carry its own response token ids.')
        # IDENTITY, not merely count: the labels AT the supervised positions ARE the
        # response token ids, so an equal-length pair that somehow encoded different
        # text still fails here. Counts alone would let it through into the KL.
        for i in range(sb['labels'].shape[0]):
            a = sb['labels'][i][sb['labels'][i] != -100]
            b = tb['labels'][i][tb['labels'][i] != -100]
            if not torch.equal(a, b):
                raise RuntimeError(f'OPD pair row {i} supervises DIFFERENT token ids on the two sides '
                                   f'({a.tolist()[:16]}... vs {b.tolist()[:16]}...), so the two logit rows are '
                                   'not comparable. replace_assistant_response_with_ids must write the same ids '
                                   'into both the student and the teacher message list.')
        # END-TO-END proof that the privilege survived encoding. Everything above
        # checks intent (path set, hint appended); this checks the PIXELS the teacher
        # will actually attend to. If the spotlight/green box failed to land -- a
        # bytes-vs-path precedence change, a mask_processor no-op, a cache collision
        # -- the two forwards would differ only by the hint sentence, the OPD term
        # would still train, and no metric would say so. Checked once per pair.
        if getattr(self.args, 'opsd_hint_mode', 'hint') != 'none':
            pv_s, pv_t = sb.get('pixel_values'), tb.get('pixel_values')
            if pv_s is not None and pv_t is not None and pv_s.shape == pv_t.shape \
                    and torch.equal(pv_s, pv_t):
                raise RuntimeError('the privileged teacher is looking at the SAME pixels as the student: the '
                                   f'{self.args.opsd_mask_mode!r} mask did not reach the encoded image. See '
                                   'template/base.py:209 -- bytes win over path. The A_OPD term would train '
                                   'against a non-privileged teacher and nothing else would report it.')
        return {'student': sb, 'teacher': tb}

    def _nzg_opd_token_weights(self, labels_row: torch.Tensor, mask_row: torch.Tensor, n: int,
                               device) -> torch.Tensor:
        """Per-token weights for one row of the OPD term (--nzg_opd_weight_mode).

        uniform  : every supervised token 1.0.
        position : a maximal run of L digit tokens (one coordinate value) gets
                   beta * [L, L-1, ..., 1], i.e. beta * k_t with k_t the significance of the
                   digit (hundreds 3, tens 2, units 1 for a 3-digit value); other tokens 1.0.
        """
        mode = getattr(self.args, 'nzg_opd_weight_mode', 'position') or 'position'
        if mode == 'uniform':
            return torch.ones(n, device=device, dtype=torch.float32)
        if mode != 'position':
            raise ValueError(f"--nzg_opd_weight_mode must be 'uniform' or 'position', got {mode!r}")
        beta = float(getattr(self.args, 'nzg_opd_pos_beta', 1.0))
        if not hasattr(self, '_nzg_digit_ids'):
            tok = getattr(self.processing_class, 'tokenizer', self.processing_class)
            ids = set()
            for d in '0123456789':
                enc = tok.encode(d, add_special_tokens=False)
                if len(enc) != 1:
                    raise RuntimeError(f'digit {d!r} is not a single token ({enc}); position weighting '
                                       'assumes one token per digit')
                ids.add(enc[0])
            self._nzg_digit_ids = ids
        # Same shift as the mask: position t is scored against token t+1.
        tokens = torch.roll(labels_row, shifts=-1, dims=0)[mask_row].tolist()
        if len(tokens) != n:
            raise RuntimeError(f'OPD weight/token misalignment: {len(tokens)} tokens vs {n} supervised')
        w = [1.0] * n
        i = 0
        while i < n:
            if tokens[i] in self._nzg_digit_ids:
                j = i
                while j < n and tokens[j] in self._nzg_digit_ids:
                    j += 1
                for k in range(i, j):
                    w[k] = beta * float(j - k)
                i = j
            else:
                i += 1
        return torch.tensor(w, device=device, dtype=torch.float32)

    def _nzg_opd_loss(self, pair: Dict[str, Any]) -> torch.Tensor:
        """Weighted reverse KL ``D_KL(P_S || P_T)`` over the response tokens.

        Calls OPSDTrainer.generalized_jsd_loss unchanged (a @staticmethod, so it needs
        no instance). ``beta=1.0`` selects pure reverse KL in that function's
        convention -- the same thing ``--beta 1`` selects for OPSD. Note this is the
        FULL-vocabulary closed form, a lower-variance estimator of the same objective
        than RSTG's single-sampled-token ``A_OPD``. Token weights come from
        _nzg_opd_token_weights (--nzg_opd_weight_mode / --nzg_opd_pos_beta).

        UNIT: returns the SUM over this micro-batch's rows of each row's own
        token-mean KL -- the same unit _nzg_sft_loss returns, because the caller
        divides by the GLOBAL row count (_nzg_scale_from('nzg_opd_total')).
        generalized_jsd_loss returns ONE token mean over whatever it is handed, so
        feeding it every row flattened together would return a single mean where n
        rows are being paid for: the term would be diluted ~n-fold AND silently
        re-weighted by sequence length (long rollouts dominating short ones). Hence
        the per-row loop.
        """
        from .opsd_trainer import OPSDTrainer
        sb, tb = pair['student'], pair['teacher']
        drop = ('prompt', 'labels', 'loss_scale', 'length', 'text_position_ids')
        s_in = {k: v for k, v in sb.items() if k not in drop}
        t_in = {k: v for k, v in tb.items() if k not in drop}
        with self.template.forward_context(self.model, sb):
            out_s = self.model(**s_in)
        with torch.no_grad(), self.template.forward_context(self.teacher_model, tb):
            out_t = self.teacher_model(**t_in)
        m_s = torch.roll(sb['labels'], shifts=-1, dims=1) != -100
        m_t = torch.roll(tb['labels'], shifts=-1, dims=1) != -100
        n_s = m_s.sum(dim=1).tolist()
        n_t = m_t.sum(dim=1).tolist()
        if n_s != n_t:
            # Cheap re-check of the encode-time invariant: per-row slicing below is
            # only meaningful if row i means the same tokens on both sides.
            raise RuntimeError(f'OPD logit misalignment per row: student {n_s} vs teacher {n_t}')
        wmean = bool(getattr(self.args, 'opsd_weighted_mean', False))
        # float32 accumulator: generalized_jsd_loss promotes to fp32 through the fp32
        # weights, and summing row means in bf16 would round away most of the term.
        total = torch.zeros((), device=out_s.logits.device, dtype=torch.float32)
        for i, n in enumerate(n_s):
            if n <= 0:
                raise RuntimeError(f'OPD row {i} has zero supervised tokens; see _encode_opd_pair')
            # [n_i, V] each: generalized_jsd_loss flattens to (-1, V) anyway, and the
            # per-row slice never materialises the whole masked [N, V] copy at once.
            s_i = out_s.logits[i][m_s[i]]
            t_i = out_t.logits[i][m_t[i]]
            w = self._nzg_opd_token_weights(sb['labels'][i], m_s[i], n, s_i.device)
            total = total + OPSDTrainer.generalized_jsd_loss(
                student_logits=s_i, teacher_logits=t_i, beta=1.0, weights=w, weighted_mean=wmean)
        return total

    def _nzg_backward_opd(self, pair: Dict[str, Any], coef: float) -> float:
        """Backward the OPD term with ZeRO-2's reduction hooks suppressed (see the SFT one)."""
        gas = getattr(self, 'current_gradient_accumulation_steps', None) or self.args.gradient_accumulation_steps
        params = [p for p in self.model.parameters() if p.requires_grad]
        opd_loss = self._nzg_opd_loss(pair)
        term = (coef / float(gas)) * opd_loss
        # Same probe as the SFT branch, for the same reason: a suppressed-hook
        # backward that deposits nothing is invisible to every aggregate metric --
        # nzg/opd_loss and nzg/opd_rows would both look perfectly healthy.
        want = self.nzg_log_routing
        before = self._nzg_grad_norm(params) if want else 0.0
        prior = [getattr(p, 'ds_grad_is_ready', True) for p in params]
        for p in params:
            p.ds_grad_is_ready = False
        try:
            term.backward()
        finally:
            for p, v in zip(params, prior):
                p.ds_grad_is_ready = v
        after = self._nzg_grad_norm(params) if want else 0.0
        # A_OPD backwards FIRST, so `before` is 0 on the first routed micro-batch of every
        # optimizer step and the probe is unambiguous there. Still test for UNCHANGED
        # rather than "did not increase": on a later micro-batch within the same step the
        # accumulator is non-zero and cancellation is possible. See the SFT probe.
        if want and after == before:
            logger.warning(f'[opsd+grpo] the A_OPD backward deposited NO gradient '
                           f'(||g|| unchanged at {before:.4e}): the branch is a silent no-op.')
        return float(opd_loss.detach().item()), (before, after)

    def _nzg_reduce_diag(self, sft_loss_val, rows, probe):
        """SUM the rank-local diagnostics across ranks. Must be called by EVERY rank."""
        t = torch.tensor([sft_loss_val, rows, probe[0], probe[1]],
                         dtype=torch.float32, device=self.accelerator.device)
        t = self.accelerator.reduce(t, reduction='sum')
        v = t.tolist()
        return v[0], v[1], (v[2], v[3])

    @staticmethod
    @torch.no_grad()
    def _nzg_grad_norm(params) -> float:
        tot = 0.0
        for p in params:
            if p.grad is not None:
                tot += float(p.grad.detach().float().pow(2).sum().item())
        return tot**0.5

    def _nzg_sft_loss(self, model, sft_inputs: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Sum over this micro-batch's owners of the per-sequence token-mean CE.

        ``A_SFT = 1`` on every token of the ground-truth tool call.

        No temperature scaling: this is a supervised target, not a policy ratio.
        """
        labels = sft_inputs['labels']
        # Denylist, never a signature whitelist: Qwen3-VL's forward takes the vision
        # tensors through **kwargs, so `k in self.model_kwarg_keys` would silently
        # drop pixel_values and train the model on text alone. This is the same
        # denylist OPSDTrainer.compute_loss uses (opsd_trainer.py:186), plus the two
        # keys compute_chord_loss pops.
        drop = ('prompt', 'labels', 'loss_scale', 'length', 'text_position_ids')
        model_inputs = {k: v for k, v in sft_inputs.items() if k not in drop and not k.startswith('nzg_')}
        with self.template.forward_context(self.model, sft_inputs):
            outputs = model(**model_inputs)
        per_token = per_token_loss_func(outputs, labels).view_as(labels)      # [B, T], -100 -> 0
        mask = torch.roll(labels, shifts=-1, dims=-1) != -100
        if not bool(mask.any()):
            raise RuntimeError('SFT batch has zero supervised tokens')
        # SUM of per-sequence token means, NOT one batch-wide token mean: the caller
        # divides by the GLOBAL owner count, so each owner must contribute its own
        # full mean. A batch-wide mean would make a 2-owner micro-batch count as one.
        per_seq = (per_token * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)
        return per_seq[mask.any(dim=1)].sum()
