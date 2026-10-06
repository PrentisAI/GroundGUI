# Copyright (c) ModelScope Contributors. All rights reserved.
"""TtRT -- Trust the Right Teacher: quality-aware self-distillation for GUI grounding.

Paper: "Trust the Right Teacher: Quality-Aware Self-Distillation for GUI
Grounding", arXiv:2606.18101v1.

TtRT keeps GUI-SD's visually privileged teacher input verbatim (in this repo:
``--opsd_mask_mode gaussian --opsd_hint_mode hint``) and replaces only the
per-token distillation weight.  For a coordinate digit token at position t::

    d*_t = argmax_{d in D} P_T^t(d)          # D = the ten decimal digit tokens
    p_t  = P_T^t(d*_t)                       # full-vocabulary probability (eq. 5)
    h_t  = 1 if appending d*_t to the student's axis-specific prefix induced by
           y_<t still leaves a completion inside the GT box interval on that
           axis, else 0                                                (eq. 3)
    g_t  = alpha + (1 - alpha) * h_t         # soft correctness gate    (eq. 4)
    w_t  = lambda * g_t * p_t                                          (eq. 6/7)

Non-coordinate response tokens keep ``w_t = 1``.  Paper values: alpha = 0.5,
lambda = 3.  The loss is the same weighted reverse-KL over response tokens the
parent already computes (eq. 8), so ``--beta 1`` still selects KL(P_S || P_T).

Why this is a subclass that edits nothing in ``OPSDTrainer``
------------------------------------------------------------
``OPSDTrainer.compute_loss`` builds the OPSD base weights inline, gathers them
as ``valid_weights = weights[mask_student]``, and then dispatches the loss
through the *instance* -- ``self.generalized_jsd_loss(...)`` with all-keyword
arguments (opsd_trainer.py:287-293) -- even though the callee is declared as a
``@staticmethod`` (opsd_trainer.py:697).  Overriding it here with a bound method
therefore wins the lookup, receives ``teacher_logits`` already gathered to
``[1, N_valid, V]``, and can hand the reweighted vector straight back to
``super().generalized_jsd_loss(...)``, inheriting the unchanged chunked-JSD loop
including the ``opsd_weighted_mean`` denominator.  ``compute_loss`` is inherited
untouched; only ``_prepare_batch_inputs`` is extended (via ``super()``) to carry
the ground-truth box into the loss.

Preconditions (asserted in ``__init__``)
----------------------------------------
* ``--opsd_token_weight_mode uniform``.  TtRT's weights are absolute, so the
  OPSD base weight they multiply must be exactly 1.0.  ``uniform`` takes the
  ``torch.ones_like`` branch and skips the entropy factor.
* not liger.  The liger path bypasses ``generalized_jsd_loss`` entirely (and is
  ``assert False`` in the parent today), which would silently degrade TtRT to
  plain OPSD.
* ``--seq_kd false`` and, when ``lmbda > 0``, ``--use_vllm true``.  The two
  prompt-only rollout branches in the parent write ``input_ids``/``labels`` at
  the OUTER dict level, where ``compute_loss`` never reads them -- they are
  already broken for OPSD, and TtRT must not appear to run on them.
"""
import json
from typing import Dict, List, Optional, Sequence, Tuple

import torch

from swift.custom_utils.ground_func import bboxreal2norm
from swift.utils import get_logger
# Importing the parent module also runs its guarded, idempotent trl MRO patch
# (``del _gkd_cls.__init__``).  Do not duplicate that block here.
from .opsd_trainer import OPSDTrainer

logger = get_logger()

# A norm-1000 coordinate is 1-3 digits; 4 tolerates a stray "1000".  A longer
# digit run means the rollout is not the canonical tool call, so the sample is
# treated as having no coordinate tokens at all (see _build_ttrt_state).
MAX_COORD_DIGITS = 4

# Number of digit spans the canonical response contains: "coordinate": [cx, cy].
N_COORD_SLOTS = 2


# ---------------------------------------------------------------------------
# Pure helpers.  No trainer state, no CUDA -- unit-testable on CPU.
# ---------------------------------------------------------------------------
def digit_token_map(tokenizer) -> Dict[int, int]:
    """``{token_id: digit_value}`` for the ten decimal digits.

    Same construction as the parent's ``self._digit_tokens``
    (opsd_trainer.py:229-230), but keeping the value so digit prefixes can be
    assembled.  Verified on Qwen3-VL-8B: ids 15..24, exactly one token per
    digit, no multi-digit and no space-prefixed digit tokens (its pre-tokenizer
    splits numbers one digit at a time).
    """
    return {tokenizer.encode(str(i), add_special_tokens=False)[-1]: i for i in range(10)}


def gt_bbox_norm1000(data: dict) -> List[int]:
    """GT bbox of one raw sample, absolute pixel xyxy -> norm-1000 xyxy.

    ``data['solution']['arguments']['coordinate']`` is in absolute pixels and
    ``data['additional_paras']`` is still a JSON *string* on the raw row (the
    parent only ``json.loads``es its own deepcopy).  Delegates to the existing
    ``ground_func.bboxreal2norm`` rather than re-inlining the conversion the
    parent hand-wrote for ``opsd_hint_mode == 'gt'``.
    """
    gt_bbox = data['solution']['arguments']['coordinate']
    additional = data.get('additional_paras', '{}')
    if isinstance(additional, str):
        additional = json.loads(additional)
    return bboxreal2norm(gt_bbox, additional['image_size'])


def iter_digit_spans(seq: Sequence[int], digit_ids) -> List[Tuple[int, int]]:
    """``(start, length)`` of every maximal run of digit tokens in ``seq``."""
    spans = []
    i, n = 0, len(seq)
    while i < n:
        if seq[i] in digit_ids:
            j = i
            while j < n and seq[j] in digit_ids:
                j += 1
            spans.append((i, j - i))
            i = j
        else:
            i += 1
    return spans


def prefix_reachable(prefix_value: int, remaining: int, lo: int, hi: int, free_length: bool = False) -> bool:
    """Can a number that starts with ``prefix_value`` still land inside ``[lo, hi]``?

    ``prefix_value`` already includes the digit under test (the paper appends
    ``d*_t`` to the student's prefix before checking), and ``remaining`` is how
    many digits follow it in the student's own number.  The faithful test is
    therefore the single interval
    ``[prefix * 10**remaining, prefix * 10**remaining + 10**remaining - 1]``.

    ``free_length=True`` relaxes it to the union over every completion length
    ``0..remaining`` (i.e. the number may also terminate early).  That is a
    strictly larger reachable set and an ablation, not the paper's reading --
    the paper evaluates the teacher along a *fixed* student trajectory, where
    the number of remaining digits is known.
    """
    lengths = range(remaining + 1) if free_length else (remaining, )
    for r in lengths:
        scale = 10**r
        span_lo = prefix_value * scale
        if span_lo <= hi and lo <= span_lo + scale - 1:
            return True
    return False


class TtRTTrainer(OPSDTrainer):
    """OPSD with TtRT's quality-aware coordinate-token weights.

    Selected by ``--rlhf_type gkd --use_ttrt true`` (see
    ``TrainerFactory.get_cls``); ``rlhf_type`` deliberately stays ``'gkd'`` so
    the whole surrounding pipeline stays wired exactly as it is for OPSD.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        mode = getattr(self.args, 'opsd_token_weight_mode', 'linear')
        assert mode == 'uniform', (
            f'TtRT requires --opsd_token_weight_mode uniform (got {mode!r}). TtRT weights are absolute '
            '(lambda*g_t*p_t on coordinate digits, 1 elsewhere), so the OPSD base weight they multiply '
            'must be exactly 1.0; any other mode folds a second weight into the objective. Note the '
            'parent dispatches this arg by substring, so a typo silently falls back to uniform.')
        assert not self.use_liger_gkd_loss, (
            'TtRT requires the non-liger loss path: the liger branch bypasses generalized_jsd_loss, '
            'so the TtRT weights would never be applied.')
        assert not self.seq_kd, 'TtRT does not support --seq_kd true (teacher-generated rollouts).'
        if self.lmbda > 0:
            assert getattr(self.args, 'use_vllm', False), (
                'TtRT requires --use_vllm true when --lmbda > 0: the non-vLLM on-policy branch encodes '
                'prompts only and overwrites labels at the outer dict level, which compute_loss never '
                'reads (already broken for OPSD).')

        self.ttrt_lambda = float(getattr(self.args, 'ttrt_lambda', 3.0))
        self.ttrt_alpha = float(getattr(self.args, 'ttrt_alpha', 0.5))
        self.ttrt_free_length = bool(getattr(self.args, 'ttrt_free_length', False))
        assert 0.0 <= self.ttrt_alpha <= 1.0, f'ttrt_alpha must be in [0, 1], got {self.ttrt_alpha}'
        assert self.ttrt_lambda > 0, f'ttrt_lambda must be > 0, got {self.ttrt_lambda}'

        # Both papers optimise the reverse KL D_KL(P_S || P_T); the parent selects
        # it with beta == 1 (beta == 0 is KL(P_T || P_S), the opposite direction,
        # and the gkd default is 0.5). Warn rather than assert so a JSD ablation
        # stays possible -- but a silently wrong divergence would invalidate the
        # comparison, so make it loud at startup.
        if float(self.beta) != 1.0:
            logger.warning(f'[TtRT] beta={self.beta} but TtRT/GUI-SD both use the reverse KL '
                           'D_KL(P_S||P_T), which this trainer selects with --beta 1.')

        self._ttrt_digit_map = digit_token_map(self.template.tokenizer)
        _items = sorted(self._ttrt_digit_map.items())
        self._ttrt_digit_ids = torch.tensor([tid for tid, _ in _items], dtype=torch.long)
        self._ttrt_digit_vals = torch.tensor([val for _, val in _items], dtype=torch.long)
        self._ttrt_state = None
        self._ttrt_stats = self._new_ttrt_stats()

        logger.info(f'[TtRT] lambda={self.ttrt_lambda} alpha={self.ttrt_alpha} '
                    f'free_length={self.ttrt_free_length} '
                    f'weighted_mean={getattr(self.args, "opsd_weighted_mean", False)} '
                    f'digit_ids={sorted(self._ttrt_digit_map)}')

    @staticmethod
    def _new_ttrt_stats() -> Dict[str, float]:
        return {'n_coord': 0.0, 'n_pass': 0.0, 'p_sum': 0.0, 'w_sum': 0.0, 'n_rows': 0.0, 'n_bad': 0.0}

    # -- carry the ground-truth box into the loss --------------------------
    def _prepare_batch_inputs(self, inputs: list, encode_prompt_only: bool = False):
        # The parent runs unmodified.  It deepcopies every row before touching
        # it, so ``inputs`` here is still the raw rows (with 'solution' and the
        # unparsed 'additional_paras' string), and it appends one encoded item
        # per row before the collator pads in list order -- so batch row b
        # corresponds to inputs[b].
        batch = super()._prepare_batch_inputs(inputs, encode_prompt_only=encode_prompt_only)
        if encode_prompt_only:
            # Prompt-only encodings carry no response; __init__ already refuses
            # the configurations that would then reach compute_loss.
            self._ttrt_state = None
        else:
            self._ttrt_state = self._build_ttrt_state(inputs, batch['student_inputs']['labels'])
        return batch

    @torch.no_grad()
    def _build_ttrt_state(self, inputs: list, labels: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Flat ``[N_valid]`` metadata aligned with the parent's ``valid_weights``.

        ``valid_weights = weights[mask_student]`` is a row-major boolean gather,
        so each row's valid positions form a contiguous block in the flat axis.
        Everything below is built at flat indices directly, which is why the
        shift/mask convention has to match the parent exactly:
        ``torch.roll(labels, -1, dims=1)`` then ``!= -100``.
        """
        shifted = torch.roll(labels, shifts=-1, dims=1)
        mask = shifted != -100
        counts = mask.sum(dim=1).tolist()

        digit_map = self._ttrt_digit_map
        n_valid = int(sum(counts))
        is_coord = [False] * n_valid
        prefix = [0] * n_valid
        remaining = [0] * n_valid
        lo = [0] * n_valid
        hi = [0] * n_valid

        n_bad = 0
        offset = 0
        for b, n_row in enumerate(counts):
            # Only the response tokens, in flat order.
            sub = shifted[b][mask[b]].tolist()
            spans = iter_digit_spans(sub, digit_map)
            if len(spans) != N_COORD_SLOTS or any(n > MAX_COORD_DIGITS for _, n in spans):
                # Unexpected layout (truncated / malformed rollout): the axis of
                # each digit cannot be established, so this row is treated as
                # having no coordinate tokens and every position keeps weight 1.
                n_bad += 1
                offset += n_row
                continue
            x1, y1, x2, y2 = gt_bbox_norm1000(inputs[b])
            if x1 > x2 or y1 > y2:
                # Inverted box: every position would fail the gate, i.e. the whole
                # sample would be silently down-weighted to alpha. Count it as a
                # bad row instead. (gui-sd.jsonl has none; 8/6990 rows do exceed
                # 1000 on the high edge, which is harmless -- it only widens the
                # reachable interval and never makes `lo` unreachable.)
                n_bad += 1
                offset += n_row
                continue
            for slot, (start, length) in enumerate(spans):
                a_lo, a_hi = (x1, x2) if slot == 0 else (y1, y2)
                prefix_val = 0
                for k in range(length):
                    f = offset + start + k
                    is_coord[f] = True
                    prefix[f] = prefix_val          # student digits BEFORE this position
                    remaining[f] = length - k - 1   # digits AFTER this position
                    lo[f] = a_lo
                    hi[f] = a_hi
                    prefix_val = prefix_val * 10 + digit_map[sub[start + k]]
            offset += n_row

        if n_bad:
            logger.warning(f'[TtRT] {n_bad}/{len(counts)} rollouts had an unexpected digit-span layout; '
                           'their tokens keep weight 1.0 (no coordinate weighting applied).')

        dev = labels.device
        return {
            'is_coord': torch.tensor(is_coord, dtype=torch.bool, device=dev),
            'prefix': torch.tensor(prefix, dtype=torch.long, device=dev),
            'remaining': torch.tensor(remaining, dtype=torch.long, device=dev),
            'lo': torch.tensor(lo, dtype=torch.long, device=dev),
            'hi': torch.tensor(hi, dtype=torch.long, device=dev),
            'n_rows': len(counts),
            'n_bad': n_bad,
        }

    # -- the TtRT weights --------------------------------------------------
    def generalized_jsd_loss(self,
                             student_logits,
                             teacher_logits,
                             labels=None,
                             beta=0.5,
                             temperature=1.0,
                             chunk_size=512,
                             weights=None,
                             weighted_mean=False):
        """Reweight, then delegate the unchanged loss to the parent.

        Declared as an instance method on purpose: the parent's is a
        ``@staticmethod`` but is dispatched as ``self.generalized_jsd_loss(...)``
        with all-keyword arguments, so this override wins the lookup and
        ``super()`` still yields the plain parent function.
        """
        weights = self._ttrt_reweight(teacher_logits, weights)
        return super().generalized_jsd_loss(
            student_logits=student_logits,
            teacher_logits=teacher_logits,
            labels=labels,
            beta=beta,
            temperature=temperature,
            chunk_size=chunk_size,
            weights=weights,
            weighted_mean=weighted_mean)

    @torch.no_grad()
    def _ttrt_reweight(self, teacher_logits: torch.Tensor, weights: Optional[torch.Tensor]):
        state = self._ttrt_state
        if state is None or weights is None:
            # Unreachable under the configurations __init__ accepts.  Fail loud:
            # silently falling back would report plain-OPSD results as TtRT.
            raise RuntimeError('TtRT: no coordinate state for this batch (state=%s, weights=%s). '
                               'compute_loss was reached through a path that does not go through '
                               'TtRTTrainer._prepare_batch_inputs.' % (state is None, weights is None))

        is_coord = state['is_coord']
        # teacher_logits is outputs_teacher.logits[mask_teacher][None] -> [1, N, V].
        # It shares the row-major order of valid_weights and of the student
        # logits because all three are boolean gathers under masks of equal
        # count (the parent already requires identical student/teacher response
        # tokens).  Turn any future divergence into a crash, not a mis-weighting.
        assert is_coord.numel() == weights.numel() == teacher_logits.shape[1], (
            f'TtRT weight misalignment: is_coord={is_coord.numel()} weights={weights.numel()} '
            f'teacher_logits={tuple(teacher_logits.shape)}')

        tl = teacher_logits[0]
        dev = tl.device
        if self._ttrt_digit_ids.device != dev:
            self._ttrt_digit_ids = self._ttrt_digit_ids.to(dev)
            self._ttrt_digit_vals = self._ttrt_digit_vals.to(dev)

        # d*_t: teacher argmax restricted to the digit tokens (eq. 2).
        digit_logits = tl.index_select(-1, self._ttrt_digit_ids).float()   # [N, 10]
        top = digit_logits.argmax(dim=-1)                                  # index into digit_ids
        d_star = self._ttrt_digit_vals[top]                                # digit VALUE

        # p_t = P_T^t(d*_t) under the FULL-vocabulary softmax (eq. 5).  Computed
        # with logsumexp so the [N, V] softmax is never materialised.  The parent
        # is called without `temperature`, so its loss also uses temperature 1.0
        # -- p_t from the raw logits is consistent with it, not a separate choice.
        lse = torch.logsumexp(tl.float(), dim=-1)                          # [N]
        p_t = torch.exp(digit_logits.gather(1, top.unsqueeze(1)).squeeze(1) - lse)

        h_t = self._reach_flags(state, d_star)                             # [N] float 0/1
        g_t = self.ttrt_alpha + (1.0 - self.ttrt_alpha) * h_t              # eq. 4

        coord_w = (self.ttrt_lambda * g_t * p_t).to(weights.dtype)         # eq. 6
        factor = torch.where(is_coord, coord_w, torch.ones_like(weights))

        s = self._ttrt_stats
        n_coord = int(is_coord.sum().item())
        if n_coord:
            s['n_coord'] += n_coord
            s['n_pass'] += float(h_t[is_coord].sum().item())
            s['p_sum'] += float(p_t[is_coord].sum().item())
            s['w_sum'] += float(coord_w[is_coord].float().sum().item())
        s['n_rows'] += float(state['n_rows'])
        s['n_bad'] += float(state['n_bad'])

        return weights * factor

    def _reach_flags(self, state: Dict[str, torch.Tensor], d_star: torch.Tensor) -> torch.Tensor:
        """Vectorised h_t (eq. 3).

        ``P = student_prefix * 10 + d*`` is the value after appending the
        teacher's top digit; with ``remaining`` digits still to come the
        reachable set is ``[P * 10**remaining, P * 10**remaining + 10**remaining - 1]``.
        """
        prefix, remaining = state['prefix'], state['remaining']
        lo, hi = state['lo'], state['hi']
        p = prefix * 10 + d_star

        def _hit(r_pow: torch.Tensor) -> torch.Tensor:
            span_lo = p * r_pow
            return (span_lo <= hi) & (lo <= span_lo + r_pow - 1)

        pow10 = torch.pow(torch.full_like(remaining, 10), remaining)
        ok = _hit(pow10)
        if self.ttrt_free_length:
            # Ablation: also allow the number to terminate before `remaining`.
            for r in range(int(remaining.max().item()) if remaining.numel() else 0):
                ok = ok | ((remaining >= r) & _hit(torch.full_like(remaining, 10**r)))
        return ok.float()

    # -- diagnostics -------------------------------------------------------
    def log(self, logs: Dict[str, float], start_time: Optional[float] = None) -> None:
        s = self._ttrt_stats
        if s['n_coord'] > 0 or s['n_rows'] > 0:
            if s['n_coord'] > 0:
                logs['ttrt/gate_pass_rate'] = s['n_pass'] / s['n_coord']
                logs['ttrt/p_t_mean'] = s['p_sum'] / s['n_coord']
                logs['ttrt/coord_weight_mean'] = s['w_sum'] / s['n_coord']
                logs['ttrt/coord_tokens_per_row'] = s['n_coord'] / max(s['n_rows'], 1.0)
            logs['ttrt/bad_layout_rate'] = s['n_bad'] / max(s['n_rows'], 1.0)
            self._ttrt_stats = self._new_ttrt_stats()
        # NOTE: these are rank-local counters; only rank 0's values are reported.
        super().log(logs, start_time)
