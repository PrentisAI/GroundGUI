"""
BBox reward, adapted from the bbox reward design of GUI-G1 (arXiv 2505.15810).

GUI-G1 uses a linear combination of three rewards:
  R_total = R_hit + α * R_iou + β * R_box

  - R_hit:  whether the predicted center falls inside the GT bbox (0/1)
  - R_iou:  IoU between the predicted bbox and the GT bbox
  - R_box:  normalized distance constraint between the four predicted and GT bbox coordinates
            R_box = 4 / (d_x1 + d_x2 + d_y1 + d_y2)
            where d_xi = 1 / (1 - |pred_xi - gt_xi| / image_dim)

Default hyperparameters:  α=0.25, β=0.125

Note: our grounding model outputs a point (not a bbox). Treating the prediction as a
degenerate bbox (pred_x, pred_y, pred_x, pred_y) gives zero area and hence IoU = 0,
so only R_hit and R_box would carry signal.

We therefore adapt the terms as follows:
  - R_hit:  whether the predicted point is inside the GT bbox (same as ground-acc)
  - R_iou:  replaced by a distance-based soft reward (Gaussian decay, same form as g2_point_reward)
  - R_box:  normalized distance constraint from the predicted point to the GT bbox center
"""

from typing import Dict, List, Union
import re
import json
import math

from swift.custom_utils.format_func import extract_action
from swift.custom_utils.ground_func import (
    pointnorm2real,
    calculate_pred_norm_point,
    bboxreal2norm,
)


class ORM:
    """Base class for synchronous outcome reward models (ORM)."""

    def __call__(self, **kwargs) -> List[float]:
        raise NotImplementedError


class BBoxReward(ORM):
    """
    GUI-G1 style bbox reward: R = R_hit + α * R_dist + β * R_box

    R_hit:  1.0 if the predicted point is inside the GT bbox, else 0.0
    R_dist: Gaussian decay of the distance from the predicted point to the GT center (soft reward, 0~1)
    R_box:  coordinate distance constraint R_box = 2 / (d_x + d_y)
            d_x = 1 / (1 - |pred_x - gt_cx| / img_w)
            d_y = 1 / (1 - |pred_y - gt_cy| / img_h)
            the closer the prediction is to the GT center, the larger R_box (max 1.0)
    """

    def __init__(self, alpha=0.25, box_beta=0.125):
        # NOTE: the second weight MUST NOT be called `beta`. `_prepare_rewards`
        # (swift/rlhf_trainers/grpo_trainer.py) auto-fills every __init__ parameter
        # whose name exists on the training args:
        #     reward_func_kwargs = {k: getattr(args, k) for k in signature(...) if hasattr(args, k)}
        # and `beta` is GRPOConfig's KL coefficient. Naming it `beta` would silently
        # replace 0.125 with the KL beta (0.04 by default, 0.0 when the reference
        # model is disabled), switching R_box off without any warning.
        self.alpha = alpha
        self.beta = box_beta

    def __call__(self, completions, solution, additional_paras, **kwargs) -> List[float]:
        rewards = []
        for predict_str, ground_truth, para in zip(completions, solution, additional_paras):
            if isinstance(para, str):
                para = json.loads(para)
            image_size = para['image_size']
            reward = self.compute_reward(predict_str, ground_truth, image_size)
            rewards.append(reward)
        return rewards

    def compute_reward(self, predict_str: str, ground_truth: dict, image_size: list) -> float:
        try:
            pred_action = extract_action(predict_str)
            if pred_action == "no action":
                return 0.0

            # Predicted point, normalized to 0-1000
            pred_point = calculate_pred_norm_point(
                image_size, pred_action['arguments']['coordinate'], "qwen3vl"
            )
            pred_x, pred_y = pred_point

            # GT bbox normalized to 0-1000
            gt_bbox = bboxreal2norm(ground_truth['arguments']['coordinate'], image_size)
            gt_x1, gt_y1, gt_x2, gt_y2 = gt_bbox
            gt_cx = (gt_x1 + gt_x2) / 2
            gt_cy = (gt_y1 + gt_y2) / 2

            # ====== R_hit: point inside bbox ======
            r_hit = 1.0 if (gt_x1 <= pred_x <= gt_x2 and gt_y1 <= pred_y <= gt_y2) else 0.0

            # ====== R_dist: Gaussian distance decay ======
            gt_w = max(gt_x2 - gt_x1, 1)
            gt_h = max(gt_y2 - gt_y1, 1)
            sigma_x = 0.5 * gt_w
            sigma_y = 0.5 * gt_h
            x_term = (pred_x - gt_cx) ** 2 / (sigma_x ** 2)
            y_term = (pred_y - gt_cy) ** 2 / (sigma_y ** 2)
            r_dist = math.exp(-0.5 * (x_term + y_term))

            # ====== R_box: coordinate distance constraint ======
            # Distances in [0, 1] (divide by 1000 since coordinates are in 0-1000)
            dx = abs(pred_x - gt_cx) / 1000.0
            dy = abs(pred_y - gt_cy) / 1000.0
            # Avoid division by zero
            d_x = 1.0 / max(1.0 - dx, 1e-6)
            d_y = 1.0 / max(1.0 - dy, 1e-6)
            r_box = 2.0 / (d_x + d_y)

            # ====== Total reward ======
            reward = r_hit + self.alpha * r_dist + self.beta * r_box
            return round(reward, 4)

        except Exception:
            return 0.0


class BBoxFormat(ORM):
    """Format reward: 1.0 if an action can be parsed from the completion, else 0.0."""

    def __call__(self, completions, solution, **kwargs) -> List[float]:
        rewards = []
        for predict_str, ground_truth in zip(completions, solution):
            action = extract_action(predict_str)
            rewards.append(1.0 if action != "no action" else 0.0)
        return rewards
