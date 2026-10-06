from typing import Dict, List, Union
import re
import json 
import os
import math

from swift.custom_utils.format_func import (
    extract_action,
    extract_ground
)


from swift.custom_utils.ground_func import (
    pointreal2norm,
    get_scroll_direction,
    pointnorm2real,
    bboxreal2norm,
    
    calculate_pred_norm_point,
    ground_reward_func,
    ground_reward_jigsaw_func
)


def gaussian_point_reward(pred_point, gt_bbox, alpha=0.5):
    """
    Gaussian point reward from GUI-G2.
    Turns the sparse 0/1 hit reward into a continuous dense reward (0~1).
    
    The Gaussian is centered on the gt_bbox center with standard deviation alpha * (width, height);
    the reward is the unnormalized Gaussian density at the predicted point.
    Returns 0.0 for a degenerate (zero-width or zero-height) bbox.
    
    Reference: https://github.com/ZJU-REAL/GUI-G2
    """
    pred_x, pred_y = pred_point
    gt_x1, gt_y1, gt_x2, gt_y2 = gt_bbox

    gt_center_x = (gt_x1 + gt_x2) / 2
    gt_center_y = (gt_y1 + gt_y2) / 2
    gt_width = gt_x2 - gt_x1
    gt_height = gt_y2 - gt_y1

    sigma_x = alpha * gt_width
    sigma_y = alpha * gt_height

    # Avoid division by zero
    if sigma_x < 1e-8 or sigma_y < 1e-8:
        return 0.0

    x_term = (pred_x - gt_center_x) ** 2 / (sigma_x ** 2)
    y_term = (pred_y - gt_center_y) ** 2 / (sigma_y ** 2)
    exponent = -0.5 * (x_term + y_term)

    point_reward = math.exp(exponent)
    point_reward = round(point_reward, 3)
    return point_reward


class ORM:
    """Base class for synchronous outcome reward models (ORM).

    Subclasses should implement the __call__ method to compute rewards.

    Example:
        class MyReward(ORM):
            def __call__(self, completions, **kwargs) -> List[float]:
                return [1.0 if len(c) > 100 else 0.0 for c in completions]
    """

    def __call__(self, **kwargs) -> List[float]:
        raise NotImplementedError


# For additional reward functions, refer to swift/plugin/orm.py.
class G2Acc(ORM):
    
        
    def __call__(self, completions, solution, additional_paras, **kwargs) -> List[float]:
        rewards = []
        for predict_str, ground_truth, para in zip(completions, solution, additional_paras):
            if isinstance(para, str):
                para = json.loads(para)
            image_size = para['image_size']

            accuracy = self.ground_acc_reward(predict_str, ground_truth, image_size)
            rewards.append(accuracy) 
        return rewards


    def ground_acc_reward(self, predict_str: str, ground_truth: str, image_size) -> float:
        """
        GUI-G2 Gaussian point reward instead of a binary 0/1 hit reward.
        The closer the predicted point is to the GT bbox center, the closer the reward is to 1;
        it decays smoothly towards 0 with distance.
        """
        try:
            # Action type and arguments of the ground truth
            gt_action_type = ground_truth['arguments']['action']
            
            pred_action = extract_action(predict_str)
            
            if pred_action == "no action":
                return 0.0

            # pred to norm 
            pred_point = calculate_pred_norm_point(image_size, pred_action['arguments']['coordinate'], "qwen3vl")
            
            # gt bbox to norm (the coordinate field is a [x1, y1, x2, y2] bbox)
            gt_bbox_norm = bboxreal2norm(ground_truth['arguments']['coordinate'], image_size)
            
            # GUI-G2 Gaussian point reward
            return gaussian_point_reward(pred_point, gt_bbox_norm)
            
        except Exception as e:
            return 0.0




class G2Format(ORM):
    

    def __call__(self, completions, solution, **kwargs) -> List[float]:
        """
            Return 1.0 per completion whose action can be parsed (valid action-space format), else 0.0.
        """
    
        rewards = []
        for predict_str, ground_truth in zip(completions, solution):
            
            format = self.ground_format_reward(predict_str)
            rewards.append(format) 
        return rewards

    def ground_format_reward(self, predict_str: str) -> float:
        action = extract_action(predict_str)
        if action == "no action":
            return 0.0
        else:
            return 1.0 




