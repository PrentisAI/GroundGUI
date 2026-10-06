# Copyright (c) Alibaba, Inc. and its affiliates.
import collections
import time
from abc import ABC, abstractmethod
from typing import Dict, List, Literal

import numpy as np
from transformers.trainer_utils import EvalPrediction
from collections import defaultdict
import re
from tqdm import tqdm 
from PIL import Image, ImageFont, ImageDraw
import os 
from torchvision.ops.boxes import box_area
import torch
import json
from glob import glob 

from swift.custom_utils.format_func import (
    extract_action, 
    extract_ground
)

from swift.custom_utils.ground_func import (
    pointreal2norm,
    get_scroll_direction,
    pointnorm2real,
    calculate_pred_norm_point,
    qwen25_get_resize,
)
from swift.utils import read_from_jsonl


# ============ OSWorld-G official scoring logic (after GroundingEval) ============

def _is_point_in_rectangle(point, rect):
    """Return True if the point lies inside rect=[x1, y1, x2, y2] (inclusive)."""
    return rect[0] <= point[0] <= rect[2] and rect[1] <= point[1] <= rect[3]


def _is_point_in_polygon(point, polygon):
    """Ray-casting point-in-polygon test; polygon is a flat list [x0,y0,x1,y1,...]."""
    x, y = point
    n = len(polygon) // 2
    inside = False
    j = n - 1
    for i in range(n):
        xi, yi = polygon[i * 2], polygon[i * 2 + 1]
        xj, yj = polygon[j * 2], polygon[j * 2 + 1]
        if (yi > y) != (yj > y) and x < (xj - xi) * (y - yi) / (yj - yi) + xi:
            inside = not inside
        j = i
    return inside


def osworldg_eval(pred_point, box_type, box_coordinates, image_size):
    """
    Mirrors the scoring logic of the official OSWorld-G GroundingEval._eval.
    
    Args:
        pred_point: predicted point [x, y] in absolute pixel coordinates
        box_type: "bbox" | "polygon" | "refusal"
        box_coordinates: 
            - bbox: [x, y, w, h]
            - polygon: [x0,y0,x1,y1,...]  flat list
            - refusal: ignored
        image_size: [width, height]
    
    Returns:
        bool: whether the prediction is a hit
    """
    center_point = pred_point  # already an [x, y] point

    if box_type == "bbox":
        # box_coordinates is [x, y, w, h] -> convert to [x1, y1, x2, y2]
        gt_rect = [
            box_coordinates[0],
            box_coordinates[1],
            box_coordinates[0] + box_coordinates[2],
            box_coordinates[1] + box_coordinates[3],
        ]
        return _is_point_in_rectangle(center_point, gt_rect)
    
    elif box_type == "polygon":
        return _is_point_in_polygon(center_point, box_coordinates)
    
    elif box_type == "refusal":
        # refusal: correct only if both predicted coordinates are negative (the model declined to output a valid point)
        return all(center_point[i] < 0 for i in range(2))
    
    else:
        print(f"WARNING: unknown box_type '{box_type}', treated as wrong.")
        return False


# ============ Main metric ============

def compute_osworldg(data_list, metric):
    """
    Compute OSWorld-G metrics.
    
    Each sample's additional_paras contains:
        - id, GUI_types, image_size, box_type, category
    Each sample's solution contains:
        - {'name': 'mobile_use', 'arguments': {'action': 'click', 'coordinate': [bbox]}}
        - where coordinate holds the ground-truth box_coordinates
    """
    corr_action = 0
    num_wrong_format = 0
    num_action = len(data_list)
    
    # Per-group accuracy by category / GUI_types / box_type
    category_eval = defaultdict(list)
    gui_type_eval = defaultdict(list)
    box_type_eval = defaultdict(list)

    for data in tqdm(data_list, total=len(data_list)):
        label, pred = data['solution'], data['response']
        para = json.loads(data['additional_paras'])
        
        box_type = para['box_type']
        image_size = para['image_size']
        category = para.get('category', 'unknown')
        gui_types = para.get('GUI_types', ['unknown'])
        if isinstance(gui_types, str):
            gui_types = [gui_types]
        
        # Ground-truth coordinates come from solution['arguments']['coordinate'], as written by
        # convert2swift in os_worldg.py:
        #   - bbox: originally [x, y, w, h], already converted to [x1, y1, x2, y2]
        #   - polygon: the original flat list [x0, y0, ...]
        #   - refusal: the original value, unchanged
        gt_coordinate = label['arguments']['coordinate']

        # ====== Parse prediction ======
        if 'ground_qwen3' in metric:
            pred_action = extract_ground(pred)
        elif 'navi_qwen3' in metric:
            pred_action = extract_action(pred)
        elif 'venus' in metric:
            try:
                pred_action = eval(pred)
            except:
                pred_action = [0, 0]
        else:
            raise NotImplementedError

        # ====== Format errors ======
        if pred_action == "no action":
            num_wrong_format += 1
            # A format error is always wrong (it must not count as a correct refusal)
            is_correct = False
        else:
            # ====== Predicted point in pixel coordinates ======
            parse_failed = False
            try:
                if 'ground' in metric:
                    pred_point = calculate_pred_norm_point(image_size, pred_action, "qwen3vl")
                    pred_point = pointnorm2real(pred_point, image_size)
                elif 'navi_qwen3' in metric:
                    pred_point = calculate_pred_norm_point(image_size, pred_action['arguments']['coordinate'], "qwen3vl")
                    pred_point = pointnorm2real(pred_point, image_size)
                elif 'venus' in metric:
                    pred_point = calculate_pred_norm_point(image_size, pred_action, "qwen3vl")
                    pred_point = pointnorm2real(pred_point, image_size)
                else:
                    pred_point = [0, 0]
                    parse_failed = True
            except:
                pred_point = [0, 0]
                parse_failed = True

            if parse_failed:
                # Coordinate parsing failed: count as wrong
                is_correct = False
            else:
                # ====== Official scoring rule per box_type ======
                if box_type == "bbox":
                    # gt_coordinate is already [x1, y1, x2, y2] (converted by convert2swift),
                    # so no xywh conversion is needed here
                    is_correct = _is_point_in_rectangle(pred_point, gt_coordinate)
                
                elif box_type == "polygon":
                    is_correct = _is_point_in_polygon(pred_point, gt_coordinate)
                
                elif box_type == "refusal":
                    # Official rule: correct only if both predicted coordinates are negative;
                    # a valid point on a refusal sample is wrong
                    is_correct = all(pred_point[i] < 0 for i in range(2))
                
                else:
                    is_correct = False

        # ====== Tally ======
        if is_correct:
            corr_action += 1

        result_val = 1 if is_correct else 0
        category_eval[category].append(result_val)
        for gt in gui_types:
            gui_type_eval[gt].append(result_val)
        box_type_eval[box_type].append(result_val)

    # ====== Accuracy per category ======
    print("\n=== Category-wise Accuracy ===")
    category_metrics = {}
    
    # Print categories in a fixed order
    ordered_categories = [
        "Text Matching",
        "Element Recognition", 
        "Layout Understanding",
        "Fine-grained-Manipulation"
    ]
    
    for cat in ordered_categories:
        if cat in category_eval:
            results = category_eval[cat]
            acc = sum(results) / len(results) * 100 if results else 0.0
            category_metrics[cat] = f"{acc:.2f}"
            print(f"  {cat}: {acc:.2f}% ({sum(results)}/{len(results)})")
    
    # Then any remaining categories, alphabetically
    other_categories = [cat for cat in category_eval.keys() if cat not in ordered_categories]
    if other_categories:
        for cat in sorted(other_categories):
            results = category_eval[cat]
            acc = sum(results) / len(results) * 100 if results else 0.0
            category_metrics[cat] = f"{acc:.2f}"
            print(f"  {cat}: {acc:.2f}% ({sum(results)}/{len(results)})")

    # ====== Accuracy per GUI_types ======
    print("\n=== GUI-type-wise Accuracy ===")
    gui_metrics = {}
    for gt, results in sorted(gui_type_eval.items()):
        acc = sum(results) / len(results) * 100 if results else 0.0
        gui_metrics[gt] = f"{acc:.2f}"
        print(f"  {gt}: {acc:.2f}% ({sum(results)}/{len(results)})")

    # ====== Accuracy per box_type ======
    print("\n=== Box-type-wise Accuracy ===")
    box_metrics = {}
    for bt, results in sorted(box_type_eval.items()):
        acc = sum(results) / len(results) * 100 if results else 0.0
        box_metrics[bt] = f"{acc:.2f}"
        print(f"  {bt}: {acc:.2f}% ({sum(results)}/{len(results)})")

    # ====== Overall ======
    overall = {
        "Total Acc": f"{corr_action / num_action * 100:.2f}" if num_action > 0 else "0.00",
        "Total num": str(num_action),
        "Correct num": str(corr_action),
        "Wrong format num": str(num_wrong_format),
    }

    print("\n=== Overall ===")
    print(overall)

    return overall


