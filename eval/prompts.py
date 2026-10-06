import json
import os
import math
from typing import Dict, List, Any, Tuple, Optional
import re


def _official_gt_1000(gt_bbox: Dict, proc) -> Optional[Dict]:
    """Official GUI-Owl / UI-Venus scoring: scale an axis-aligned gt box from the
    eval's resized-pixel space into 0-1000 space and round OUTWARD (floor lows /
    ceil highs), matching eval_grounding_benchmarks.py (use_qwen3vl=True). The
    prediction is then compared as a raw 0-1000 point. Returns None if not an AABB.
    Enabled per-run via env GROUNDING_OFFICIAL_SCORING=1."""
    if not (proc and isinstance(gt_bbox, dict) and all(k in gt_bbox for k in ("x1", "y1", "x2", "y2"))):
        return None
    w, h = proc
    sx, sy = 1000.0 / w, 1000.0 / h
    return {
        "x1": math.floor(gt_bbox["x1"] * sx), "y1": math.floor(gt_bbox["y1"] * sy),
        "x2": math.ceil(gt_bbox["x2"] * sx),  "y2": math.ceil(gt_bbox["y2"] * sy),
    }


def extract_and_parse_json(input_string: str, wrapper: str) -> Optional[List]:
    """
    Attempt to extract and parse a JSON array from a string using a given pair of wrapper characters.
    
    The function searches for the first occurrence of the start wrapper and the last occurrence
    of the end wrapper, and tries to parse the substring between them as JSON.
    """
    if len(wrapper) != 2:
        raise ValueError("Wrapper must be exactly two characters long")

    start_char, end_char = wrapper
    start_index = input_string.find(start_char)
    end_index = input_string.rfind(end_char)

    if start_index == -1 or end_index == -1 or start_index >= end_index:
        return None

    json_string = input_string[start_index:end_index + 1]

    try:
        return json.loads(json_string)
    except json.JSONDecodeError:
        return None


def point_in_polygon(point: List[float], polygon: List[List[float]]) -> bool:
    """
    Ray casting algorithm to determine if a point lies inside a polygon.
    
    Args:
        point: Point coordinates as [x, y].
        polygon: List of polygon vertices [[x1, y1], [x2, y2], ...].
    
    Returns:
        True if the point is inside the polygon; False otherwise.
    """
    x, y = point
    n = len(polygon)
    inside = False
    
    p1x, p1y = polygon[0]
    for i in range(1, n + 1):
        p2x, p2y = polygon[i % n]
        if y > min(p1y, p2y):
            if y <= max(p1y, p2y):
                if x <= max(p1x, p2x):
                    if p1y != p2y:
                        xinters = (y - p1y) * (p2x - p1x) / (p2y - p1y) + p1x
                    if p1x == p2x or x <= xinters:
                        inside = not inside
        p1x, p1y = p2x, p2y
    
    return inside


# def is_point_in_polygon(point, polygon):
#     x, y = point
#     n = len(polygon) // 2
#     inside = False

#     j = n - 1
#     for i in range(n):
#         xi, yi = polygon[i * 2], polygon[i * 2 + 1]
#         xj, yj = polygon[j * 2], polygon[j * 2 + 1]

#         if (yi > y) != (yj > y) and x < (xj - xi) * (y - yi) / (yj - yi) + xi:
#             inside = not inside
#         j = i

#     return inside



def is_point_inside_element(element: Dict, point: List[float]) -> bool:
    """
    Check whether a predicted point lies inside a ground-truth region.
    
    Args:
        element: Dictionary describing the region, may contain a bbox or a polygon.
        point: Predicted point coordinates [x, y].
    
    Returns:
        True if the point is inside the element area; False otherwise.
    """
    # Axis-aligned bounding box
    if "x1" in element and "y1" in element and "x2" in element and "y2" in element:
        return (element["x1"] <= point[0] <= element["x2"] and 
                element["y1"] <= point[1] <= element["y2"])
    
    # Polygon
    elif "polygon" in element:
        polygon = element["polygon"]
        if len(polygon) < 3:  # A polygon requires at least 3 vertices
            return False
        return point_in_polygon(point, polygon)

    elif "refusal" in element:
        return all(point[i] < 0 for i in range(2))
    
    # Unsupported format
    else:
        return False


class BasicPrompt:
    
    @staticmethod
    def get_metric_keys() -> Dict[str, str]:
        """
        Return the metric keys supported by this processor and their types.
        
        Returns:
            Mapping from metric key to type, where type is one of:
            - 'sum': metrics to be summed
            - 'avg': metrics to be averaged (sum and count are handled separately)
            - 'count': count metrics
        """
        return {
            "total": "sum",
            "correct": "sum", 
            "has_correct": "sum",
            "num_answers": "avg",  # This is averaged across samples
        }
    
    @staticmethod
    def get_accuracy_pairs() -> List[Tuple[str, str]]:
        """
        Return numerator/denominator pairs for accuracy calculations.
        
        Returns:
            List of (numerator, denominator) pairs.
        """
        return [
            ("correct", "total"),       # correct_accuracy = correct / total
            ("has_correct", "total"),   # has_correct_accuracy = has_correct / total
        ]
  
    @staticmethod
    def generate_prompt(sample: Dict, image_width: int, image_height: int, think_mode: bool = True) -> List[Dict]:
        """
        Construct the message prompt for a single sample.
        """
        system_prompt = ('You are a helpful assistant.\n\n'
                        '# Tools\n\n'
                        'You may call one or more functions to assist with the user query.\n\n'
                        'You are provided with function signatures within <tools></tools> XML tags:\n'
                        '<tools>\n' 
                        '{{"type": "function", "function": {{"name": "computer_use", "description": "Use a mouse and keyboard to interact with a computer, and take screenshots.\n'
                        '* This is an interface to a desktop GUI. You do not have access to a terminal or applications menu. You must click on desktop icons to start applications.\n'
                        '* Some applications may take time to start or process actions, so you may need to wait and take successive screenshots to see the results of your actions. E.g. if you click on Firefox and a window doesn\'t open, try wait and taking another screenshot.\n'
                        '* The screen\'s resolution is {display_width_px}x{display_height_px}.\n* Whenever you intend to move the cursor to click on an element like an icon, you should consult a screenshot to determine the coordinates of the element before moving the cursor.\n'
                        '* If you tried clicking on a program or link but it failed to load, even after waiting, try adjusting your cursor position so that the tip of the cursor visually falls on the element that you want to click.\n'
                        '* Make sure to click any buttons, links, icons, etc with the cursor tip in the center of the element. Don\'t click boxes on their edges unless asked.", "parameters": {{"properties": {{"action": {{"description": "The action to perform. The available actions are:\n'
                        '* `key`: Performs key down presses on the arguments passed in order, then performs key releases in reverse order.\n'
                        '* `type`: Type a string of text on the keyboard.\n'
                        '* `mouse_move`: Move the cursor to a specified (x, y) pixel coordinate on the screen.\n'
                        '* `left_click`: Click the left mouse button.\n'
                        '* `left_click_drag`: Click and drag the cursor to a specified (x, y) pixel coordinate on the screen.\n'
                        '* `right_click`: Click the right mouse button.\n'
                        '* `middle_click`: Click the middle mouse button.\n'
                        '* `double_click`: Double-click the left mouse button.\n'
                        '* `scroll`: Performs a scroll of the mouse scroll wheel.\n'
                        '* `wait`: Wait specified seconds for the change to happen.\n'
                        '* `terminate`: Terminate the current task and report its completion status.", "enum": ["key", "type", "mouse_move", "left_click", "left_click_drag", "right_click", "middle_click", "double_click", "scroll", "wait", "terminate"], "type": "string"}}, "keys": {{"description": "Required only by `action=key`.", "type": "array"}}, "text": {{"description": "Required only by `action=type`.", "type": "string"}}, "coordinate": {{"description": "(x, y): The x (distance from the left edge) and y (distance from the top edge) coordinates to move the mouse to. Required only by `action=mouse_move` and `action=left_click_drag`.", "type": "array"}}, "pixels": {{"description": "The amount of scrolling to perform. Positive values scroll up, negative values scroll down. Required only by `action=scroll`.", "type": "number"}}, "time": {{"description": "The seconds to wait. Required only by `action=wait`.", "type": "number"}}, "status": {{"description": "The status of the task. Required only by `action=terminate`.", "type": "string", "enum": ["success", "failure"]}}, "required": ["action"], "type": "object"}}}}\n'
                        '</tools>\n\n'
                        'For each function call, return a json object with function name and arguments within <tool_call></tool_call> XML tags:\n'
                        '<tool_call>\n'
                        '{{"name": <function-name>, "arguments": <args-json-object>}}\n'
                        '</tool_call>\n')


            
        messages = [
            {
                "role": "system",
                "content": system_prompt.format(display_width_px=image_width, display_height_px=image_height)
            },
            {
                "role": "user",
                "content": f"<image>Hover over the UI element '{sample['instruction']}'"
            }
        ]
        return messages
    
    @staticmethod
    def extract_coordinates(response: str, think_mode: bool = False) -> Optional[List]:
        """
        Extract coordinates from the model response.
        """
        
        # result = extract_and_parse_json(response, "[]")
        json_str = response.split('<tool_call>')[1].split('</tool_call>')[0].strip()
        
        try:
            json_response = json.loads(json_str)
            click_point = json_response.get("arguments").get('coordinate')
            assert len(click_point) == 2
            click_point = [int(x) for x in click_point]
        except (json.JSONDecodeError, KeyError) as e:
            click_point = re.findall(r"\d+", json_str)
            
            if len(click_point) == 2:
                click_point = [int(x) for x in click_point]
            else:
                click_point = None


        return click_point
        
    @staticmethod
    def calculate_metrics(sample: Dict, point: Optional[List], gt_bbox: Dict) -> Dict[str, Any]:
        """
        Compute evaluation metrics for a single sample.
        
        Args:
            sample: Sample dictionary.
            predictions: List of predictions, format [{"point_2d": [x, y]}, ...].
            gt_bbox: Ground-truth bounding box, format {"x1": x1, "y1": y1, "x2": x2, "y2": y2} or {"polygon": [[x,y], ...]}.
        
        Returns:
            Dictionary containing metric values.
        """
        # Initialize metric dictionary dynamically based on metric_keys
        metric_keys = BasicPrompt.get_metric_keys()
        metrics = {}
        
        # Initialize base metrics
        for key, key_type in metric_keys.items():
            if key == "total":
                metrics[key] = 1  # Each sample counts as 1
            elif key_type in ["sum", "count"]:
                metrics[key] = 0
            elif key_type == "avg":
                metrics[key] = None
        
        # Add fixed field
        metrics["predictions"] = None
        
        # Handle samples without ground-truth (empty bbox)
        if not gt_bbox:
            if point is None:
                metrics["correct"] = 1
                metrics["has_correct"] = 1
            else:
                metrics["correct"] = 0
                metrics["has_correct"] = 0
            # Accuracies are computed later; no need to set here
            return metrics
        
        # No predictions
        if point is None:
            return metrics
        
        try:
            # Check if any predicted point is correct
            
            if is_point_inside_element(gt_bbox, point):
                metrics["correct"] = 1
                metrics["has_correct"] = 1
            
            metrics["predictions"] = point
            
        except (KeyError, IndexError, TypeError) as e:
            # Parsing error; return defaults in metrics
            pass
        
        # Accuracies are computed later; no need to set here
        return metrics
  

class InfiguiR1Prompt(BasicPrompt):
    """
    Default prompt processor for infigui-r1.
    """
    
    @staticmethod
    def generate_prompt(sample: Dict, image_width: int, image_height: int, think_mode: bool = True) -> List[Dict]:
        """
        Construct the message prompt for a single sample.
        """
        if think_mode:
            system_prompt = "You FIRST think about the reasoning process as an internal monologue and then provide the final answer.\nThe reasoning process MUST BE enclosed within <think> </think> tags."
        else:
            system_prompt = "You are a helpful assistant."

        grounding_prompt = f'''The screen\'s resolution is {image_width}x{image_height}.
Point to the UI element most relevant to "{sample['instruction']}", output its coordinates using JSON format:\n```json\n[\n    {{"point_2d": [x, y], "label": "object name/description"}}\n]```'''

        messages = [
            {
                "role": "system",
                "content": system_prompt
            },
            {
                "role": "user",
                "content": f'''<image>{grounding_prompt}'''
            }
        ]
        return messages
    
    @staticmethod
    def extract_coordinates(response: str, think_mode: bool = False) -> Optional[List[int]]:
        """
        Extract [x, y] coordinates from a model response containing JSON with "point_2d".
        Returns None if no valid coordinates are found.
        """
        try:
            # Try to parse JSON directly
            json_response = json.loads(response)
            if isinstance(json_response, list) and len(json_response) > 0:
                coords = json_response[0].get("point_2d")
                if isinstance(coords, (list, tuple)) and len(coords) == 2:
                    return [int(float(x)) for x in coords]
        except json.JSONDecodeError:
            pass

        # Fallback: try to extract numbers with regex
        numbers = re.findall(r"-?\d+", response.split("point_2d")[-1])
        if len(numbers) >= 2:
            return [int(numbers[0]), int(numbers[1])]

        return None

class InfiguiG1Prompt(BasicPrompt):
    """
    Default prompt processor for infigui-g1.
    """
    
    @staticmethod
    def generate_prompt(sample: Dict, image_width: int, image_height: int, think_mode: bool = True) -> List[Dict]:
        """
        Construct the message prompt for a single sample.
        """
        if think_mode:
            system_prompt = "You FIRST think about the reasoning process as an internal monologue and then provide the final answer.\nThe reasoning process MUST BE enclosed within <think> </think> tags."
        else:
            system_prompt = "You are a helpful assistant."
            
        messages = [
            {
                "role": "system",
                "content": system_prompt
            },
            {
                "role": "user",
                "content": f'''<image>The screen's resolution is {image_width}x{image_height}.
Locate the UI element(s) for "{sample['instruction']}", output the coordinates using JSON format: [{{"point_2d": [x, y]}}, ...]'''
            }
        ]
        return messages
    
    @staticmethod
    def extract_coordinates(response: str, think_mode: bool = False) -> Optional[List]:
        """
        Extract coordinates from the model response.
        """
        if think_mode and "</think>" in response:
            response = response.split("</think>")[-1]
        
        result = extract_and_parse_json(response, "[]")
        return result
    
    @staticmethod
    def calculate_metrics(sample: Dict, predictions: Optional[List], gt_bbox: Dict) -> Dict[str, Any]:
        """
        Compute evaluation metrics for a single sample.
        
        Args:
            sample: Sample dictionary.
            predictions: List of predictions, format [{"point_2d": [x, y]}, ...].
            gt_bbox: Ground-truth bounding box, format {"x1": x1, "y1": y1, "x2": x2, "y2": y2} or {"polygon": [[x,y], ...]}.
        
        Returns:
            Dictionary containing metric values.
        """
        # Initialize metric dictionary dynamically based on metric_keys
        metric_keys = InfiguiG1Prompt.get_metric_keys()
        metrics = {}
        
        # Initialize base metrics
        for key, key_type in metric_keys.items():
            if key == "total":
                metrics[key] = 1  # Each sample counts as 1
            elif key_type in ["sum", "count"]:
                metrics[key] = 0
            elif key_type == "avg":
                metrics[key] = None
        
        # Add fixed field
        metrics["predictions"] = None
        
        # Handle samples without ground-truth (empty bbox)
        if not gt_bbox:
            if predictions is None or len(predictions) == 0:
                metrics["correct"] = 1
                metrics["has_correct"] = 1
            else:
                metrics["correct"] = 0
                metrics["has_correct"] = 0
            # Accuracies are computed later; no need to set here
            return metrics
        
        # No predictions
        if predictions is None or len(predictions) == 0:
            return metrics
        
        try:
            # Check if any predicted point is correct
            has_correct = 0
            for pred in predictions:
                point = pred["point_2d"]
                if is_point_inside_element(gt_bbox, point):
                    has_correct = 1
                    break
            
            metrics["has_correct"] = has_correct
            
            # Check the first prediction
            first_pred = predictions[0]["point_2d"]
            if is_point_inside_element(gt_bbox, first_pred):
                metrics["correct"] = 1
            
            metrics["predictions"] = first_pred
            metrics["num_answers"] = sum(1 for pred in predictions if "point_2d" in pred)
            
        except (KeyError, IndexError, TypeError) as e:
            # Parsing error; return defaults in metrics
            pass
        
        # Accuracies are computed later; no need to set here
        return metrics

class GroundCUAPrompt(BasicPrompt):
    
    @staticmethod
    def generate_prompt(sample: Dict, image_width: int, image_height: int, think_mode: bool = True) -> List[Dict]:
        """
        Construct the message prompt for a single sample.
        """
        system_prompt = ('You are a helpful assistant.\n\n'
                        '# Tools\n\n'
                        'You may call one or more functions to assist with the user query.\n\n'
                        'You are provided with function signatures within <tools></tools> XML tags:\n'
                        '<tools>\n'
                        '{{"type": "function", "function": {{"name": "computer_use", "description": "Use a mouse and keyboard to interact with a computer, and take screenshots.\n'
                        '* This is an interface to a desktop GUI. You do not have access to a terminal or applications menu. You must click on desktop icons to start applications.\n'
                        '* Some applications may take time to start or process actions, so you may need to wait and take successive screenshots to see the results of your actions. E.g. if you click on Firefox and a window doesn\'t open, try wait and taking another screenshot.\n'
                        '* The screen\'s resolution is {display_width_px}x{display_height_px}.\n* Whenever you intend to move the cursor to click on an element like an icon, you should consult a screenshot to determine the coordinates of the element before moving the cursor.\n'
                        '* If you tried clicking on a program or link but it failed to load, even after waiting, try adjusting your cursor position so that the tip of the cursor visually falls on the element that you want to click.\n'
                        '* Make sure to click any buttons, links, icons, etc with the cursor tip in the center of the element. Don\'t click boxes on their edges unless asked.", "parameters": {{"properties": {{"action": {{"description": "The action to perform. The available actions are:\n'
                        '* `key`: Performs key down presses on the arguments passed in order, then performs key releases in reverse order.\n'
                        '* `type`: Type a string of text on the keyboard.\n'
                        '* `mouse_move`: Move the cursor to a specified (x, y) pixel coordinate on the screen.\n'
                        '* `left_click`: Click the left mouse button.\n'
                        '* `left_click_drag`: Click and drag the cursor to a specified (x, y) pixel coordinate on the screen.\n'
                        '* `right_click`: Click the right mouse button.\n'
                        '* `middle_click`: Click the middle mouse button.\n'
                        '* `double_click`: Double-click the left mouse button.\n'
                        '* `scroll`: Performs a scroll of the mouse scroll wheel.\n'
                        '* `wait`: Wait specified seconds for the change to happen.\n'
                        '* `terminate`: Terminate the current task and report its completion status.", "enum": ["key", "type", "mouse_move", "left_click", "left_click_drag", "right_click", "middle_click", "double_click", "scroll", "wait", "terminate"], "type": "string"}}, "keys": {{"description": "Required only by `action=key`.", "type": "array"}}, "text": {{"description": "Required only by `action=type`.", "type": "string"}}, "coordinate": {{"description": "(x, y): The x (distance from the left edge) and y (distance from the top edge) coordinates to move the mouse to. Required only by `action=mouse_move` and `action=left_click_drag`.", "type": "array"}}, "pixels": {{"description": "The amount of scrolling to perform. Positive values scroll up, negative values scroll down. Required only by `action=scroll`.", "type": "number"}}, "time": {{"description": "The seconds to wait. Required only by `action=wait`.", "type": "number"}}, "status": {{"description": "The status of the task. Required only by `action=terminate`.", "type": "string", "enum": ["success", "failure"]}}, "required": ["action"], "type": "object"}}}}\n'
                        '</tools>\n\n'
                        'For each function call, return a json object with function name and arguments within <tool_call></tool_call> XML tags:\n'
                        '<tool_call>\n'
                        '{{"name": <function-name>, "arguments": <args-json-object>}}\n'
                        '</tool_call>\n')

        messages = [
            {
                "role": "system",
                "content": system_prompt.format(display_width_px=image_width, display_height_px=image_height)
            },
            {
                "role": "user",
                "content": f"<image>{sample['instruction']}"
            }
        ]
        return messages
    
    @staticmethod
    def extract_coordinates(response: str, think_mode: bool = False) -> Optional[List]:
        """
        Extract coordinates from the model response.
        """
        
        # result = extract_and_parse_json(response, "[]")
        json_str = response.split('<tool_call>')[1].split('</tool_call>')[0].strip()
        
        try:
            json_response = json.loads(json_str)
            click_point = json_response.get("arguments").get('coordinate')
            assert len(click_point) == 2
            click_point = [int(x) for x in click_point]
        except (json.JSONDecodeError, KeyError) as e:
            click_point = re.findall(r"\d+", json_str)
            
            if len(click_point) == 2:
                click_point = [int(x) for x in click_point]
            else:
                click_point = None


        return click_point
   
class GTAPrompt(BasicPrompt):
    
    @staticmethod
    def generate_prompt(sample: Dict, image_width: int, image_height: int, think_mode: bool = True) -> List[Dict]:
        """
        Construct the message prompt for a single sample.
        """
        system_prompt = '''
You are an expert UI element locator. Given a GUI image and a user's element description, provide the coordinates of the specified element as a single (x,y) point. The image resolution is height {display_height_px} and width {display_width_px}. For elements with area, return the center point.

Output the coordinate pair exactly:
(x,y)
'''
            
        messages = [
            {
                "role": "system",
                "content": system_prompt.format(display_width_px=image_width, display_height_px=image_height)
            },
            {
                "role": "user",
                "content": f"<image>'{sample['instruction']}'"
            }
        ]
        return messages
    
    @staticmethod
    def extract_coordinates(response: str, think_mode: bool = False) -> Optional[List]:
        try:
            matches = re.findall(r"\((-?\d*\.?\d+),\s*(-?\d*\.?\d+)\)", response)
            click_point = [tuple(map(int, match)) for match in matches][0]
        except:
            click_point = None
            
        return click_point

class OpenCUAPrompt(BasicPrompt):
    
    @staticmethod
    def generate_prompt(sample: Dict, image_width: int, image_height: int, think_mode: bool = True) -> List[Dict]:
        """
        Construct the message prompt for a single sample.
        """
        system_prompt = (
            "You are a GUI agent. You are given a task and a screenshot of the screen. "
            "You need to perform a series of pyautogui actions to complete the task."
        )
            
        messages = [
            {
                "role": "system",
                "content": system_prompt.format(display_width_px=image_width, display_height_px=image_height)
            },
            {
                "role": "user",
                "content": f"<image>'{sample['instruction']}'"
            }
        ]
        return messages
    
    @staticmethod
    def extract_coordinates(response: str, think_mode: bool = False) -> Optional[List]:
        try:
            matches = re.findall(r"\((-?\d*\.?\d+),\s*(-?\d*\.?\d+)\)", response)
            click_point = [tuple(map(int, match)) for match in matches][0]
        except:
            click_point = None
            
        return click_point


class GUIG2Prompt(BasicPrompt):
    @staticmethod
    def generate_prompt(sample: Dict, image_width: int, image_height: int, think_mode: bool = True) -> List[Dict]:
        system_prompt = "Outline the position corresponding to the instruction: {problem}. The output should be only [x1,y1,x2,y2]."
        
        messages = [
            {
                "role": "user",
                "content": f"<image>'{system_prompt.format(problem=sample['instruction'])}'"
            }
        ]
        return messages
    
    @staticmethod
    def extract_coordinates(response: str, think_mode: bool = False) -> Optional[List]:
        bbox = None
        m = re.search(r"\[\s*(-?\d*\.?\d+)\s*,\s*(-?\d*\.?\d+)\s*,\s*(-?\d*\.?\d+)\s*,\s*(-?\d*\.?\d+)\s*\]", response)
        if m:
            bbox = [int(round(float(x))) for x in m.groups()]

        # 3) Last resort: grab any four numbers in order
        nums = re.findall(r"-?\d*\.?\d+", response)
        if len(nums) >= 4:
            bbox = [int(round(float(x))) for x in nums[:4]]

        if bbox is not None:
            click_point = [(bbox[0] + bbox[2]) // 2, (bbox[1] + bbox[3]) // 2]
        else:
            click_point = None
        return click_point
        
class UGroundV1Prompt(BasicPrompt):
    @staticmethod
    def generate_prompt(sample: Dict, image_width: int, image_height: int, think_mode: bool = True) -> List[Dict]:
        system_prompt = """
  Your task is to help the user identify the precise coordinates (x, y) of a specific area/element/object on the screen based on a description.

  - Your response should aim to point to the center or a representative point within the described area/element/object as accurately as possible.
  - If the description is unclear or ambiguous, infer the most relevant area or element based on its likely context or purpose.
  - Your answer should be a single string (x, y) corresponding to the point of the interest.

  Description: {description}

  Answer:"""
        
        messages = [
            {
                "role": "user",
                "content": f"<image>'{system_prompt.format(description=sample['instruction'])}'"
            }
        ]
        return messages
    
    @staticmethod
    def extract_coordinates(response: str, think_mode: bool = False) -> Optional[List]:
        try:
            matches = re.findall(r"\((-?\d*\.?\d+),\s*(-?\d*\.?\d+)\)", response)
            click_point = [tuple(map(int, match)) for match in matches][0]
        except:
            click_point = None
            
        return click_point
        
class SEGUIPrompt(BasicPrompt):
    @staticmethod
    def generate_prompt(sample: Dict, image_width: int, image_height: int, think_mode: bool = True) -> List[Dict]:
        system_prompt =  ('You are a helpful assistant.\n\n'
                        '# Tools\n\n'
                        'You may call one or more functions to assist with the user query.\n\n'
                        'You are provided with function signatures within <tools></tools> XML tags:\n'
                        '<tools>\n'
                        '{{"type": "function", "function": {{"name_for_human": "computer_use", "name": "computer_use", "description": "Use a mouse and keyboard to interact with a computer, and take screenshots.'
                        '* This is an interface to a desktop GUI. You do not have access to a terminal or applications menu . You must click on desktop icons to start applications. ' 
                        '* Some applications may take time to start or process actions, so you may need to wait and take successive screenshots to see the results of your actions.'
                        '* The screen\'s resolution is {image_width}x{image_height}.'
                        '* Whenever you intend to move the cursor to click on an element like an icon, you should consult a screenshot to determine the coordinates of the element before moving the cursor.}}}}'
                        '</tools>\n\n'
                        'For each function call, return a json object with function name and arguments within <tool_call></tool_call> XML tags:\n'
                        '<tool_call>\n'
                        '{{"name": <function-name>, "arguments": <args-json-object>}}\n'
                        '</tool_call>\n')
        messages = [
            {
                "role": "system",
                "content": system_prompt.format(image_width=image_width, image_height=image_height)
            },
            {
                "role": "user",
                "content": f"<image>'{sample['instruction']}'"
            }
        ]
        return messages
    
    @staticmethod
    def extract_coordinates(response: str, think_mode: bool = False) -> Optional[List]:
        """
        Extract coordinates from the model response.
        """
        
        # result = extract_and_parse_json(response, "[]")
        json_str = response.split('<tool_call>')[1].split('</tool_call>')[0].strip()
        
        try:
            json_response = json.loads(json_str)
            click_point = json_response.get("arguments").get('coordinate')
            assert len(click_point) == 2
            click_point = [int(x) for x in click_point]
        except (json.JSONDecodeError, KeyError) as e:
            click_point = re.findall(r"\d+", json_str)
            
            if len(click_point) == 2:
                click_point = [int(x) for x in click_point]
            else:
                click_point = None


        return click_point
        
class GUIG1Prompt(BasicPrompt):
    @staticmethod
    def generate_prompt(sample: Dict, image_width: int, image_height: int, think_mode: bool = True) -> List[Dict]:
        user_prompt =  'Grounding instruction is:{question}. Help to locate and output its bbox coordinates using JSON format::\n```json\n[\n{{"point_2d": [x, y], "label": "object name/description"}}\n]```'

        messages = [
            {
                "role": "user",
                "content": f"<image>'{user_prompt.format(question=sample['instruction'])}'"
            }
        ]
        return messages
    
    @staticmethod
    def extract_coordinates(response: str, think_mode: bool = False) -> Optional[List[int]]:
        """
        Extract [x, y] coordinates from a model response containing JSON with "point_2d".
        Returns None if no valid coordinates are found.
        """
        try:
            # Try to parse JSON directly
            json_response = json.loads(response)
            if isinstance(json_response, list) and len(json_response) > 0:
                coords = json_response[0].get("point_2d")
                if isinstance(coords, (list, tuple)) and len(coords) == 2:
                    return [int(float(x)) for x in coords]
        except json.JSONDecodeError:
            pass

        # Fallback: try to extract numbers with regex
        numbers = re.findall(r"-?\d+", response.split("point_2d")[-1])
        if len(numbers) >= 2:
            return [int(numbers[0]), int(numbers[1])]

        return None

        
class UITARSPrompt(BasicPrompt):
    @staticmethod
    def generate_prompt(sample: Dict, image_width: int, image_height: int, think_mode: bool = True) -> List[Dict]:
        user_prompt = """You are a GUI agent. You are given a task and your action history, with screenshots. You need to perform the next action to complete the task. \n\n## Output Format\n\nAction: ...\n\n\n## Action Space\nclick(point='<point>x1 y1</point>'')\n\n## User Instruction
{instruction}"""

        messages = [
            {
                "role": "user",
                "content": f"<image>'{user_prompt.format(instruction=sample['instruction'])}'"
            }
        ]
        return messages
    
    @staticmethod
    def extract_coordinates(response: str, think_mode: bool = False) -> Optional[List[int]]:
        # Try <point>x y</point>
        m = re.search(r"<point>\s*(\d+)\s+(\d+)\s*</point>", response)
        if m:
            return [int(m.group(1)), int(m.group(2))]

        # Try JSON { "coordinate": [x, y] }
        try:
            data = json.loads(re.search(r"\{.*\}", response).group(0))
            coords = data.get("coordinate") or data.get("point")
            if coords and len(coords) == 2:
                return [int(coords[0]), int(coords[1])]
        except:
            pass

        # Fallback: first two numbers
        nums = re.findall(r"\d+", response)
        return [int(nums[0]), int(nums[1])] if len(nums) >= 2 else None

class UIR1EPrompt(BasicPrompt):
    @staticmethod
    def generate_prompt(sample: Dict, image_width: int, image_height: int, think_mode: bool = True) -> List[Dict]:
        
        user_prompt = (
        "In this UI screenshot, I want to perform the command '{instruction}'.\n"
        "Please provide the action to perform (enumerate in ['click'])"
        "and the coordinate where the cursor is moved to(integer) if click is performed.\n"
        "Output the final answer in <answer> </answer> tags directly."
        "The output answer format should be as follows:\n"
        "<answer>[{{'action': 'click', 'coordinate': [x, y]}}]</answer>\n"
        "Please strictly follow the format."
    )


        messages = [
            {
                "role": "user",
                "content": f"<image>\n'{user_prompt.format(instruction=sample['instruction'])}'"
            }
        ]
        return messages
    
    @staticmethod
    def extract_coordinates(response: str, think_mode: bool = False) -> Optional[List[int]]:
        # Extract content inside <answer>...</answer>
        match = re.search(r"<answer>(.*?)</answer>", response, re.DOTALL)
        if not match:
            return None
        content = match.group(1).strip()

        # Try JSON/dict parsing
        try:
            # Ensure valid JSON (replace single quotes with double quotes)
            data = json.loads(content.replace("'", '"'))
            if isinstance(data, list) and "coordinate" in data[0]:
                coords = data[0]["coordinate"]
                if len(coords) == 2:
                    return [int(coords[0]), int(coords[1])]
        except:
            pass

        # Fallback: just grab two numbers
        nums = re.findall(r"\d+", content)
        return [int(nums[0]), int(nums[1])] if len(nums) >= 2 else None


class GUIActorPrompt(BasicPrompt):
    @staticmethod
    def generate_prompt(sample: Dict, image_width: int, image_height: int, think_mode: bool = True) -> List[Dict]:


        messages = [
            {
                "role": "system",
                "content": [
                    {
                        "type": "text",
                        "text": "You are a GUI agent. Given a screenshot of the current GUI and a human instruction, your task is to locate the screen element that corresponds to the instruction. You should output a PyAutoGUI action that performs a click on the correct position. To indicate the click location, we will use some special tokens, which is used to refer to a visual patch later. For example, you can output: pyautogui.click(<your_special_token_here>).",
                    }
                ]
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": sample['instruction']
                    },
                ],
            },
        ]

        return messages
    
    @staticmethod
    def extract_coordinates(response: str, think_mode: bool = False) -> Optional[List[int]]:
        # Extract content inside <answer>...</answer>
        click_point = json.loads(response)['predicted_coords']
        return click_point
        
class PhiGroundPrompt(BasicPrompt):
    @staticmethod
    def generate_prompt(sample: Dict, image_width: int, image_height: int, think_mode: bool = True) -> List[Dict]:

        system_prompt = """
The description of the element: 
{RE}


Locate the above described element in the image. The output should be bounding box using relative coordinates multiplying 1000.
"""

        messages = [

            {
                "role": "user",
                "content": system_prompt.format(RE=sample['instruction'])
            },
        ]

        return messages
    
    @staticmethod
    def extract_coordinates(response: str, think_mode: bool = False) -> Optional[List[int]]:
        try:
            pred = response.strip().split("<point>")[1].split("</point>")[0]
            coors = [float(o) / 1000 for o in pred.split(", ")]
            assert len(coors) == 2
        except:
            coors = None

        return coors

    
class ScaleCUAPrompt(BasicPrompt):
    """
    Prompt processor for ScaleCUA (e.g. OpenGVLab/ScaleCUA-3B / -7B).

    Uses the model card's "Direct Action Mode as grounder" setup
    (https://huggingface.co/OpenGVLab/ScaleCUA-3B): the model is given the
    pyautogui-style action space and emits an action inside <action></action>;
    for grounding we read the click(x=.., y=..) coordinate.

    Coordinates: ScaleCUA outputs absolute pixel coordinates in the (resized)
    image space, which already matches the bbox space the GroundCUA eval
    compares against, so no rescaling is needed here.

    NOTE on resolution: ScaleCUA is trained with a native max_pixels of
    ~2109744. Running it at eval.py's default max_image_pixels (12845056) is
    out-of-distribution and degrades grounding accuracy -- set
    MAX_IMAGE_PIXELS=2109744 (read by eval.py) when evaluating ScaleCUA.
    """

    SYSTEM_PROMPT = '''You are an autonomous GUI agent capable of operating on desktops, mobile devices, and web browsers. Your primary function is to analyze screen captures and perform appropriate UI actions to complete assigned tasks.

## Action Space
def click(
    x: float | None = None,
    y: float | None = None,
    clicks: int = 1,
    button: str = "left",
) -> None:
    """Clicks on the screen at the specified coordinates. The `x` and `y` parameter specify where the mouse event occurs. If not provided, the current mouse position is used. The `clicks` parameter specifies how many times to click, and the `button` parameter specifies which mouse button to use ('left', 'right', or 'middle')."""
    pass

def doubleClick(
    x: float | None = None,
    y: float | None = None,
    button: str = "left",
) -> None:
    """Performs a double click. This is a wrapper function for click(x, y, 2, 'left')."""
    pass

def rightClick(x: float | None = None, y: float | None = None) -> None:
    """Performs a right mouse button click. This is a wrapper function for click(x, y, 1, 'right')."""
    pass

def moveTo(x: float, y: float) -> None:
    """Move the mouse to the specified coordinates."""
    pass

def dragTo(
    x: float | None = None, y: float | None = None, button: str = "left"
) -> None:
    """Performs a drag-to action with optional `x` and `y` coordinates and button."""
    pass

def swipe(
    from_coord: tuple[float, float] | None = None,
    to_coord: tuple[float, float] | None = None,
    direction: str = "up",
    amount: float = 0.5,
) -> None:
    """Performs a swipe action on the screen. The `from_coord` and `to_coord` specify the starting and ending coordinates of the swipe. If `to_coord` is not provided, the `direction` and `amount` parameters are used to determine the swipe direction and distance. The `direction` can be 'up', 'down', 'left', or 'right', and the `amount` specifies how far to swipe relative to the screen size (0 to 1)."""
    pass

def long_press(x: float, y: float, duration: int = 1) -> None:
    """Long press on the screen at the specified coordinates. The `duration` specifies how long to hold the press in seconds."""
    pass

## Input Specification
- Screenshot of the current screen + task description

## Output Format
<action>
[A set of executable action command]
</action>

## Note
- Avoid action(s) that would lead to invalid states.
- The generated action(s) must exist within the defined action space.
- The generated action(s) should be enclosed within <action></action> tags.'''

    @staticmethod
    def generate_prompt(sample: Dict, image_width: int, image_height: int, think_mode: bool = True) -> List[Dict]:
        """Construct the message prompt for a single sample (grounder / direct action mode)."""
        messages = [
            {
                "role": "system",
                "content": ScaleCUAPrompt.SYSTEM_PROMPT,
            },
            {
                "role": "user",
                "content": f"<image>{sample['instruction']}",
            },
        ]
        return messages

    @staticmethod
    def extract_coordinates(response: str, think_mode: bool = False) -> Optional[List]:
        """
        Parse the click coordinate from a ScaleCUA action such as
        `<action> click(x=150, y=200) </action>`. Returns absolute [x, y] in the
        (resized) image space, or None if no coordinate can be parsed.
        """
        text = response
        if "<action>" in text and "</action>" in text:
            text = text.split("<action>")[1].split("</action>")[0]

        # Preferred: keyword form `x=.., y=..` (click/doubleClick/rightClick/moveTo/dragTo)
        m = re.search(r"x\s*=\s*([-+]?\d*\.?\d+)\s*,\s*y\s*=\s*([-+]?\d*\.?\d+)", text)
        if m is None:
            # Fallback: positional form `(x, y)`
            m = re.search(r"\(\s*([-+]?\d*\.?\d+)\s*,\s*([-+]?\d*\.?\d+)\s*\)", text)
        if m is None:
            return None

        try:
            click_point = [int(round(float(m.group(1)))), int(round(float(m.group(2))))]
        except (ValueError, TypeError):
            return None

        return click_point


class Qwen2_5_VLPrompt(BasicPrompt):
    """
    Prompt processor for the *original* Qwen2.5-VL-Instruct models
    (e.g. Qwen/Qwen2.5-VL-3B-Instruct), for GUI element grounding.

    Qwen2.5-VL's native grounding emits absolute-pixel bounding boxes in the
    (resized) image space, as JSON `{"bbox_2d": [x1, y1, x2, y2], ...}`. We ask
    for that box and return its center as the click point. The coordinates are
    already in the eval's resized-image / gt-bbox space (the eval pre-resizes
    the image AND the bbox with the same max_pixels), so no rescaling is needed.

    Resolution: uses the model's native max_pixels (Qwen2.5-VL default 12845056,
    == eval.py's default), so no MAX_IMAGE_PIXELS override is required.
    """

    @staticmethod
    def generate_prompt(sample: Dict, image_width: int, image_height: int, think_mode: bool = True) -> List[Dict]:
        instruction = sample["instruction"]
        user = (
            f"<image>\nLocate the UI element for the instruction: \"{instruction}\". "
            f"Output only its bounding box as JSON in the format "
            f"{{\"bbox_2d\": [x1, y1, x2, y2]}}."
        )
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": user},
        ]
        return messages

    @staticmethod
    def extract_coordinates(response: str, think_mode: bool = False) -> Optional[List]:
        """
        Return the click point [x, y] (center of the predicted box). Robust to
        the Qwen native JSON bbox, a bare [x1,y1,x2,y2], the (x1,y1),(x2,y2)
        box-token form, or a single (x, y) point.
        """
        def center4(a, b, c, d):
            return [int(round((a + c) / 2)), int(round((b + d) / 2))]

        # 1) Qwen native JSON: {"bbox_2d": [x1,y1,x2,y2], ...}
        m = re.search(r'"bbox_2d"\s*:\s*\[\s*(-?\d*\.?\d+)\s*,\s*(-?\d*\.?\d+)\s*,\s*(-?\d*\.?\d+)\s*,\s*(-?\d*\.?\d+)\s*\]', response)
        if m:
            return center4(*[float(g) for g in m.groups()])

        # 2) (x1,y1),(x2,y2)  box-token form -> center
        m = re.search(r'\(\s*(-?\d*\.?\d+)\s*,\s*(-?\d*\.?\d+)\s*\)\s*,\s*\(\s*(-?\d*\.?\d+)\s*,\s*(-?\d*\.?\d+)\s*\)', response)
        if m:
            return center4(*[float(g) for g in m.groups()])

        # 3) bare [x1,y1,x2,y2] -> center
        m = re.search(r'\[\s*(-?\d*\.?\d+)\s*,\s*(-?\d*\.?\d+)\s*,\s*(-?\d*\.?\d+)\s*,\s*(-?\d*\.?\d+)\s*\]', response)
        if m:
            return center4(*[float(g) for g in m.groups()])

        # 4) single point (x, y)
        m = re.search(r'\(\s*(-?\d*\.?\d+)\s*,\s*(-?\d*\.?\d+)\s*\)', response)
        if m:
            return [int(round(float(m.group(1)))), int(round(float(m.group(2))))]

        # 5) last resort: first 4 (-> center) or 2 numbers
        nums = re.findall(r'-?\d*\.?\d+', response)
        if len(nums) >= 4:
            return center4(*[float(n) for n in nums[:4]])
        if len(nums) >= 2:
            return [int(round(float(nums[0]))), int(round(float(nums[1])))]
        return None


class GPTPrompt(BasicPrompt):
    """
    Prompt processor for OpenAI GPT models (gpt-4o / gpt-5 family) used as a
    zero-shot GUI grounder on ScreenSpot-style benchmarks.

    We ask the model for coordinates NORMALIZED to [0, 1] (fraction of width /
    height) rather than absolute pixels. GPT vision internally downsamples the
    image (esp. high-res desktop/web screenshots), so it cannot reliably report
    absolute pixel positions -- but relative position is fine. `eval.py`'s
    `process_response` detects a normalized point (0 < x,y < 1) and multiplies by
    the resized-image size, mapping it into the gt-bbox space. If the model
    ignores the instruction and emits pixels (> 1), they pass through as
    absolute pixels, which is a reasonable fallback.
    """

    @staticmethod
    def generate_prompt(sample: Dict, image_width: int, image_height: int, think_mode: bool = True) -> List[Dict]:
        instruction = sample["instruction"]
        user = (
            f"<image>\n"
            f"Locate the single UI element on this screenshot that best matches "
            f"the instruction:\n\"{instruction}\"\n"
            f"Reply with ONLY the click point as JSON on one line, using "
            f"coordinates NORMALIZED to the image size -- each a decimal fraction "
            f"in the range 0.0 to 1.0, NOT pixel values: "
            f"{{\"x\": <float>, \"y\": <float>}}. x is the horizontal fraction from "
            f"the left edge, y is the vertical fraction from the top edge. "
            f"For example the exact center is {{\"x\": 0.5, \"y\": 0.5}} and the "
            f"bottom-right corner is {{\"x\": 1.0, \"y\": 1.0}}. Both values MUST be "
            f"between 0 and 1. Use 4 decimal places. Output nothing else."
        )
        messages = [
            {"role": "system", "content": "You are a precise GUI grounding assistant that locates on-screen UI elements."},
            {"role": "user", "content": user},
        ]
        return messages

    @staticmethod
    def extract_coordinates(response: str, think_mode: bool = False) -> Optional[List]:
        """
        Return the click point [x, y] as normalized floats in (0, 1). eval.py
        scales these by the resized-image size. Values > 1 (if the model emits
        pixels anyway) pass through unchanged as an absolute-pixel fallback.
        """
        if not response:
            return None

        x = y = None
        # 1) JSON {"x": .., "y": ..} (order-insensitive)
        mx = re.search(r'"x"\s*:\s*(-?\d*\.?\d+)', response)
        my = re.search(r'"y"\s*:\s*(-?\d*\.?\d+)', response)
        if mx and my:
            x, y = float(mx.group(1)), float(my.group(1))
        else:
            # 2) (x, y) or [x, y]
            m = re.search(r'[\(\[]\s*(-?\d*\.?\d+)\s*,\s*(-?\d*\.?\d+)\s*[\)\]]', response)
            if m:
                x, y = float(m.group(1)), float(m.group(2))
            else:
                # 3) last resort: first two numbers
                nums = re.findall(r'-?\d*\.?\d+', response)
                if len(nums) >= 2:
                    x, y = float(nums[0]), float(nums[1])
                else:
                    return None

        # Clamp normalized coords into the OPEN interval (0, 1) so eval.py's
        # `0 < v < 1` scaling path always fires (handles elements at the very
        # edge where the model returns exactly 0.0 or 1.0). Pixel values (>1)
        # are left untouched.
        def _clip(v: float) -> float:
            if 0.0 <= v <= 1.0:
                return min(max(v, 1e-4), 1.0 - 1e-4)
            return v

        return [_clip(x), _clip(y)]


class GPTRefusalPrompt(BasicPrompt):
    """
    GPT grounding prompt that ALLOWS refusal -- for OSWorld-G, which contains
    "element absent" cases scored correct only when the model abstains. Same
    normalized-coordinate scheme as GPTPrompt, but the model may instead reply
    NOT_FOUND, which we map to a negative point that the eval's refusal check
    scores as a correct abstention.
    """

    @staticmethod
    def generate_prompt(sample: Dict, image_width: int, image_height: int, think_mode: bool = True) -> List[Dict]:
        instruction = sample["instruction"]
        user = (
            f"<image>\n"
            f"Locate the single UI element on this screenshot that best matches "
            f"the instruction:\n\"{instruction}\"\n"
            f"If that element IS present, reply with ONLY the click point as JSON "
            f"on one line, NORMALIZED to [0,1]: {{\"x\": <float>, \"y\": <float>}} "
            f"(4 decimals; the image center is {{\"x\": 0.5, \"y\": 0.5}}). "
            f"If the described element does NOT exist anywhere in this screenshot, "
            f"reply with exactly: NOT_FOUND. Output nothing else."
        )
        return [
            {"role": "system", "content": "You are a precise GUI grounding assistant that locates on-screen UI elements."},
            {"role": "user", "content": user},
        ]

    @staticmethod
    def extract_coordinates(response: str, think_mode: bool = False) -> Optional[List]:
        if not response:
            return None
        if re.search(r"NOT[_\s]?FOUND", response, re.I):
            return [-1.0, -1.0]  # model abstained -> refusal
        coords = GPTPrompt.extract_coordinates(response, think_mode)
        if coords is None:
            return [-1.0, -1.0]  # no parseable coordinate -> treat as refusal
        return coords


class ClaudePrompt(BasicPrompt):
    """
    Prompt processor for Anthropic Claude models used as a zero-shot GUI
    grounder on ScreenSpot-style benchmarks.

    We ask Claude for coordinates NORMALIZED to [0, 1] (fraction of width /
    height) rather than absolute pixels. Claude *does* emit absolute pixels
    natively, but in the resolution of the image it actually sees -- and
    Anthropic silently downscales any screenshot above ~1568px on the long edge
    / ~1.15MP, so pixel output on large (mobile / web) screenshots lands in that
    hidden downscaled space and mismatches the gt-bbox space. A normalized
    fraction is resolution-invariant, so it stays correct regardless of
    downscaling; `eval.py`'s `0 < v < 1` path multiplies it by the resized-image
    size. The Anthropic backend additionally pre-fits the image to Anthropic's
    window and converts any stray pixel output to a fraction using the exact
    dimensions it sent, so a non-compliant pixel answer is still recoverable.
    """

    @staticmethod
    def generate_prompt(sample: Dict, image_width: int, image_height: int, think_mode: bool = True) -> List[Dict]:
        instruction = sample["instruction"]
        user = (
            f"<image>\n"
            f"Locate the single UI element on this screenshot that best matches "
            f"the instruction:\n\"{instruction}\"\n"
            f"Reply with ONLY the click point as JSON on one line, using "
            f"coordinates NORMALIZED to the image size -- each a decimal fraction "
            f"between 0.0 and 1.0 (NOT pixel values). x is the fraction from the "
            f"left edge, y the fraction from the top edge; the image center is "
            f"{{\"x\": 0.5, \"y\": 0.5}}. Format: {{\"x\": <float>, \"y\": <float>}}. "
            f"Use 4 decimal places. Output nothing else."
        )
        messages = [
            {"role": "system", "content": "You are a precise GUI grounding assistant that locates on-screen UI elements."},
            {"role": "user", "content": user},
        ]
        return messages

    @staticmethod
    def extract_coordinates(response: str, think_mode: bool = False) -> Optional[List]:
        """
        Return the click point [x, y] as normalized floats in (0, 1). The
        Anthropic backend emits canonical normalized JSON, so values are already
        fractions; eval.py scales them by the resized-image size.
        """
        if not response:
            return None
        x = y = None
        mx = re.search(r'"x"\s*:\s*(-?\d*\.?\d+)', response)
        my = re.search(r'"y"\s*:\s*(-?\d*\.?\d+)', response)
        if mx and my:
            x, y = float(mx.group(1)), float(my.group(1))
        else:
            m = re.search(r'[\(\[]\s*(-?\d*\.?\d+)\s*,\s*(-?\d*\.?\d+)\s*[\)\]]', response)
            if m:
                x, y = float(m.group(1)), float(m.group(2))
            else:
                nums = re.findall(r'-?\d*\.?\d+', response)
                if len(nums) >= 2:
                    x, y = float(nums[0]), float(nums[1])
                else:
                    return None

        def _clip(v: float) -> float:
            if 0.0 <= v <= 1.0:
                return min(max(v, 1e-4), 1.0 - 1e-4)
            return v

        return [_clip(x), _clip(y)]


class ClaudeCUAPrompt(BasicPrompt):
    """
    Prompt processor for Anthropic Claude used with its **computer-use tool**
    for GUI grounding -- the protocol that matches published numbers
    (Claude 3.7 Sonnet = 87.6 on ScreenSpot-V2, Mobile-Agent-v3.5 Table 7).

    The computer-use tool is RLHF-tuned for precise clicking, but on its own the
    model often responds with a screenshot request, a clarifying question, or an
    explanation instead of a click (~40% of the time), and every non-click is a
    miss. The system prompt below forces an immediate single left_click. The
    Anthropic backend (computer_use mode) resizes the image to Anthropic's
    ~1280px window, attaches the `computer_20250124` tool at the sent
    resolution, parses the tool's click coordinate, and returns it as a
    normalized fraction; eval.py then scales by the resized-image size.
    """

    SYSTEM = (
        "You are a GUI grounding tool. The screenshot is ALREADY provided in this "
        "message -- never ask for or take another screenshot. You are given an "
        "instruction naming one on-screen element. Immediately call the computer "
        "tool with a single left_click at the center of the described element. "
        "Always output your single best-guess click even if you are unsure or the "
        "element is hard to see. Never refuse, never say you cannot find or see "
        "it, never ask a question, never explain -- just left_click."
    )

    @staticmethod
    def generate_prompt(sample: Dict, image_width: int, image_height: int, think_mode: bool = True) -> List[Dict]:
        return [
            {"role": "system", "content": ClaudeCUAPrompt.SYSTEM},
            {"role": "user", "content": f"<image>\nClick on: {sample['instruction']}"},
        ]

    @staticmethod
    def extract_coordinates(response: str, think_mode: bool = False) -> Optional[List]:
        """Backend returns canonical normalized JSON; read it back as [x, y] in (0,1)."""
        return ClaudePrompt.extract_coordinates(response, think_mode)


class ClaudeCUARefusalPrompt(BasicPrompt):
    """
    Computer-use grounding prompt that ALLOWS refusal -- for benchmarks like
    OSWorld-G that include "element absent" cases (scored correct only when the
    model abstains, i.e. predicts a negative point). Unlike ClaudeCUAPrompt this
    does NOT force a click: the model clicks a present element, or replies
    NOT_FOUND when the element is genuinely absent. The Anthropic backend
    (allow_refusal=True) maps a no-click to a negative sentinel point.
    """

    SYSTEM = (
        "You are a GUI grounding tool. The screenshot is ALREADY provided -- never "
        "ask for or take another screenshot. You are given an instruction naming "
        "one on-screen element. If that element IS present in the screenshot, "
        "immediately call the computer tool with a single left_click at its "
        "center. ONLY if the described element genuinely does not exist anywhere "
        "in the screenshot, do not click -- instead reply with exactly the text "
        "NOT_FOUND. Strongly prefer clicking; refuse only when you are confident "
        "the element is truly absent. Never ask questions, never explain."
    )

    @staticmethod
    def generate_prompt(sample: Dict, image_width: int, image_height: int, think_mode: bool = True) -> List[Dict]:
        return [
            {"role": "system", "content": ClaudeCUARefusalPrompt.SYSTEM},
            {"role": "user", "content": f"<image>\nClick on: {sample['instruction']}"},
        ]

    @staticmethod
    def extract_coordinates(response: str, think_mode: bool = False) -> Optional[List]:
        return ClaudePrompt.extract_coordinates(response, think_mode)


class GUIOwlPrompt(BasicPrompt):
    """
    Prompt processor for GUI-Owl-1.5 (X-PLUG/MobileAgent, Mobile-Agent-v3.5),
    e.g. a local copy of GUI-Owl-1.5-32B-Instruct. It is a Qwen3-VL model.

    Faithfully mirrors the official grounding eval
    (Mobile-Agent-v3.5/grounding_and_kb/eval_grounding_benchmarks.py):
      * the system message carries a `computer_use` tool schema that PINS the
        screen resolution at 1000x1000, so the model emits coordinates in a
        normalized 0-1000 space, independent of the real image size;
      * the model answers with <tool_call>{... "coordinate": [x, y]}</tool_call>
        (left_click / mouse_move), with 0 <= x, y <= 1000;
      * we take the LAST (x, y) pair (official regex order: "(x, y)" then
        "[x, y]") and, in calculate_metrics, map it from the 0-1000 space into
        the eval's resized-image pixel space via sample["processed_imgsize"], so
        the point lines up with the (same-space) resized gt box.

    We convert in calculate_metrics -- NOT by returning a 0..1 fraction and
    leaning on eval.py's "0<pred<1" auto-rescale -- because that heuristic silently
    fails for edge clicks (x==0) and never fires once a coord is >=1.

    Refusal (OSWorld-G): set GUIOWL_ALLOW_REFUSAL=1 to append the infeasible-task
    prefix; a terminate/failure reply (no coordinate) becomes the sentinel
    [-1, -1], which is_point_inside_element only accepts for a refusal-type gt.

    Resolution: GUI-Owl-1.5 grounds at up to 9800 vision tokens
    (9800 * (patch16 * merge2)^2 = 10,035,200 px). Set MAX_IMAGE_PIXELS=10035200
    when evaluating it.
    """

    # Verbatim from Mobile-Agent-v3.5 eval_grounding_benchmarks.py
    # (only_two_action_system_prompt). The <tools> block is fed as plain system
    # text -- matching the official code and the Qwen3-VL no-`tools` template path.
    SYSTEM_PROMPT = (
        '# Tools\n\nYou may call one or more functions to assist with the user query.\n\n'
        'You are provided with function signatures within <tools></tools> XML tags:\n<tools>\n'
        '{"type": "function", "function": {"name": "computer_use", "description": "Use a mouse to interact with a computer.\n'
        "* The screen's resolution is 1000x1000.\n"
        "* Make sure to click any buttons, links, icons, etc with the cursor tip in the center of the element. Don't click boxes on their edges unless asked.\n"
        "* don't use any other computer use tool like type, key, scroll, left_click_drag and so on.\n"
        "* you can only use the left_click and mouse_move action to interact with the computer. if you can't find the element, you should terminate the task and report the failure.\", "
        '"parameters": {"properties": {"action": {"description": "The action to perform. The available actions are:\n'
        '* `mouse_move`: Move the cursor to a specified (x, y) pixel coordinate on the screen.\n'
        '* `left_click`: Click the left mouse button with coordinate (x, y) pixel coordinate on the screen.", '
        '"enum": ["mouse_move", "left_click"], "type": "string"}, '
        '"coordinate": {"description": "(x, y): The x (pixels from the left edge) and y (pixels from the top edge) coordinates to move the mouse to. '
        'Required only by `action=mouse_move` and `action=left_click`.", "type": "array"}}, "required": ["action"], "type": "object"}}}\n'
        '</tools>\n\n'
        'For each function call, return a json object with function name and arguments within <tool_call></tool_call> XML tags:\n'
        '<tool_call>\n{"name": <function-name>, "arguments": <args-json-object>}\n</tool_call>\n'
    )

    INFEASIBLE_PREFIX = (
        'Additionally, if you think the task is infeasible (e.g., the task is not related to the image), '
        'return <tool_call>\n{"name": "computer_use", "arguments": {"action": "terminate", "status": "failure"}}\n</tool_call>'
    )

    @staticmethod
    def generate_prompt(sample: Dict, image_width: int, image_height: int, think_mode: bool = True) -> List[Dict]:
        system_text = GUIOwlPrompt.SYSTEM_PROMPT
        if os.environ.get("GUIOWL_ALLOW_REFUSAL") == "1":
            system_text = system_text + "\n" + GUIOwlPrompt.INFEASIBLE_PREFIX
        return [
            {"role": "system", "content": system_text},
            {"role": "user", "content": f"<image>{sample['instruction']}"},
        ]

    @staticmethod
    def extract_coordinates(response: str, think_mode: bool = False) -> Optional[List]:
        # Explicit refusal -> refusal sentinel. terminate/failure is the form used by the
        # *_refusal_grounding_v1 training templates; abstain is the form used by
        # *_abstain_grounding_v1 (no arguments, lexically separate from the task-level
        # terminate). Both schemas coexist, so both are accepted.
        if ('"action": "terminate"' in response or '"status": "failure"' in response
                or '"action": "abstain"' in response):
            return [-1, -1]
        # Official parse order: "(x, y)" first, then "[x, y]"; take the LAST pair.
        matches = re.findall(r"\((\d+),\s*(\d+)\)", response)
        if not matches:
            matches = re.findall(r"\[(\d+),\s*(\d+)\]", response)
        if not matches:
            return None
        x, y = matches[-1]
        return [int(x), int(y)]

    @staticmethod
    def calculate_metrics(sample: Dict, point: Optional[List], gt_bbox: Dict) -> Dict[str, Any]:
        # A real 0-1000 point (>=0) is scored either in the official 0-1000 space
        # (GROUNDING_OFFICIAL_SCORING=1: gt rounded outward, point kept raw) or, by
        # default, mapped to the eval's resized-image pixel space for exact float
        # containment. None and the refusal sentinel [-1,-1] pass through untouched.
        if point is not None and point[0] >= 0 and point[1] >= 0:
            proc = sample.get("processed_imgsize")
            if proc:
                if os.environ.get("GROUNDING_OFFICIAL_SCORING") == "1":
                    gt1000 = _official_gt_1000(gt_bbox, proc)
                    if gt1000 is not None:
                        gt_bbox = gt1000            # keep `point` raw 0-1000
                    else:
                        w, h = proc
                        point = [point[0] / 1000.0 * w, point[1] / 1000.0 * h]
                else:
                    w, h = proc
                    point = [point[0] / 1000.0 * w, point[1] / 1000.0 * h]
        return BasicPrompt.calculate_metrics(sample, point, gt_bbox)


class UIVenusPrompt(BasicPrompt):
    """
    Prompt processor for UI-Venus-1.5 (inclusionAI/UI-Venus), e.g.
    a local copy of UI-Venus-1.5-30B-A3B. It is a Qwen3-VL-MoE model.

    Faithfully mirrors the official grounding code
    (UI-Venus repo, models/grounding/ui_venus1_5_gd.py & eval_screenspot_pro.py):
      * NO system prompt -- a single user turn: image + a plain instruction
        asking for the element's CENTER POINT as "[x, y]";
      * the model answers in a 0-1000 normalized space (a point "[x, y]", a box
        "[x1, y1, x2, y2]" whose center we take, or two points averaged);
      * "[-1, -1]" means the task is infeasible (refusal); the refusal clause is
        part of the prompt by default, matching the official eval;
      * extract_coordinates returns the raw 0-1000 point (or [-1,-1] / None) and
        calculate_metrics maps it into the eval's resized-image pixel space via
        sample["processed_imgsize"] (official divides by 1000 then x original
        size; we x resized size, since the gt box is resized to that same space).

    Note: the trailing '.' of the instruction is stripped, exactly as upstream.
    dtype: config.json top-level dtype is float16 but the model is bf16-native
    (the official loader uses torch.bfloat16); set VLLM_DTYPE=bfloat16 to avoid
    fp16 overflow -> NaN.
    """

    # Verbatim from ui_venus1_5_gd.py (refusal-enabled variant; the official eval
    # calls inference() with the default do_not_use_refusal=False).
    PROMPT = (
        "Output the center point of the position corresponding to the following instruction: \n{instruction}. "
        "\n\nThe output should just be the coordinates of a point, in the format [x,y]. "
        "Additionally, if the task is infeasible (e.g., the task is not related to the image), "
        "the output should be [-1,-1]."
    )

    @staticmethod
    def generate_prompt(sample: Dict, image_width: int, image_height: int, think_mode: bool = True) -> List[Dict]:
        instruction = sample["instruction"]
        if instruction.endswith("."):
            instruction = instruction[:-1]
        # Single user turn, no system message (matches upstream apply_chat_template call).
        return [
            {"role": "user", "content": "<image>" + UIVenusPrompt.PROMPT.format(instruction=instruction)},
        ]

    @staticmethod
    def extract_coordinates(response: str, think_mode: bool = False) -> Optional[List]:
        """
        Return the raw 0-1000 click point [x, y]; [-1, -1] for an infeasible/refusal
        answer; None if nothing parses. Handles the upstream output variants: a bare
        point, a 4-number box (center), and two points (averaged).
        """
        text = response.strip()
        # Refusal / infeasible.
        if re.search(r"\[\s*-1\s*,\s*-1\s*\]", text):
            return [-1, -1]
        # Two points "[x1,y1],[x2,y2]" -> center.
        m2 = re.search(
            r"\[\s*(-?\d+)\s*,\s*(-?\d+)\s*\]\s*,\s*\[\s*(-?\d+)\s*,\s*(-?\d+)\s*\]", text
        )
        if m2:
            a, b, c, d = (int(v) for v in m2.groups())
            return [(a + c) / 2.0, (b + d) / 2.0]
        # Box "[x1,y1,x2,y2]" -> center.
        m4 = re.search(
            r"\[\s*(-?\d+)\s*,\s*(-?\d+)\s*,\s*(-?\d+)\s*,\s*(-?\d+)\s*\]", text
        )
        if m4:
            x1, y1, x2, y2 = (int(v) for v in m4.groups())
            return [(x1 + x2) / 2.0, (y1 + y2) / 2.0]
        # Point "[x,y]".
        m1 = re.search(r"\[\s*(-?\d+)\s*,\s*(-?\d+)\s*\]", text)
        if m1:
            x, y = (int(v) for v in m1.groups())
            return [float(x), float(y)]
        return None

    @staticmethod
    def calculate_metrics(sample: Dict, point: Optional[List], gt_bbox: Dict) -> Dict[str, Any]:
        # Official 0-1000 scoring (GROUNDING_OFFICIAL_SCORING=1) or default exact
        # containment in resized-pixel space; None / refusal sentinel pass through.
        if point is not None and point[0] >= 0 and point[1] >= 0:
            proc = sample.get("processed_imgsize")
            if proc:
                if os.environ.get("GROUNDING_OFFICIAL_SCORING") == "1":
                    gt1000 = _official_gt_1000(gt_bbox, proc)
                    if gt1000 is not None:
                        gt_bbox = gt1000            # keep `point` raw 0-1000
                    else:
                        w, h = proc
                        point = [point[0] / 1000.0 * w, point[1] / 1000.0 * h]
                else:
                    w, h = proc
                    point = [point[0] / 1000.0 * w, point[1] / 1000.0 * h]
        return BasicPrompt.calculate_metrics(sample, point, gt_bbox)


class Qwen3VLPrompt(GUIOwlPrompt):
    """
    Prompt processor for BASE Qwen3-VL-*-Instruct models, faithful to the
    OFFICIAL Qwen3-VL GUI-grounding recipe
    (QwenLM/Qwen3-VL cookbooks/computer_use.ipynb, local-HF path):
      * system = "You are a helpful assistant." + NousFnCallPrompt tool block
        (qwen_agent FN_CALL_TEMPLATE) carrying the FULL `computer_use` function
        from cookbooks/utils/agent_function_call.py::ComputerUse with
        display_width_px = display_height_px = 1000 -- NOT GUI-Owl's stripped
        two-action variant (base Qwen3-VL was never trained on that);
      * user turn = query text FIRST, then the image (cookbook order);
      * the model replies <tool_call>{"name": "computer_use", "arguments":
        {"action": "left_click", "coordinate": [x, y]}}</tool_call> with
        coordinates in the 0-1000 space, which GUIOwlPrompt.calculate_metrics
        (inherited) already maps/scores against processed_imgsize;
      * cookbook resolution: smart_resize(factor=32, min=16*32^2,
        max=6400*32^2=6553600) -> set IMAGE_FACTOR=32 and
        MAX_IMAGE_PIXELS=6553600.

    extract_coordinates parses the <tool_call> JSON first (exact cookbook
    parse), falling back to the GUI-Owl "(x, y)" / "[x, y]" regexes for
    slightly malformed replies. A terminate/failure reply becomes the refusal
    sentinel [-1, -1] (only creditable on refusal-type gt, e.g. OSWorld-G).
    """

    # ComputerUse.description / .parameters verbatim from
    # Qwen3-VL cookbooks/utils/agent_function_call.py, with the f-string
    # resolved at cfg = {"display_width_px": 1000, "display_height_px": 1000}.
    COMPUTER_USE_FUNCTION = {
        "name": "computer_use",
        "description": (
            "Use a mouse and keyboard to interact with a computer, and take screenshots.\n"
            "* This is an interface to a desktop GUI. You do not have access to a terminal or applications menu. You must click on desktop icons to start applications.\n"
            "* Some applications may take time to start or process actions, so you may need to wait and take successive screenshots to see the results of your actions. E.g. if you click on Firefox and a window doesn't open, try wait and taking another screenshot.\n"
            "* The screen's resolution is 1000x1000.\n"
            "* Whenever you intend to move the cursor to click on an element like an icon, you should consult a screenshot to determine the coordinates of the element before moving the cursor.\n"
            "* If you tried clicking on a program or link but it failed to load, even after waiting, try adjusting your cursor position so that the tip of the cursor visually falls on the element that you want to click.\n"
            "* Make sure to click any buttons, links, icons, etc with the cursor tip in the center of the element. Don't click boxes on their edges."
        ),
        "parameters": {
            "properties": {
                "action": {
                    "description": (
                        "The action to perform. The available actions are:\n"
                        "* `key`: Performs key down presses on the arguments passed in order, then performs key releases in reverse order.\n"
                        "* `type`: Type a string of text on the keyboard.\n"
                        "* `mouse_move`: Move the cursor to a specified (x, y) pixel coordinate on the screen.\n"
                        "* `left_click`: Click the left mouse button at a specified (x, y) pixel coordinate on the screen.\n"
                        "* `left_click_drag`: Click and drag the cursor to a specified (x, y) pixel coordinate on the screen.\n"
                        "* `right_click`: Click the right mouse button at a specified (x, y) pixel coordinate on the screen.\n"
                        "* `middle_click`: Click the middle mouse button at a specified (x, y) pixel coordinate on the screen.\n"
                        "* `double_click`: Double-click the left mouse button at a specified (x, y) pixel coordinate on the screen.\n"
                        "* `triple_click`: Triple-click the left mouse button at a specified (x, y) pixel coordinate on the screen (simulated as double-click since it's the closest action).\n"
                        "* `scroll`: Performs a scroll of the mouse scroll wheel.\n"
                        "* `hscroll`: Performs a horizontal scroll (mapped to regular scroll).\n"
                        "* `wait`: Wait specified seconds for the change to happen.\n"
                        "* `terminate`: Terminate the current task and report its completion status.\n"
                        "* `answer`: Answer a question."
                    ),
                    "enum": [
                        "key", "type", "mouse_move", "left_click", "left_click_drag",
                        "right_click", "middle_click", "double_click", "triple_click",
                        "scroll", "hscroll", "wait", "terminate", "answer",
                    ],
                    "type": "string",
                },
                "keys": {
                    "description": "Required only by `action=key`.",
                    "type": "array",
                },
                "text": {
                    "description": "Required only by `action=type` and `action=answer`.",
                    "type": "string",
                },
                "coordinate": {
                    "description": "(x, y): The x (pixels from the left edge) and y (pixels from the top edge) coordinates to move the mouse to.",
                    "type": "array",
                },
                "pixels": {
                    "description": "The amount of scrolling to perform. Positive values scroll up, negative values scroll down. Required only by `action=scroll` and `action=hscroll`.",
                    "type": "number",
                },
                "time": {
                    "description": "The seconds to wait. Required only by `action=wait`.",
                    "type": "number",
                },
                "status": {
                    "description": "The status of the task. Required only by `action=terminate`.",
                    "type": "string",
                    "enum": ["success", "failure"],
                },
            },
            "required": ["action"],
            "type": "object",
        },
    }

    # qwen_agent nous_fncall_prompt.FN_CALL_TEMPLATE, verbatim.
    FN_CALL_TEMPLATE = (
        "# Tools\n\n"
        "You may call one or more functions to assist with the user query.\n\n"
        "You are provided with function signatures within <tools></tools> XML tags:\n"
        "<tools>\n"
        "{tool_descs}\n"
        "</tools>\n\n"
        "For each function call, return a json object with function name and arguments within "
        "<tool_call></tool_call> XML tags:\n"
        "<tool_call>\n"
        '{{"name": <function-name>, "arguments": <args-json-object>}}\n'
        "</tool_call>"
    )

    # NousFnCallPrompt appends '\n\n' + tool block to the existing system text.
    SYSTEM_PROMPT = "You are a helpful assistant.\n\n" + FN_CALL_TEMPLATE.format(
        tool_descs=json.dumps(
            {"type": "function", "function": COMPUTER_USE_FUNCTION}, ensure_ascii=False
        )
    )

    @staticmethod
    def generate_prompt(sample: Dict, image_width: int, image_height: int, think_mode: bool = True) -> List[Dict]:
        # The cookbook is inconsistent about the user-turn order: the local-HF cell
        # puts the query text BEFORE the image, the API cell puts the image first
        # (as does every GUI grounding recipe incl. GUI-Owl's official eval).
        # Default to image-first; QWEN3VL_TEXT_FIRST=1 switches to the HF-cell order.
        if os.environ.get("QWEN3VL_TEXT_FIRST") == "1":
            user_content = f"{sample['instruction']}<image>"
        else:
            user_content = f"<image>{sample['instruction']}"
        return [
            {"role": "system", "content": Qwen3VLPrompt.SYSTEM_PROMPT},
            {"role": "user", "content": user_content},
        ]

    @staticmethod
    def extract_coordinates(response: str, think_mode: bool = False) -> Optional[List]:
        # Exact cookbook parse: the JSON object inside <tool_call>...</tool_call>.
        blobs = re.findall(r"<tool_call>\s*(\{.*?\})\s*</tool_call>", response, re.DOTALL)
        for blob in reversed(blobs):
            try:
                action = json.loads(blob)
            except json.JSONDecodeError:
                continue
            args = action.get("arguments") or {}
            # terminate is the abstain action learned from the *_refusal_grounding_v1
            # templates; abstain is the one from *_abstain_grounding_v1. Both are accepted:
            # adding abstain is purely additive for checkpoints trained on the other schemas
            # (they have no such action and never emit it), so parsing need not fork per
            # processor.
            if args.get("action") in ("terminate", "abstain"):
                return [-1, -1]
            coord = args.get("coordinate")
            if isinstance(coord, (list, tuple)) and len(coord) >= 2:
                try:
                    return [float(coord[0]), float(coord[1])]
                except (TypeError, ValueError):
                    continue
        # Fallback for malformed tool calls: the GUI-Owl regex + refusal handling.
        return GUIOwlPrompt.extract_coordinates(response, think_mode)


class Qwen3VLPointPrompt(GUIOwlPrompt):
    """
    Qwen3-VL NATIVE point-grounding format (QwenLM/Qwen3-VL cookbooks/
    2d_grounding.ipynb), as opposed to the agentic computer_use tool format:
      * no tool block -- plain "You are a helpful assistant." system + image +
        a 'Locate ... report its point coordinates in JSON format like this:
        {"point_2d": [x, y], ...}' instruction (cookbook Example-5 phrasing);
      * Qwen3-VL grounding answers in RELATIVE 0-1000 coordinates, so scoring
        reuses GUIOwlPrompt.calculate_metrics unchanged.
    Unlike the computer_use schema this format has no terminate/type/answer
    escape hatch, so the model always commits to a coordinate.
    (Qwen's exact internal ScreenSpot harness is unpublished; this is the
    closest public-recipe candidate for it.)
    """

    USER_PROMPT = (
        'Locate the UI element according to the instruction "{instruction}" with a point, '
        'report its point coordinates in JSON format like this: '
        '{{"point_2d": [x, y], "label": "target element"}}'
    )

    # Without this the model answers "There are none." on ~2.6% of ScreenSpot-V2
    # (33/1272) instead of committing to a point; grounding-benchmark harnesses
    # conventionally assert existence (cf. UI-Venus). Enable via QWEN3VL_POINT_FORCE=1.
    FORCE_SUFFIX = " The target element is guaranteed to be visible on the screen."

    @staticmethod
    def generate_prompt(sample: Dict, image_width: int, image_height: int, think_mode: bool = True) -> List[Dict]:
        text = Qwen3VLPointPrompt.USER_PROMPT.format(instruction=sample["instruction"])
        if os.environ.get("QWEN3VL_POINT_FORCE") == "1":
            text += Qwen3VLPointPrompt.FORCE_SUFFIX
        return [
            {"role": "user", "content": "<image>" + text},
        ]

    @staticmethod
    def extract_coordinates(response: str, think_mode: bool = False) -> Optional[List]:
        text = response.strip()
        if "```json" in text:
            text = text.split("```json", 1)[1].split("```", 1)[0]
        # "point_2d": [x, y] -- take the first occurrence (single-target query).
        m = re.search(r'"point_2d"\s*:\s*\[\s*(-?\d+(?:\.\d+)?)\s*,\s*(-?\d+(?:\.\d+)?)\s*\]', text)
        if m:
            return [float(m.group(1)), float(m.group(2))]
        # Fallback: any bare [x, y] pair.
        m = re.search(r"\[\s*(-?\d+(?:\.\d+)?)\s*,\s*(-?\d+(?:\.\d+)?)\s*\]", text)
        if m:
            return [float(m.group(1)), float(m.group(2))]
        # Last resort: GUI-Owl "(x, y)" / "[x, y]" integer regexes.
        return GUIOwlPrompt.extract_coordinates(response, think_mode)


class ElementPointPrompt(GUIOwlPrompt):
    """Eval-side counterpart of the `internvl_grounding_point_{desktop,ubuntu,web,mobile}_v1` training templates.

    On the training side this is `element_point_system_prompt` in ScaleCUA's internvl
    conversation templates; the four platform templates differ only in two fill-ins, surface
    and verb. This class likewise keeps a single template plus a platform table and renders it
    exactly as training does.

    Difference from scalecua_toolcall (two target formats for the same data)
    ========================================================================
    The point-type grounding data originally wrote its target as a <tool_call> click, i.e. the
    GUI-action path -- the model learns to "emit an action". After conversion to the <point>
    format the target becomes pure localization with the contract "return one point", so it
    must be evaluated with this prompt; evaluating it with scalecua_toolcall would make the model
    answer under a contract it has never seen.

    Input / output
    ==============
    user  : "<image>\n{instruction}" -- identical to the human turn of the training data (the
            newline is real: models/qwen2vl.py only replaces <image> in place with vision
            tokens and leaves the rest of the text untouched, so a missing \n would differ
            from training).
            Note that <ref> appears in the **gpt target**, not in the user turn -- the human
            value in the training data is the bare instruction, so do not add markers to the
            user turn just because the system prompt says "wrapped in <ref>...</ref>".
    gpt   : ```json\n[\n    {"point_2d": [x, y], "label": "<instruction>"}\n]\n```
            produced by find_point in sft/qwenvl/data/data_qwen.py.

    Coordinate space
    ================
    find_point writes `point/1000*new_image_size`, and data_qwen.py sets
    `coord_size = (1000, 1000) if self.coord_norm else new_image_size`.
    Training uses coord_norm=True (the default in argument.py; the SFT scripts also pass it
    explicitly), so that conversion is the identity and the model outputs **0-1000**,
    consistent with "integers in a 1000x1000 space" in the system prompt. Hence
    GUIOwlPrompt.calculate_metrics is reused as-is (both official 1000-space scoring and exact
    pixel-space containment are available).
    **If training switches to coord_norm=False, the model outputs absolute pixels of the
    resized image and this class must change accordingly; otherwise every coordinate is
    silently off.**
    """

    # Verbatim copy of the training-side element_point_system_prompt (including {surface}/{verb}).
    _TEMPLATE = (
        "You are a GUI grounding assistant operating on {surface}. Given a screenshot and a "
        "referring expression that identifies one on-screen element, locate that element and "
        "return a single point inside it.\n"
        "\n"
        "## Input Specification\n"
        "- A screenshot of the current screen + a referring expression wrapped in <ref>...</ref>. "
        "The expression is either the element's visible text, a description of it, or its position "
        "relative to another element.\n"
        "\n"
        "## Output Format\n"
        "Return a JSON array holding the single referred element and one point [x, y] inside it:\n"
        "```json\n"
        '[{{"point_2d": [x, y], "label": "<referring expression>"}}]\n'
        "```\n"
        "\n"
        "## Note\n"
        "- Return exactly one element: the one the referring expression names. Do not list other elements.\n"
        "- The point must fall inside the element, as close to its centre as you can judge.\n"
        "- If the expression names an anchor element and states where the target sits relative to it, "
        "return the point of the TARGET, never the anchor.\n"
        '- The expression may be phrased as an instruction ("{verb} the Save button"). Do not perform '
        "the action; only return the point of the element it names.\n"
        "- All coordinates are integers in a 1000x1000 space (x from the left edge, y from the top edge).\n"
        "- Output only the JSON array."
    )

    # Training registers four per-platform templates, but they differ only in the surface/verb
    # fill-ins, so the eval side needs only **one** prompt: generate_prompt dispatches on the
    # sample's group (same approach as ScaleCUAToolCallPrompt) instead of registering one name
    # per platform.
    # ubuntu and desktop share the same surface text, as in training (both templates render
    # identically).
    _SURFACE = {
        "desktop": ("desktop application windows", "click"),
        "web":     ("web pages in a browser", "click"),
        "mobile":  ("mobile phone and tablet screens", "tap"),
    }

    _JSON_PT = re.compile(r'"point_2d"\s*:\s*\[\s*(-?\d+(?:\.\d+)?)\s*,\s*(-?\d+(?:\.\d+)?)\s*\]')

    @staticmethod
    def _system(platform: str) -> str:
        surface, verb = ElementPointPrompt._SURFACE.get(
            platform, ElementPointPrompt._SURFACE["desktop"])
        return ElementPointPrompt._TEMPLATE.format(surface=surface, verb=verb)

    @staticmethod
    def generate_prompt(sample: Dict, image_width: int, image_height: int, think_mode: bool = True) -> List[Dict]:
        plat = ScaleCUAToolCallPrompt._platform_of(sample)
        user = f"<image>\n{sample['instruction']}"
        if os.environ.get("QWEN3VL_TEXT_FIRST") == "1":
            user = f"{sample['instruction']}\n<image>"
        return [
            {"role": "system", "content": ElementPointPrompt._system(plat)},
            {"role": "user", "content": user},
        ]

    @staticmethod
    def extract_coordinates(response: str, think_mode: bool = False) -> Optional[List]:
        m = ElementPointPrompt._JSON_PT.findall(response or "")
        if m:
            x, y = m[0]                      # contract is "exactly one element"; take the first
            return [float(x), float(y)]
        # Fallback: the model occasionally drops the "point_2d" key and emits a bare [x, y]
        m = re.findall(r"\[\s*(-?\d+(?:\.\d+)?)\s*,\s*(-?\d+(?:\.\d+)?)\s*\]", response or "")
        if not m:
            return None
        x, y = m[0]
        return [float(x), float(y)]


class ScaleCUAToolCallPrompt(Qwen3VLPrompt):
    """ScaleCUA-recipe **Qwen3-VL** checkpoints trained on tool_call-format grounding data
    (coord_norm=True -> the model emits [0,1000] coordinates inside a <tool_call> JSON
    block), e.g. the SFT in sft/.

    Difference from the neighbouring processors:
      * `scalecua`  -> ScaleCUA's original pyautogui `<action>click(x=..,y=..)</action>`
                       format; does NOT match these checkpoints.
      * `qwen3vl`   -> base Qwen3-VL cookbook computer_use prompt (generic).
      * `scalecua_toolcall` (this) -> the EXACT grounding system prompt the model
                       was trained with (the internvl2_5_{platform}_grounding_v1 conv_style
                       of the SFT data), selected PER PLATFORM:
                         desktop -> computer_use, mobile -> mobile_use,
                         web     -> browser_use (grounding subsets).

    extract_coordinates / calculate_metrics are inherited from Qwen3VLPrompt: parse the
    <tool_call> `coordinate` (0-1000); calculate_metrics (from GUIOwlPrompt) maps it into the
    resized pixel space itself. Leave COORD_SPACE unset: COORD_SPACE=norm1000 would rescale a
    second time and accuracy drops to ~0.
    """

    # Frozen (hard-coded) per-platform grounding system prompts: the EXACT
    # internvl2_5_{platform}_grounding_v1 system_message the checkpoint was trained
    # with. Hard-coded (NOT imported from the training conversation templates at runtime)
    # so this eval is self-contained. Regenerate if the training grounding action
    # space changes. left_click_drag is described with two points (`coordinate` start +
    # `coordinate2` end), matching the training templates.
    _GROUNDING_SYSTEM = {
        "desktop": r"""You are an autonomous GUI agent capable of operating on desktops, mobile devices, and web browsers. Your primary function is to analyze a screen capture, locate the target UI element, and issue the single appropriate action that fulfills the given instruction.

# Tools

You may call one or more functions to assist with the user query.

You are provided with function signatures within <tools></tools> XML tags:
<tools>
{"type": "function", "function": {"name": "computer_use", "description": "Use a mouse and keyboard to interact with a computer, and take screenshots.\n* This is an interface to a desktop GUI. You do not have access to a terminal or applications menu. You must click on desktop icons to start applications.\n* Some applications may take time to start or process actions, so you may need to wait and take successive screenshots to see the results of your actions. E.g. if you click on Firefox and a window doesn't open, try wait and taking another screenshot.\n* The screen's resolution is 1000x1000.\n* Make sure to click any buttons, links, icons, etc with the cursor tip in the center of the element. Don't click boxes on their edges unless asked.", "parameters": {"properties": {"action": {"description": "The action to perform. The available actions are:\n* `mouse_move`: Move the cursor to a specified (x, y) pixel coordinate on the screen.\n* `left_click`: Click the left mouse button at a specified (x, y) pixel coordinate on the screen.\n* `left_click_drag`: Press the left mouse button at the start point with coordinate (x, y), drag to the end point with coordinate2 (x2, y2), and release.\n* `right_click`: Click the right mouse button at a specified (x, y) pixel coordinate on the screen.\n* `middle_click`: Click the middle mouse button at a specified (x, y) pixel coordinate on the screen.\n* `double_click`: Double-click the left mouse button at a specified (x, y) pixel coordinate on the screen.\n* `triple_click`: Triple-click the left mouse button at a specified (x, y) pixel coordinate on the screen.", "enum": ["mouse_move", "left_click", "left_click_drag", "right_click", "middle_click", "double_click", "triple_click"], "type": "string"}, "coordinate": {"description": "(x, y): The x (pixels from the left edge) and y (pixels from the top edge) coordinates to move the mouse to. Required only by `action=mouse_move`, `action=left_click`, `action=left_click_drag`, `action=right_click`, `action=middle_click`, `action=double_click`, and `action=triple_click`.", "type": "array"}, "coordinate2": {"description": "(x, y): The x (pixels from the left edge) and y (pixels from the top edge) coordinates to drag the cursor to. Required only by `action=left_click_drag`.", "type": "array"}}, "required": ["action"], "type": "object"}}}
</tools>

For each function call, return a json object with function name and arguments within <tool_call></tool_call> XML tags:
<tool_call>
{"name": <function-name>, "arguments": <args-json-object>}
</tool_call>

## Input Specification
- A screenshot of the current screen + an instruction describing the action to perform.

## Output Format
- Emit exactly one <tool_call></tool_call> block containing a single JSON object with the function name and its arguments.
- Coordinates are integers in a 1000x1000 space (x measured from the left edge, y from the top edge).

## Note
- Choose exactly one function whose name and arguments are defined in the tool schema above.
- Do not output anything other than the <tool_call></tool_call> block.""",
        "mobile": r"""You are an autonomous GUI agent capable of operating on desktops, mobile devices, and web browsers. Your primary function is to analyze a screen capture, locate the target UI element, and issue the single appropriate action that fulfills the given instruction.

# Tools

You may call one or more functions to assist with the user query.

You are provided with function signatures within <tools></tools> XML tags:
<tools>
{"type": "function", "function": {"name": "mobile_use", "description": "Use a touchscreen to interact with a mobile device, and take screenshots.\n* This is an interface to a mobile device with touchscreen. You can perform actions like clicking, typing, swiping, etc.\n* Some applications may take time to start or process actions, so you may need to wait and take successive screenshots to see the results of your actions.\n* The screen's resolution is 1000x1000.\n* Make sure to click any buttons, links, icons, etc with the cursor tip in the center of the element. Don't click boxes on their edges unless asked.", "parameters": {"properties": {"action": {"description": "The action to perform. The available actions are:\n* `click`: Click the point on the screen with coordinate (x, y).\n* `long_press`: Press the point on the screen with coordinate (x, y) for specified seconds.", "enum": ["click", "long_press"], "type": "string"}, "coordinate": {"description": "(x, y): the point on the screen the action acts on — the element to touch, the region to scroll, or the object to start dragging. x is measured from the left edge and y from the top edge, both on a 0-1000 scale, so y gets larger toward the bottom of the screen. Required only by `action=click` and `action=long_press`.", "type": "array"}, "time": {"description": "The seconds to wait. Required only by `action=long_press`.", "type": "number"}}, "required": ["action"], "type": "object"}}}
</tools>

For each function call, return a json object with function name and arguments within <tool_call></tool_call> XML tags:
<tool_call>
{"name": <function-name>, "arguments": <args-json-object>}
</tool_call>

## Input Specification
- A screenshot of the current screen + an instruction describing the action to perform.

## Output Format
- Emit exactly one <tool_call></tool_call> block containing a single JSON object with the function name and its arguments.
- Coordinates are integers in a 1000x1000 space (x measured from the left edge, y from the top edge).

## Note
- Choose exactly one function whose name and arguments are defined in the tool schema above.
- Do not output anything other than the <tool_call></tool_call> block.""",
        "web": r"""You are an autonomous GUI agent capable of operating on desktops, mobile devices, and web browsers. Your primary function is to analyze a screen capture, locate the target UI element, and issue the single appropriate action that fulfills the given instruction.

# Tools

You may call one or more functions to assist with the user query.

You are provided with function signatures within <tools></tools> XML tags:
<tools>
{"type": "function", "function": {"name": "browser_use", "description": "Use a mouse and keyboard to interact with a web browser, and take screenshots.\n* This is an interface to a web browser. You interact with the rendered web page by clicking, typing, scrolling, selecting options and navigating between pages.\n* Some pages may take time to load or process actions, so you may need to wait and take successive screenshots to see the results of your actions.\n* The screen's resolution is 1000x1000.\n* Make sure to click any buttons, links, icons, etc with the cursor tip in the center of the element. Don't click boxes on their edges unless asked.", "parameters": {"properties": {"action": {"description": "The action to perform. The available actions are:\n* `mouse_move`: Move the cursor to a specified (x, y) pixel coordinate on the screen.\n* `left_click`: Click the left mouse button at a specified (x, y) pixel coordinate on the screen.\n* `left_click_drag`: Press the left mouse button at the start point with coordinate (x, y), drag to the end point with coordinate2 (x2, y2), and release.\n* `right_click`: Click the right mouse button at a specified (x, y) pixel coordinate on the screen.\n* `middle_click`: Click the middle mouse button at a specified (x, y) pixel coordinate on the screen.\n* `double_click`: Double-click the left mouse button at a specified (x, y) pixel coordinate on the screen.\n* `triple_click`: Triple-click the left mouse button at a specified (x, y) pixel coordinate on the screen.\n* `select`: Select an option in a dropdown or list element at the specified (x, y) coordinate; the option to choose is given by text.", "enum": ["mouse_move", "left_click", "left_click_drag", "right_click", "middle_click", "double_click", "triple_click", "select"], "type": "string"}, "coordinate": {"description": "(x, y): The x (pixels from the left edge) and y (pixels from the top edge) coordinates to move the mouse to. Required only by `action=mouse_move`, `action=left_click`, `action=left_click_drag`, `action=right_click`, `action=middle_click`, `action=double_click`, `action=triple_click`, and `action=select`.", "type": "array"}, "coordinate2": {"description": "(x, y): The x (pixels from the left edge) and y (pixels from the top edge) coordinates to drag the cursor to. Required only by `action=left_click_drag`.", "type": "array"}, "text": {"description": "Required only by `action=select`.", "type": "string"}}, "required": ["action"], "type": "object"}}}
</tools>

For each function call, return a json object with function name and arguments within <tool_call></tool_call> XML tags:
<tool_call>
{"name": <function-name>, "arguments": <args-json-object>}
</tool_call>

## Input Specification
- A screenshot of the current screen + an instruction describing the action to perform.

## Output Format
- Emit exactly one <tool_call></tool_call> block containing a single JSON object with the function name and its arguments.
- Coordinates are integers in a 1000x1000 space (x measured from the left edge, y from the top edge).

## Note
- Choose exactly one function whose name and arguments are defined in the tool schema above.
- Do not output anything other than the <tool_call></tool_call> block.""",
    }

    # Frozen per-platform ABSTAIN grounding prompts: the training-side
    # internvl2_5_{desktop,web,mobile}_abstain_grounding_v1 system_message. Difference from
    # _GROUNDING_REFUSAL_SYSTEM below: the abstain action is the argument-less `abstain`, so
    # the schema has no `status` property at all (the success/failure enum goes with it).
    # Both strings are rendered from the training templates and pasted in verbatim.
    _GROUNDING_ABSTAIN_SYSTEM = {
        "desktop": r"""You are an autonomous GUI agent capable of operating on desktops, mobile devices, and web browsers. Your primary function is to analyze a screen capture, locate the target UI element, and issue the single appropriate action that fulfills the given instruction.

# Tools

You may call one or more functions to assist with the user query.

You are provided with function signatures within <tools></tools> XML tags:
<tools>
{"type": "function", "function": {"name": "computer_use", "description": "Use a mouse and keyboard to interact with a computer, and take screenshots.\n* This is an interface to a desktop GUI. You do not have access to a terminal or applications menu. You must click on desktop icons to start applications.\n* Some applications may take time to start or process actions, so you may need to wait and take successive screenshots to see the results of your actions. E.g. if you click on Firefox and a window doesn't open, try wait and taking another screenshot.\n* The screen's resolution is 1000x1000.\n* Make sure to click any buttons, links, icons, etc with the cursor tip in the center of the element. Don't click boxes on their edges unless asked.", "parameters": {"properties": {"action": {"description": "The action to perform. The available actions are:\n* `mouse_move`: Move the cursor to a specified (x, y) pixel coordinate on the screen.\n* `left_click`: Click the left mouse button at a specified (x, y) pixel coordinate on the screen.\n* `left_click_drag`: Press the left mouse button at the start point with coordinate (x, y), drag to the end point with coordinate2 (x2, y2), and release.\n* `right_click`: Click the right mouse button at a specified (x, y) pixel coordinate on the screen.\n* `middle_click`: Click the middle mouse button at a specified (x, y) pixel coordinate on the screen.\n* `double_click`: Double-click the left mouse button at a specified (x, y) pixel coordinate on the screen.\n* `triple_click`: Triple-click the left mouse button at a specified (x, y) pixel coordinate on the screen.\n* `abstain`: Do not click and return no coordinate. Use this if, and only if, the element named by the instruction does not exist anywhere on the screen, so no click could be correct. This reports only about the current screenshot; it does not end the task.", "enum": ["mouse_move", "left_click", "left_click_drag", "right_click", "middle_click", "double_click", "triple_click", "abstain"], "type": "string"}, "coordinate": {"description": "(x, y): The x (pixels from the left edge) and y (pixels from the top edge) coordinates to move the mouse to. Required only by `action=mouse_move`, `action=left_click`, `action=left_click_drag`, `action=right_click`, `action=middle_click`, `action=double_click`, and `action=triple_click`.", "type": "array"}, "coordinate2": {"description": "(x, y): The x (pixels from the left edge) and y (pixels from the top edge) coordinates to drag the cursor to. Required only by `action=left_click_drag`.", "type": "array"}}, "required": ["action"], "type": "object"}}}
</tools>

For each function call, return a json object with function name and arguments within <tool_call></tool_call> XML tags:
<tool_call>
{"name": <function-name>, "arguments": <args-json-object>}
</tool_call>

## Input Specification
- A screenshot of the current screen + an instruction describing the action to perform.

## Output Format
- Emit exactly one <tool_call></tool_call> block containing a single JSON object with the function name and its arguments.
- Coordinates are integers in a 1000x1000 space (x measured from the left edge, y from the top edge).

## Note
- Choose exactly one function whose name and arguments are defined in the tool schema above.
- Do not output anything other than the <tool_call></tool_call> block.
- If, and only if, the element named by the instruction genuinely does not exist anywhere in the screenshot, do not click: emit `abstain`.
- Strongly prefer locating and clicking the element; abstain only when you are confident it is truly absent.""",
        "web": r"""You are an autonomous GUI agent capable of operating on desktops, mobile devices, and web browsers. Your primary function is to analyze a screen capture, locate the target UI element, and issue the single appropriate action that fulfills the given instruction.

# Tools

You may call one or more functions to assist with the user query.

You are provided with function signatures within <tools></tools> XML tags:
<tools>
{"type": "function", "function": {"name": "browser_use", "description": "Use a mouse and keyboard to interact with a web browser, and take screenshots.\n* This is an interface to a web browser. You interact with the rendered web page by clicking, typing, scrolling, selecting options and navigating between pages.\n* Some pages may take time to load or process actions, so you may need to wait and take successive screenshots to see the results of your actions.\n* The screen's resolution is 1000x1000.\n* Make sure to click any buttons, links, icons, etc with the cursor tip in the center of the element. Don't click boxes on their edges unless asked.", "parameters": {"properties": {"action": {"description": "The action to perform. The available actions are:\n* `mouse_move`: Move the cursor to a specified (x, y) pixel coordinate on the screen.\n* `left_click`: Click the left mouse button at a specified (x, y) pixel coordinate on the screen.\n* `left_click_drag`: Press the left mouse button at the start point with coordinate (x, y), drag to the end point with coordinate2 (x2, y2), and release.\n* `right_click`: Click the right mouse button at a specified (x, y) pixel coordinate on the screen.\n* `middle_click`: Click the middle mouse button at a specified (x, y) pixel coordinate on the screen.\n* `double_click`: Double-click the left mouse button at a specified (x, y) pixel coordinate on the screen.\n* `triple_click`: Triple-click the left mouse button at a specified (x, y) pixel coordinate on the screen.\n* `select`: Select an option in a dropdown or list element at the specified (x, y) coordinate; the option to choose is given by text.\n* `abstain`: Do not click and return no coordinate. Use this if, and only if, the element named by the instruction does not exist anywhere on the screen, so no click could be correct. This reports only about the current screenshot; it does not end the task.", "enum": ["mouse_move", "left_click", "left_click_drag", "right_click", "middle_click", "double_click", "triple_click", "select", "abstain"], "type": "string"}, "coordinate": {"description": "(x, y): The x (pixels from the left edge) and y (pixels from the top edge) coordinates to move the mouse to. Required only by `action=mouse_move`, `action=left_click`, `action=left_click_drag`, `action=right_click`, `action=middle_click`, `action=double_click`, `action=triple_click`, and `action=select`.", "type": "array"}, "coordinate2": {"description": "(x, y): The x (pixels from the left edge) and y (pixels from the top edge) coordinates to drag the cursor to. Required only by `action=left_click_drag`.", "type": "array"}, "text": {"description": "Required only by `action=select`.", "type": "string"}}, "required": ["action"], "type": "object"}}}
</tools>

For each function call, return a json object with function name and arguments within <tool_call></tool_call> XML tags:
<tool_call>
{"name": <function-name>, "arguments": <args-json-object>}
</tool_call>

## Input Specification
- A screenshot of the current screen + an instruction describing the action to perform.

## Output Format
- Emit exactly one <tool_call></tool_call> block containing a single JSON object with the function name and its arguments.
- Coordinates are integers in a 1000x1000 space (x measured from the left edge, y from the top edge).

## Note
- Choose exactly one function whose name and arguments are defined in the tool schema above.
- Do not output anything other than the <tool_call></tool_call> block.
- If, and only if, the element named by the instruction genuinely does not exist anywhere in the screenshot, do not click: emit `abstain`.
- Strongly prefer locating and clicking the element; abstain only when you are confident it is truly absent.""",
        "mobile": r"""You are an autonomous GUI agent capable of operating on desktops, mobile devices, and web browsers. Your primary function is to analyze a screen capture, locate the target UI element, and issue the single appropriate action that fulfills the given instruction.

# Tools

You may call one or more functions to assist with the user query.

You are provided with function signatures within <tools></tools> XML tags:
<tools>
{"type": "function", "function": {"name": "mobile_use", "description": "Use a touchscreen to interact with a mobile device, and take screenshots.\n* This is an interface to a mobile device with touchscreen. You can perform actions like clicking, typing, swiping, etc.\n* Some applications may take time to start or process actions, so you may need to wait and take successive screenshots to see the results of your actions.\n* The screen's resolution is 1000x1000.\n* Make sure to click any buttons, links, icons, etc with the cursor tip in the center of the element. Don't click boxes on their edges unless asked.", "parameters": {"properties": {"action": {"description": "The action to perform. The available actions are:\n* `click`: Click the point on the screen with coordinate (x, y).\n* `long_press`: Press the point on the screen with coordinate (x, y) for specified seconds.\n* `abstain`: Do not click and return no coordinate. Use this if, and only if, the element named by the instruction does not exist anywhere on the screen, so no click could be correct. This reports only about the current screenshot; it does not end the task.", "enum": ["click", "long_press", "abstain"], "type": "string"}, "coordinate": {"description": "(x, y): the point on the screen the action acts on — the element to touch, the region to scroll, or the object to start dragging. x is measured from the left edge and y from the top edge, both on a 0-1000 scale, so y gets larger toward the bottom of the screen. Required only by `action=click` and `action=long_press`.", "type": "array"}, "time": {"description": "The seconds to wait. Required only by `action=long_press`.", "type": "number"}}, "required": ["action"], "type": "object"}}}
</tools>

For each function call, return a json object with function name and arguments within <tool_call></tool_call> XML tags:
<tool_call>
{"name": <function-name>, "arguments": <args-json-object>}
</tool_call>

## Input Specification
- A screenshot of the current screen + an instruction describing the action to perform.

## Output Format
- Emit exactly one <tool_call></tool_call> block containing a single JSON object with the function name and its arguments.
- Coordinates are integers in a 1000x1000 space (x measured from the left edge, y from the top edge).

## Note
- Choose exactly one function whose name and arguments are defined in the tool schema above.
- Do not output anything other than the <tool_call></tool_call> block.
- If, and only if, the element named by the instruction genuinely does not exist anywhere in the screenshot, do not click: emit `abstain`.
- Strongly prefer locating and clicking the element; abstain only when you are confident it is truly absent.""",
    }

    # Frozen per-platform REFUSAL grounding prompts: the EXACT
    # internvl2_5_{ubuntu,web,android}_refusal_grounding_v1 system_message. It is the plain
    # prompt above plus (a) `terminate` in the tool schema, re-described for abstention (the
    # shared action wording talks about task "completion status", which points away from
    # "the element is absent"), and (b) two extra `## Note` bullets saying when abstaining is
    # correct. Regenerate from the training templates if they change. A checkpoint served the
    # PLAIN prompt cannot emit terminate at all (measured 0/564 emissions, score
    # bit-identical), so any drift here silently forfeits the whole OSWorld-G refusal subset.
    # left_click_drag takes two points (`coordinate` start + `coordinate2` end), as in training.
    _GROUNDING_REFUSAL_SYSTEM = {
        "desktop": r"""You are an autonomous GUI agent capable of operating on desktops, mobile devices, and web browsers. Your primary function is to analyze a screen capture, locate the target UI element, and issue the single appropriate action that fulfills the given instruction.

# Tools

You may call one or more functions to assist with the user query.

You are provided with function signatures within <tools></tools> XML tags:
<tools>
{"type": "function", "function": {"name": "computer_use", "description": "Use a mouse and keyboard to interact with a computer, and take screenshots.\n* This is an interface to a desktop GUI. You do not have access to a terminal or applications menu. You must click on desktop icons to start applications.\n* Some applications may take time to start or process actions, so you may need to wait and take successive screenshots to see the results of your actions. E.g. if you click on Firefox and a window doesn't open, try wait and taking another screenshot.\n* The screen's resolution is 1000x1000.\n* Make sure to click any buttons, links, icons, etc with the cursor tip in the center of the element. Don't click boxes on their edges unless asked.", "parameters": {"properties": {"action": {"description": "The action to perform. The available actions are:\n* `mouse_move`: Move the cursor to a specified (x, y) pixel coordinate on the screen.\n* `left_click`: Click the left mouse button at a specified (x, y) pixel coordinate on the screen.\n* `left_click_drag`: Press the left mouse button at the start point with coordinate (x, y), drag to the end point with coordinate2 (x2, y2), and release.\n* `right_click`: Click the right mouse button at a specified (x, y) pixel coordinate on the screen.\n* `middle_click`: Click the middle mouse button at a specified (x, y) pixel coordinate on the screen.\n* `double_click`: Double-click the left mouse button at a specified (x, y) pixel coordinate on the screen.\n* `triple_click`: Triple-click the left mouse button at a specified (x, y) pixel coordinate on the screen.\n* `terminate`: Abort the task without clicking and report the outcome in `status`. Use `status=\"failure\"` when the element named by the instruction does not exist anywhere on the screen, so no click could be correct.", "enum": ["mouse_move", "left_click", "left_click_drag", "right_click", "middle_click", "double_click", "triple_click", "terminate"], "type": "string"}, "coordinate": {"description": "(x, y): The x (pixels from the left edge) and y (pixels from the top edge) coordinates to move the mouse to. Required only by `action=mouse_move`, `action=left_click`, `action=left_click_drag`, `action=right_click`, `action=middle_click`, `action=double_click`, and `action=triple_click`.", "type": "array"}, "coordinate2": {"description": "(x, y): The x (pixels from the left edge) and y (pixels from the top edge) coordinates to drag the cursor to. Required only by `action=left_click_drag`.", "type": "array"}, "status": {"description": "The status of the task. Required only by `action=terminate`.", "type": "string", "enum": ["success", "failure"]}}, "required": ["action"], "type": "object"}}}
</tools>

For each function call, return a json object with function name and arguments within <tool_call></tool_call> XML tags:
<tool_call>
{"name": <function-name>, "arguments": <args-json-object>}
</tool_call>

## Input Specification
- A screenshot of the current screen + an instruction describing the action to perform.

## Output Format
- Emit exactly one <tool_call></tool_call> block containing a single JSON object with the function name and its arguments.
- Coordinates are integers in a 1000x1000 space (x measured from the left edge, y from the top edge).

## Note
- Choose exactly one function whose name and arguments are defined in the tool schema above.
- Do not output anything other than the <tool_call></tool_call> block.
- If, and only if, the element named by the instruction genuinely does not exist anywhere in the screenshot, do not click: emit `terminate` with `status="failure"`.
- Strongly prefer locating and clicking the element; abstain only when you are confident it is truly absent.""",
        "mobile": r"""You are an autonomous GUI agent capable of operating on desktops, mobile devices, and web browsers. Your primary function is to analyze a screen capture, locate the target UI element, and issue the single appropriate action that fulfills the given instruction.

# Tools

You may call one or more functions to assist with the user query.

You are provided with function signatures within <tools></tools> XML tags:
<tools>
{"type": "function", "function": {"name": "mobile_use", "description": "Use a touchscreen to interact with a mobile device, and take screenshots.\n* This is an interface to a mobile device with touchscreen. You can perform actions like clicking, typing, swiping, etc.\n* Some applications may take time to start or process actions, so you may need to wait and take successive screenshots to see the results of your actions.\n* The screen's resolution is 1000x1000.\n* Make sure to click any buttons, links, icons, etc with the cursor tip in the center of the element. Don't click boxes on their edges unless asked.", "parameters": {"properties": {"action": {"description": "The action to perform. The available actions are:\n* `click`: Click the point on the screen with coordinate (x, y).\n* `long_press`: Press the point on the screen with coordinate (x, y) for specified seconds.\n* `terminate`: Abort the task without clicking and report the outcome in `status`. Use `status=\"failure\"` when the element named by the instruction does not exist anywhere on the screen, so no click could be correct.", "enum": ["click", "long_press", "terminate"], "type": "string"}, "coordinate": {"description": "(x, y): the point on the screen the action acts on — the element to touch, the region to scroll, or the object to start dragging. x is measured from the left edge and y from the top edge, both on a 0-1000 scale, so y gets larger toward the bottom of the screen. Required only by `action=click` and `action=long_press`.", "type": "array"}, "time": {"description": "The seconds to wait. Required only by `action=long_press`.", "type": "number"}, "status": {"description": "The status of the task. Required only by `action=terminate`.", "type": "string", "enum": ["success", "failure"]}}, "required": ["action"], "type": "object"}}}
</tools>

For each function call, return a json object with function name and arguments within <tool_call></tool_call> XML tags:
<tool_call>
{"name": <function-name>, "arguments": <args-json-object>}
</tool_call>

## Input Specification
- A screenshot of the current screen + an instruction describing the action to perform.

## Output Format
- Emit exactly one <tool_call></tool_call> block containing a single JSON object with the function name and its arguments.
- Coordinates are integers in a 1000x1000 space (x measured from the left edge, y from the top edge).

## Note
- Choose exactly one function whose name and arguments are defined in the tool schema above.
- Do not output anything other than the <tool_call></tool_call> block.
- If, and only if, the element named by the instruction genuinely does not exist anywhere in the screenshot, do not click: emit `terminate` with `status="failure"`.
- Strongly prefer locating and clicking the element; abstain only when you are confident it is truly absent.""",
        "web": r"""You are an autonomous GUI agent capable of operating on desktops, mobile devices, and web browsers. Your primary function is to analyze a screen capture, locate the target UI element, and issue the single appropriate action that fulfills the given instruction.

# Tools

You may call one or more functions to assist with the user query.

You are provided with function signatures within <tools></tools> XML tags:
<tools>
{"type": "function", "function": {"name": "browser_use", "description": "Use a mouse and keyboard to interact with a web browser, and take screenshots.\n* This is an interface to a web browser. You interact with the rendered web page by clicking, typing, scrolling, selecting options and navigating between pages.\n* Some pages may take time to load or process actions, so you may need to wait and take successive screenshots to see the results of your actions.\n* The screen's resolution is 1000x1000.\n* Make sure to click any buttons, links, icons, etc with the cursor tip in the center of the element. Don't click boxes on their edges unless asked.", "parameters": {"properties": {"action": {"description": "The action to perform. The available actions are:\n* `mouse_move`: Move the cursor to a specified (x, y) pixel coordinate on the screen.\n* `left_click`: Click the left mouse button at a specified (x, y) pixel coordinate on the screen.\n* `left_click_drag`: Press the left mouse button at the start point with coordinate (x, y), drag to the end point with coordinate2 (x2, y2), and release.\n* `right_click`: Click the right mouse button at a specified (x, y) pixel coordinate on the screen.\n* `middle_click`: Click the middle mouse button at a specified (x, y) pixel coordinate on the screen.\n* `double_click`: Double-click the left mouse button at a specified (x, y) pixel coordinate on the screen.\n* `triple_click`: Triple-click the left mouse button at a specified (x, y) pixel coordinate on the screen.\n* `select`: Select an option in a dropdown or list element at the specified (x, y) coordinate; the option to choose is given by text.\n* `terminate`: Abort the task without clicking and report the outcome in `status`. Use `status=\"failure\"` when the element named by the instruction does not exist anywhere on the screen, so no click could be correct.", "enum": ["mouse_move", "left_click", "left_click_drag", "right_click", "middle_click", "double_click", "triple_click", "select", "terminate"], "type": "string"}, "coordinate": {"description": "(x, y): The x (pixels from the left edge) and y (pixels from the top edge) coordinates to move the mouse to. Required only by `action=mouse_move`, `action=left_click`, `action=left_click_drag`, `action=right_click`, `action=middle_click`, `action=double_click`, `action=triple_click`, and `action=select`.", "type": "array"}, "coordinate2": {"description": "(x, y): The x (pixels from the left edge) and y (pixels from the top edge) coordinates to drag the cursor to. Required only by `action=left_click_drag`.", "type": "array"}, "text": {"description": "Required only by `action=select`.", "type": "string"}, "status": {"description": "The status of the task. Required only by `action=terminate`.", "type": "string", "enum": ["success", "failure"]}}, "required": ["action"], "type": "object"}}}
</tools>

For each function call, return a json object with function name and arguments within <tool_call></tool_call> XML tags:
<tool_call>
{"name": <function-name>, "arguments": <args-json-object>}
</tool_call>

## Input Specification
- A screenshot of the current screen + an instruction describing the action to perform.

## Output Format
- Emit exactly one <tool_call></tool_call> block containing a single JSON object with the function name and its arguments.
- Coordinates are integers in a 1000x1000 space (x measured from the left edge, y from the top edge).

## Note
- Choose exactly one function whose name and arguments are defined in the tool schema above.
- Do not output anything other than the <tool_call></tool_call> block.
- If, and only if, the element named by the instruction genuinely does not exist anywhere in the screenshot, do not click: emit `terminate` with `status="failure"`.
- Strongly prefer locating and clicking the element; abstain only when you are confident it is truly absent.""",
    }

    @staticmethod
    def _grounding_system(platform: str) -> str:
        g = ScaleCUAToolCallPrompt._GROUNDING_SYSTEM
        return g.get(platform, g["desktop"])

    @staticmethod
    def _platform_of(sample: Dict) -> str:
        # Pick the grounding tool by platform. Handle the various platform
        # namings across benchmarks:
        #   ScreenSpot-V2: "desktop"/"mobile"/"web"
        #   MMBench-GUI:   "os_windows"/"os_android"/"os_mac"/"os_linux"/"os_ios"/"os_web"
        #   ScreenSpot-Pro: application category (falls through to desktop; Pro is all desktop)
        # Its own method so the refusal subclass reuses this table instead of copying it --
        # a second copy would drift and serve some other platform's tool schema.
        group = sample.get("group") or []
        for g in group:
            gg = str(g).lower().replace("os_", "")
            if gg == "web":
                return "web"
            if gg in ("android", "ios", "iphone", "ipad", "mobile"):
                return "mobile"
            if gg in ("windows", "mac", "macos", "linux", "ubuntu", "desktop"):
                return "desktop"
        return "desktop"

    @staticmethod
    def generate_prompt(sample: Dict, image_width: int, image_height: int, think_mode: bool = True) -> List[Dict]:
        system = ScaleCUAToolCallPrompt._grounding_system(
            ScaleCUAToolCallPrompt._platform_of(sample))
        if os.environ.get("QWEN3VL_TEXT_FIRST") == "1":
            user_content = f"{sample['instruction']}<image>"
        else:
            user_content = f"<image>{sample['instruction']}"
        return [
            {"role": "system", "content": system},
            {"role": "user", "content": user_content},
        ]


class ScaleCUAToolCallRefusalPrompt(ScaleCUAToolCallPrompt):
    """
    `scalecua_toolcall` + an explicit refusal branch, for OSWorld-G.

    Why this exists: OSWorld-G ships 54 "element absent" cases that are scored
    correct only when the model ABSTAINS (data.py gives them a
    `{"refusal": -1}` gt; is_point_inside_element credits them only for a
    negative point). The parsing side already handles it --
    Qwen3VLPrompt.extract_coordinates maps `"action": "terminate"` to the
    [-1, -1] sentinel -- but the frozen grounding system prompt never tells the
    model refusal is an option, and `terminate` is not in its action enum. So the
    model always clicks and scores a hard 0/54. `--allow-refusal` does NOT help:
    it is wired only to the Anthropic backend's ClaudeCUARefusalPrompt
    (see eval.py).

    Everything except the appended clause is inherited, so the per-platform
    grounding system prompt the checkpoint was trained on stays byte-identical
    and the non-refusal samples see the same prompt as under `scalecua_toolcall` -- plus
    this clause. That is not free: the clause can make the model abstain on hard but
    PRESENT elements, which costs accuracy on the other 510 samples. Always read
    this prompt against plain `scalecua_toolcall` on the SAME checkpoint (both
    overall and the refusal subset) instead of assuming it is a pure win.
    """

    # Mirrors GUIOwlPrompt.INFEASIBLE_PREFIX (same sentinel, same tool name) so
    # the inherited extractor recognises it. `terminate` is deliberately outside
    # the trained action enum -- the trained grounding subset has no such action,
    # so it has to be introduced in prose here.
    REFUSAL_CLAUSE = (
        "\n- If, and only if, the instructed element genuinely does not exist anywhere in "
        "the screenshot, do not click: return "
        '<tool_call>\n{"name": "computer_use", "arguments": {"action": "terminate", "status": "failure"}}\n</tool_call>\n'
        "  Strongly prefer locating and clicking the element; abstain only when you are "
        "confident it is truly absent."
    )

    @staticmethod
    def generate_prompt(sample: Dict, image_width: int, image_height: int, think_mode: bool = True) -> List[Dict]:
        msgs = ScaleCUAToolCallPrompt.generate_prompt(sample, image_width, image_height, think_mode)
        # msgs[0] is the system turn; append rather than rebuild so the frozen
        # per-platform prompt selection stays in one place.
        msgs[0] = dict(msgs[0])
        msgs[0]["content"] = msgs[0]["content"] + ScaleCUAToolCallRefusalPrompt.REFUSAL_CLAUSE
        return msgs



class ScaleCUAToolCallSpatialPrompt(ScaleCUAToolCallPrompt):
    """
    `scalecua_toolcall` + an explicit RELATIONAL-REASONING clause, for UI-Vision's
    `spatial` subset.

    Why this exists: UI-Vision ships three grounding subsets. `basic` and `functional`
    name the target directly ("Measurements" / "Select measurement tool"), but `spatial`
    (1935 samples) asks a two-hop question instead:

        What is the element that is vertically above the "document" and closest to it?

    The model must (a) locate the quoted REFERENCE element, then (b) return the element
    standing in the named spatial relation to it -- not the reference itself. On one of
    our SFT checkpoints that subset scores 22.84% (icon 17.86%, 1407 errors) versus 31.57%
    on `basic` icon, and 46.7% of its errors land >10% of the image diagonal away from
    the target, i.e. the model is not missing narrowly -- it is answering the wrong
    element. The frozen grounding system prompt never mentions that an instruction may be
    relational, so a plausible failure mode is that the model grounds the quoted reference
    and stops.

    Unlike `scalecua_toolcall_refusal` -- which tried to summon a `terminate` action that
    is outside the trained schema and measured 0/564 emissions -- this clause introduces
    NO new action. The output contract is unchanged (`left_click` + a coordinate); only the
    reading of the instruction is steered. That is why prompting can plausibly move this
    subset where it could not move refusal.

    Everything except the appended clause is inherited, so the per-platform grounding
    system prompt the checkpoint was trained on stays byte-identical. That is not free:
    the clause can make the model over-apply relational reading on the DIRECT instructions
    in `basic` / `functional` (3544 samples) and click a neighbour instead of the named
    element. Always read this prompt against plain `scalecua_toolcall` on the SAME
    checkpoint, reporting `spatial` and `basic`/`functional` separately -- the overall
    number alone hides that trade.
    """

    SPATIAL_CLAUSE = (
        "\n- Some instructions are RELATIONAL: they describe the target by its position "
        "relative to another element, e.g. \"the element immediately to the right of X\", "
        "\"the element vertically above X and closest to it\". In that case, first locate the "
        "referenced element X, then click the DIFFERENT element that stands in the stated "
        "relation to it. Do not click X itself."
    )

    @staticmethod
    def generate_prompt(sample: Dict, image_width: int, image_height: int, think_mode: bool = True) -> List[Dict]:
        msgs = ScaleCUAToolCallPrompt.generate_prompt(sample, image_width, image_height, think_mode)
        # Append rather than rebuild so the frozen per-platform prompt selection stays in
        # one place (same contract as ScaleCUAToolCallRefusalPrompt).
        msgs[0] = dict(msgs[0])
        msgs[0]["content"] = msgs[0]["content"] + ScaleCUAToolCallSpatialPrompt.SPATIAL_CLAUSE
        return msgs


class ScaleCUAToolCallSchemaRefusalPrompt(ScaleCUAToolCallPrompt):
    """
    `scalecua_toolcall` served with the TRAINED refusal schema -- the eval-side counterpart of
    the `internvl2_5_{ubuntu,web,android}_refusal_grounding_v1` training templates
    (abstention via terminate(status="failure")). Use this for OSWorld-G on any checkpoint
    trained with those templates.

    Difference from `scalecua_toolcall_refusal` (keep both): that one is the PROMPTING
    PROBE -- it serves the plain trained schema and introduces `terminate` in prose only.
    Measured on one checkpoint: 0 terminate emissions out of 564, score bit-identical
    (0.7199 -> 0.7199). Prompting cannot summon an action that is outside the schema the
    model was trained under. This processor serves that schema instead: `terminate` is in
    the tool JSON (re-described for abstention) and two `## Note` bullets state when
    abstaining is correct -- byte-identical to training.

    Only the system turn differs from `scalecua_toolcall`. extract_coordinates
    (Qwen3VLPrompt: `args["action"] == "terminate"` -> the [-1, -1] refusal sentinel) and
    calculate_metrics are inherited unchanged.

    Reading the result: OSWorld-G is 54 refusal + 510 locatable, and an abstain-capable
    prompt can trade the 510 for the 54. Always report the refusal subset and the
    remainder separately, against plain `scalecua_toolcall` on the SAME checkpoint. The
    overall number alone hides that trade.
    """

    @staticmethod
    def generate_prompt(sample: Dict, image_width: int, image_height: int, think_mode: bool = True) -> List[Dict]:
        msgs = ScaleCUAToolCallPrompt.generate_prompt(sample, image_width, image_height, think_mode)
        # Swap only the system turn, reusing the parent's platform detection and user-turn
        # assembly (QWEN3VL_TEXT_FIRST etc.) so neither is duplicated here.
        g = ScaleCUAToolCallPrompt._GROUNDING_REFUSAL_SYSTEM
        platform = ScaleCUAToolCallPrompt._platform_of(sample)
        msgs[0] = dict(msgs[0], content=g.get(platform, g["desktop"]))
        return msgs


class ScaleCUAToolCallSchemaAbstainPrompt(ScaleCUAToolCallPrompt):
    """`scalecua_toolcall` served with the **abstain** schema -- counterpart of the training-side
    internvl2_5_{desktop,web,mobile}_abstain_grounding_v1 templates.

    Coexists with scalecua_toolcall_refusal_schema rather than replacing it: that one serves the
    terminate(status="failure") refusal schema, which existing checkpoints were trained on;
    this one serves the argument-less abstain schema. Which one to use depends on the schema the
    checkpoint was trained under -- serving the wrong one silently forfeits the 54-sample
    OSWorld-G refusal subset (9.6 pt).

    Why training uses a different action name: the same model is trained on both grounding and
    trajectory data, and in the planning data terminate(status="failure") means the **whole
    task** failed. Same token, opposite meaning: under joint training "element not found" could
    generalize to "terminate the task" -- a hard failure inside an agent loop.

    extract_coordinates needs no override: both abstention checks up the parent chain already
    accept terminate and abstain (purely additive for checkpoints trained on the other schemas,
    which have no abstain action).
    """

    @staticmethod
    def generate_prompt(sample: Dict, image_width: int, image_height: int, think_mode: bool = True) -> List[Dict]:
        msgs = ScaleCUAToolCallPrompt.generate_prompt(sample, image_width, image_height, think_mode)
        g = ScaleCUAToolCallPrompt._GROUNDING_ABSTAIN_SYSTEM
        platform = ScaleCUAToolCallPrompt._platform_of(sample)
        msgs[0] = dict(msgs[0], content=g.get(platform, g["desktop"]))
        return msgs


class Holo2Prompt(GUIOwlPrompt):
    """
    Prompt processor for Holo2 (H Company), e.g. Hcompany/Holo2-{4B,8B,30B-A3B}.
    Holo2-8B is fine-tuned from **Qwen3-VL-8B-Thinking**.

    Faithful to the official localization cookbook
    (github.com/hcompai/hai-cookbook, holo2/holo_2_localization_huggingface.ipynb):
      * NO system message -- one user turn: image, then the localization prompt
        followed by a newline and the target. Holo2's own chat template injects no
        default system block either, which is why generate_prompt returns
        {"role": "system", "content": None} (see models/qwen2vl.py:_format_prompt).
      * the prompt embeds the pydantic schema of

            class ClickCoordinates(BaseModel):
                x: int = Field(ge=0, le=1000, description="The x coordinate, normalized between 0 and 1000.")
                y: int = Field(ge=0, le=1000, description="The y coordinate, normalized between 0 and 1000.")

        interpolated with an f-string, so it lands in the prompt as a PYTHON DICT
        repr (single quotes, keys sorted by pydantic), not as JSON. Reproduced
        verbatim below rather than regenerated, so a pydantic upgrade cannot
        silently reword the prompt.
      * the model answers with a JSON object {"x": ..., "y": ...} in a 0-1000
        normalized space -> calculate_metrics is inherited from GUIOwlPrompt,
        which maps 0-1000 into the eval's resized-image pixel space (or scores in
        the official 0-1000 space when GROUNDING_OFFICIAL_SCORING=1).
      * the cookbook calls apply_chat_template(..., thinking=False) for
        localization, i.e. the assistant turn is PREFILLED with an
        already-closed think block. Set ASSISTANT_PREFIX='<think>\n\n</think>\n\n'
        when running it.

    Geometry (from the checkpoint's preprocessor_config.json):
      patch_size 16 * merge_size 2 -> IMAGE_FACTOR=32; size.longest_edge
      16777216 -> MAX_IMAGE_PIXELS=16777216.

    ⚠ OSWorld-G refusal subset: this protocol has NO abstain action, so all 54
    refusal samples (9.6%) are guaranteed misses and the ceiling is
    510/564 = 90.4%. The model card's 70.1% is on the same 564-sample file, so it
    is comparable; do not "fix" this by bolting an abstain clause onto the prompt.
    """

    # Verbatim from get_chat_messages() in the cookbook notebook, including the
    # five-space indent on the bullet line and the trailing "Your target is:".
    LOCALIZATION_PROMPT = (
        "Localize an element on the GUI image according to the provided target "
        "and output a click position.\n"
        "     * You must output a valid JSON following the format: "
        "{'properties': {'x': {'description': 'The x coordinate, normalized between 0 and 1000.', "
        "'maximum': 1000, 'minimum': 0, 'title': 'X', 'type': 'integer'}, "
        "'y': {'description': 'The y coordinate, normalized between 0 and 1000.', "
        "'maximum': 1000, 'minimum': 0, 'title': 'Y', 'type': 'integer'}}, "
        "'required': ['x', 'y'], 'title': 'ClickCoordinates', 'type': 'object'}\n"
        "     Your target is:"
    )

    @staticmethod
    def generate_prompt(sample: Dict, image_width: int, image_height: int, think_mode: bool = False) -> List[Dict]:
        return [
            {"role": "system", "content": None},
            {"role": "user",
             "content": f"<image>{Holo2Prompt.LOCALIZATION_PROMPT}\n{sample['instruction']}"},
        ]

    @staticmethod
    def extract_coordinates(response: str, think_mode: bool = False) -> Optional[List]:
        # Expected: {"x": 123, "y": 456}. Read the LAST x/y pair so a stray
        # example inside a think block cannot win over the final answer.
        xs = re.findall(r'["\']x["\']\s*:\s*(-?\d+(?:\.\d+)?)', response)
        ys = re.findall(r'["\']y["\']\s*:\s*(-?\d+(?:\.\d+)?)', response)
        if xs and ys:
            return [float(xs[-1]), float(ys[-1])]
        # Fallback: a bare "[x, y]" / "(x, y)" pair, same order as GUIOwlPrompt.
        return GUIOwlPrompt.extract_coordinates(response, think_mode)


class MolmoPointPrompt(BasicPrompt):
    """Prompt processor for **allenai/MolmoPoint-GUI-8B** (requires `--engine hf --model-type molmopoint`;
    vLLM cannot load it, see the docstring of models/molmopoint_hf.py).

    Matches the model card's Image Pointing Example verbatim:
      * **No system message**, a single user turn whose content order is **text first, then
        image** (the model card uses [{"type":"text",...},{"type":"image",...}]), so this sends
        f"{instruction}<image>" rather than the usual "<image>{instruction}".
        The chat_template renders content in order; reversing it is no longer the official setup.
      * The prompt is the **bare instruction** with no prefix. Molmo's template is style-driven
        (a "style: " prefix is added only when the content carries a `style` that is not in
        DEMO_STYLES), and `pointing`, used for GUI pointing, is in DEMO_STYLES, so bare text is right.

    Coordinates: the model outputs special tokens, not text coordinates. models/molmopoint_hf.py
    decodes them with `extract_image_points()` into a point in the **input image's pixel space**
    and serializes it as `{"x": ..., "y": ...}`. This class only reads those numbers back.

    ⚠ calculate_metrics is **deliberately not overridden** and inherits BasicPrompt's pixel
      containment test: the predicted point and the GT box are already in the same (resized)
      pixel space, and any rescaling would be wrong. For the same reason do **not** set
      COORD_SPACE=norm1000.

    ⚠ eval.py's process_response has a heuristic: when 0 < x < 1 and 0 < y < 1 it treats the
      prediction as normalized and multiplies it by the image size. A pixel coordinate can only
      fall into the open interval (0,1) at a sub-pixel position in the top-left corner, which is
      negligibly rare; if it does happen it gets mis-scaled. Noted here for reference.

    ⚠ No abstention protocol -> all 54 OSWorld-G refusal samples are misses, ceiling 510/564 = 90.4%.
    """

    _XY = re.compile(r'"x"\s*:\s*(-?\d+(?:\.\d+)?)\s*,\s*"y"\s*:\s*(-?\d+(?:\.\d+)?)')

    @staticmethod
    def generate_prompt(sample: Dict, image_width: int, image_height: int, think_mode: bool = False) -> List[Dict]:
        return [{"role": "user", "content": f"{sample['instruction']}<image>"}]

    @staticmethod
    def extract_coordinates(response: str, think_mode: bool = False) -> Optional[List]:
        if not response:
            return None            # the wrapper decoded no point / inference failed for this sample
        m = MolmoPointPrompt._XY.search(response)
        if not m:
            return None
        return [float(m.group(1)), float(m.group(2))]


# Register available prompt processors
PROMPT_PROCESSORS = {
    "gpt": GPTPrompt,
    "gpt_refusal": GPTRefusalPrompt,  # GPT normalized-coord + refusal (OSWorld-G)
    "claude": ClaudePrompt,        # free-form pixel/normalized prompt (baseline)
    "claude_cua": ClaudeCUAPrompt,  # computer-use tool protocol (matches paper numbers)
    "claude_cua_refusal": ClaudeCUARefusalPrompt,  # computer-use + refusal (OSWorld-G)
    "qwen2_5_vl": Qwen2_5_VLPrompt,
    "scalecua": ScaleCUAPrompt,
    "infigui-g1": InfiguiG1Prompt,
    "groundcua": GroundCUAPrompt,
    "infigui-r1": InfiguiR1Prompt,
    "gta1": GTAPrompt,
    "opencua": OpenCUAPrompt,
    "jedi": BasicPrompt, # same as system prompt, with "wait" action
    "guig2": GUIG2Prompt,
    "guig1": GUIG1Prompt, # Yuqi-Zhou/GUI-G1-3B-v1
    "uground-v1": UGroundV1Prompt,
    "segui": SEGUIPrompt,
    "uitars": UITARSPrompt,
    "uir1e": UIR1EPrompt,
    "guiactor": GUIActorPrompt,
    "phiground": PhiGroundPrompt,
    "guiowl": GUIOwlPrompt,  # GUI-Owl-1.5 (Qwen3-VL), 0-1000 coord space
    "holo2": Holo2Prompt,  # Holo2 (Qwen3-VL-Thinking), cookbook JSON {x,y} 0-1000
    "uivenus": UIVenusPrompt,  # UI-Venus-1.5 (Qwen3-VL-MoE), 0-1000 point/box
    "qwen3vl": Qwen3VLPrompt,  # base Qwen3-VL-Instruct, official cookbook computer_use format
    "qwen3vl_point": Qwen3VLPointPrompt,  # base Qwen3-VL-Instruct, native point_2d grounding format
    # Eval side of internvl_grounding_point_*_v1: <point> target, answers point_2d JSON, 0-1000.
    # One prompt covers all benchmarks; the platform is dispatched per sample group
    # (same approach as scalecua_toolcall), switching surface/verb accordingly.
    "element_point": ElementPointPrompt,
    "molmopoint": MolmoPointPrompt,  # allenai/MolmoPoint-GUI-8B, grounding-token output, needs --engine hf
    "scalecua_toolcall": ScaleCUAToolCallPrompt,  # ScaleCUA-recipe Qwen3-VL SFT: per-platform grounding tool_call, 0-1000
    "scalecua_toolcall_refusal": ScaleCUAToolCallRefusalPrompt,  # same + terminate in PROSE only -- prompting probe, measured 0/564 emissions
    # For OSWorld-G on checkpoints trained with *_refusal_grounding_v1: terminate(status="failure")
    # is in the trained tool schema, byte-identical to training.
    "scalecua_toolcall_refusal_schema": ScaleCUAToolCallSchemaRefusalPrompt,
    # abstain schema (argument-less abstain action), for checkpoints trained with
    # *_abstain_grounding_v1; *_refusal_grounding_v1 checkpoints use refusal_schema above.
    "scalecua_toolcall_abstain_schema": ScaleCUAToolCallSchemaAbstainPrompt,
    # UI-Vision spatial probe: appends a relational-reasoning clause to the system prompt, no new action.
    "scalecua_toolcall_spatial": ScaleCUAToolCallSpatialPrompt,

}


def get_prompt_processor(name: str):
    """
    Retrieve a prompt processor by name.
    """
    if name not in PROMPT_PROCESSORS:
        raise ValueError(f"Unknown prompt processor: {name}. Available: {list(PROMPT_PROCESSORS.keys())}")
    return PROMPT_PROCESSORS[name]