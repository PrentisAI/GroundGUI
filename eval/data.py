import json
import os
import math
from typing import Dict, List, Tuple, Union


def load_benchmark_info(benchmark_name: str) -> Dict:
    """
    Loads benchmark information.
    """
    info_path = "./dataset_info.json"
    if not os.path.exists(info_path):
        raise FileNotFoundError(f"Benchmark info file not found: {info_path}")
    
    with open(info_path, "r", encoding="utf-8") as f:
        info = json.load(f)
    
    if benchmark_name not in info:
        raise ValueError(f"Unknown benchmark: {benchmark_name}. Available: {list(info.keys())}")
    
    return info[benchmark_name]


def load_dataset(benchmark_info: Dict) -> List[Dict]:
    """
    Loads the dataset.
    """
    data_path = benchmark_info["data_path"]
    dataset = []

    if os.path.isfile(data_path):
        # Single file
        data_name = os.path.basename(data_path).split('.')[0]
        if data_path.endswith(".json"):
            with open(data_path, "r", encoding="utf-8") as f:
                dataset = json.load(f)
        elif data_path.endswith(".jsonl"):
            with open(data_path, "r", encoding="utf-8") as f:
                for line in f:
                    sample = json.loads(line)
                    dataset.append(sample)
        for sample in dataset:
            sample["_data_name"] = data_name
    elif os.path.isdir(data_path):
        # Directory, load all json files
        json_files = [f for f in os.listdir(data_path) if f.endswith('.json')]
        for json_file in json_files:
            file_path = os.path.join(data_path, json_file)
            data_name = os.path.basename(file_path).split('.')[0]
            with open(file_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            for sample in data:
                sample["_data_name"] = data_name
            dataset.extend(data)
    else:
        raise FileNotFoundError(f"Data path not found: {data_path}")
    
    return dataset


def standardize_sample(sample: Dict, benchmark_name: str) -> Dict:
    """
    Standardizes a sample to a uniform format.
    """
    if benchmark_name == "screenspot-pro":
        # Convert screenspot-pro format to standard format
        bbox = sample.get("bbox", [])
        if bbox and len(bbox) == 4:
            bbox_dict = {
                "x1": bbox[0],
                "y1": bbox[1], 
                "x2": bbox[2],
                "y2": bbox[3]
            }
        else:
            raise ValueError(f"Invalid bbox: {bbox}")
        
        standardized = {
            "images": sample["img_filename"],
            "instruction": sample["instruction"],
            "bbox": bbox_dict,
            "group": [sample["group"].lower()]  # Convert to list format
        }
        
        # If ui_type exists, add it to group
        if "ui_type" in sample:
            standardized["group"].append(sample["ui_type"].lower())
            
        return standardized
    elif benchmark_name == "screenspot-v2":
        # Convert screenspot-v2 format to standard format
        bbox = sample.get("bbox", [])
        if bbox and len(bbox) == 4:
            bbox_dict = {
                "x1": bbox[0],
                "y1": bbox[1], 
                "x2": bbox[0] + bbox[2],
                "y2": bbox[1] + bbox[3]
            }
        else:
            raise ValueError(f"Invalid bbox: {bbox}")
        
        standardized = {
            "images": sample["img_filename"],
            "instruction": sample["instruction"],
            "bbox": bbox_dict,
            "group": [sample["_data_name"].split('_')[1].lower()]  # Convert to list format
        }
        
        # If ui_type exists, add it to group
        if "data_type" in sample:
            standardized["group"].append(sample["data_type"].lower())
            
        return standardized
    elif benchmark_name == "ui-vision":
        cat2platform = {
            "Education":[
                "Anki",
                "Zotero",
                "GrassGIS",
                "Calibre",
                "Audacity",
                "QGIS",
                "OpenBoard",
                "Mendeley"
            ],
            "Browsers": [
                "Brave",
                "Chromium",
                "Mozilla Firefox",
                "DuckDuckGo"
            ],
            "Development": [
                "VSCode",
                "Atom",
                "FreeCAD",
                "Eclipse",
                "NetBeans",
                "PyCharm",
                "IntelliJ IDEA",
                "Brackets",
                "Geany",
                "Bluefish",
                "KDevelop",
                "Komodo Edit",
                "Code::Blocks",
                "Qt Creator",
                "Arduino IDE",
                "Spyder",
                "Ubuntu Terminal",
                "Conky",
                "Bash",
                "Gedit"
            ],
            "Productivity": [
                "LibreOffice Calc",
                "LibreOffice Draw",
                "LibreOffice Impress",
                "LibreOffice Writer",
                "draw.io",
                "Joplin",
                "OpenProject",
                "Affine",
                "Zulip",
                "PDFedit",
                "OnlyOffice Calendar",
                "OnlyOffice Document Editor",
                "OnlyOffice Forms",
                "OnlyOffice PDF Forms",
                "OnlyOffice Presentation",
                "OnlyOffice Spreadsheet",
                "Nextcloud",
                "Gnumeric",
                "Simplenote",
                "Cryptomator",
                "WeKan",
                "7-Zip",
                "GnuCash",
                "Bitwarden",
                "Metabase",
                "Jitsi",
                "Flameshot",
                "Nemo"
            ],
            "Creativity": [
                "Blender",
                "GIMP",
                "Inkscape",
                "Krita",
                "Darktable",
                "FontForge",
                "MuseScore",
                "Scribus",
                "OpenShot",
                "OBS Studio",
                "Lightworks",
                "Shotcut",
                "Natron",
                "OpenToonz",
                "WordPress"
            ],
            "Entertainment": [
                "VLC Media Player",
                "Kodi",
                "Element",
                "Signal",
                "Mastodon",
                "Lemmy",
                "Matrix",
                "Emby"
            ]
        }
        platform2cat = {}
        for k, v in cat2platform.items():
            for vv in v:
                assert vv not in platform2cat, f"Duplicate platform: {vv}"
                platform2cat[vv.lower()] = k

        # Convert ui-vision format to standard format
        bbox = sample.get("bbox", [])
        if bbox and len(bbox) == 4:
            bbox_dict = {
                "x1": bbox[0],
                "y1": bbox[1], 
                "x2": bbox[2],
                "y2": bbox[3]
            }
        else:
            raise ValueError(f"Invalid bbox: {bbox}")
        
        standardized = {
            "images": sample["image_path"],
            "instruction": sample["prompt_to_evaluate"],
            "bbox": bbox_dict,
            "group": [[sample["_data_name"].split('_')[-1].lower()]]  # Convert to list format
        }
        
        platform = sample["platform"]
        if platform.lower() in platform2cat:
            standardized["group"][0].append(platform2cat[platform.lower()])
            
        return standardized
    elif benchmark_name == "ui-vision-layout":
        # UI-Vision's SECOND task (README: Element Grounding / Layout Grounding /
        # Action Prediction). Registered as its own benchmark rather than folded
        # into "ui-vision" on purpose: mixing 311 region samples into the 5,479
        # element samples would change that metric's denominator and break
        # comparability with published element-grounding numbers.
        #
        # Schema differs from element_grounding: there is no `prompt_to_evaluate`.
        # Each record is a REGION -- {name, explanation, bbox, platform,
        # image_size} -- e.g. name="Formatting Tools",
        # explanation="This region contains tools for formatting text, including
        # font, size, and alignment options.", bbox = a 707x31 strip.
        #
        # PROTOCOL WARNING: UI-Vision ships no eval code here, so the query we
        # build is OURS, not the paper's -- these numbers are our own baseline
        # and are NOT comparable to published layout-grounding results. Default is
        # the bare region name (mirrors how element_grounding_basic reads); set
        # UIVISION_LAYOUT_WITH_EXPLANATION=1 to append the explanation.
        bbox = sample.get("bbox", [])
        if not (bbox and len(bbox) == 4):
            raise ValueError(f"Invalid bbox: {bbox}")
        name = (sample.get("name") or "").strip()
        if not name:
            raise ValueError(f"ui-vision-layout sample has no `name`: {sample.get('image_path')}")
        instruction = name
        if os.environ.get("UIVISION_LAYOUT_WITH_EXPLANATION") == "1":
            expl = (sample.get("explanation") or "").strip()
            if expl:
                instruction = f"{name}: {expl}"

        standardized = {
            "images": sample["image_path"],
            "instruction": instruction,
            "bbox": {"x1": bbox[0], "y1": bbox[1], "x2": bbox[2], "y2": bbox[3]},
            "group": [["layout"]],  # single group; 77 platforms would explode the stats tree
        }
        return standardized
    elif benchmark_name == "mmbench-gui":
        # Convert mmbench-gui format to standard format
        image_size = sample["image_size"]
        bbox = sample.get("bbox", [])
        if bbox and len(bbox) == 4:
            bbox_dict = {
                "x1": bbox[0] * image_size[0],
                "y1": bbox[1] * image_size[1], 
                "x2": bbox[2] * image_size[0],
                "y2": bbox[3] * image_size[1]
            }
        else:
            raise ValueError(f"Invalid bbox: {bbox}")
        
        standardized = {
            "images": os.path.join(sample["platform"], sample["image_path"]),
            "instruction": sample["instruction"],
            "bbox": bbox_dict,
            "group": [sample["platform"].lower()]  # Convert to list format
        }
        
        # If ui_type exists, add it to group
        if "grounding_type" in sample:
            standardized["group"].append(sample["grounding_type"].lower())
            
        return standardized
    elif benchmark_name in ("osworld-g", "osworld-g-refined"):
        # Convert osworld-g format to standard format
        
        # buckets.json sits next to the OSWorld-G annotation file, so derive its
        # path from dataset_info.json instead of hardcoding one.
        bucket_path = os.path.join(
            # Look it up via the CURRENT benchmark, not a hardcoded "osworld-g": if the
            # osworld-g and osworld-g-refined entries point to different directories,
            # a hardcoded name would silently read the wrong buckets.
            os.path.dirname(load_benchmark_info(benchmark_name)["data_path"]),
            "buckets.json",
        )
        with open(bucket_path, 'r') as f:
            bucket_info = json.load(f)
        
        image_size = sample["image_size"]
        bbox = sample.get("box_coordinates", [])
        bbox_type = sample['box_type']
        if bbox_type == 'bbox':
            bbox_dict = {
                "x1": bbox[0],
                "y1": bbox[1],
                "x2": bbox[0] + bbox[2],
                "y2": bbox[1] + bbox[3]
            }
        elif bbox_type == 'polygon':
            # pair bbox coordinates
            polygon = []
            for i in range(0, len(bbox), 2):
                polygon.append([bbox[i], bbox[i+1]])
            
            bbox_dict = {
                "polygon": polygon
            }
        elif bbox_type == 'refusal':
            bbox_dict = {
                "refusal": -1,
            }
        else:
            raise ValueError(f"Invalid bbox: {bbox}")
        
        # Subgrouping is driven by box_type FIRST, then by GUI_types.
        #
        # Upstream GroundCUA groups by GUI_types alone, with a synthetic bucket
        # `bucket_info['refusal'] = ["/"]`. That makes the "refusal" subgroup mean
        # "sample carries no GUI type", not "sample is a refusal case": in
        # OSWorld-G.json 54 samples have box_type == 'refusal', but only 9 of them
        # have GUI_types == "/", so the reported `refusal total` would be 9 while the
        # real refusal subset is 54. (No non-refusal sample carries "/", so the two
        # sets nest cleanly.)
        #
        # The other 45 refusal samples would fall into the four functional buckets and
        # depress them by a group-dependent amount, because a refusal case is scored
        # correct only when the model ABSTAINS -- so under a prompt with no abstain
        # action every one of them is a guaranteed miss:
        #     text_matching              21/261  ( 8.0%)
        #     element_recognition        24/330  ( 7.3%)
        #     layout_understanding       14/253  ( 5.5%)
        #     fine_grained_manipulation  17/152  (11.2%)
        # A bucket named `text_matching` cannot measure text matching on samples
        # whose correct answer is "do not click", so refusal samples are kept out of
        # the functional buckets entirely.
        #
        # NOTE: per-subgroup numbers therefore differ from upstream GroundCUA
        # (`refusal` 9 -> 54, the four functional buckets each rise by the share
        # above). The top-level total/correct over all 564 samples is identical, so
        # overall accuracy stays comparable; per-subgroup numbers do not.
        groups = []
        if bbox_type == 'refusal':
            groups.append('refusal')
        else:
            for k, v in bucket_info.items():
                for sub_group in sample["GUI_types"]:
                    if sub_group in v and k not in groups:
                        groups.append(k)

        assert len(groups) > 0, (
            f"Cannot find group for box_type={bbox_type!r} "
            f"GUI_types={sample['GUI_types']!r}"
        )
        
        standardized = {
            "images": sample["image_path"],
            "instruction": sample["instruction"],
            "bbox": bbox_dict,
            "group": [groups]
        }
        
            
        return standardized
    
    elif benchmark_name == "i2e-bench":
        # Convert i2e-bench format to standard format
        bbox = sample.get("bounding_box", [])
        if bbox and len(bbox) == 4:
            bbox_dict = {
                "x1": bbox[0],
                "y1": bbox[1], 
                "x2": bbox[2],
                "y2": bbox[3]
            }
        else:
            raise ValueError(f"Invalid bbox: {bbox}")
        
        standardized = {
            "images": sample["image"],
            "instruction": sample["instruction"],
            "bbox": bbox_dict,
            "group": [sample["annotations"]["instr_type"].lower()]  # Convert to list format
        }
        
        # If ui_type exists, add it to group
        if "source" in sample:
            standardized["group"].append(sample["source"].lower())
            
        return standardized
    else:
        raise ValueError(f"Unknown benchmark: {benchmark_name}.")


def resize_image(width: int, height: int, max_pixels: int) -> Tuple[int, int]:
    """
    Resizes an image so that its total number of pixels does not exceed the specified maximum,
    while maintaining the aspect ratio.
    """
    current_pixels = width * height
    
    if current_pixels <= max_pixels:
        return width, height
    
    scale_factor = math.sqrt(max_pixels / current_pixels)
    new_width = round(width * scale_factor)
    new_height = round(height * scale_factor)
    
    return new_width, new_height


def create_hierarchical_stats(metric_keys: Dict[str, str]) -> Dict:
    """
    Creates a hierarchical statistics structure.
    
    Args:
        metric_keys: Mapping of metric keys to their types.
    """
    stats = {"subgroups": {}}
    
    # Dynamically create metric fields
    for key, key_type in metric_keys.items():
        stats[key] = 0
        if key_type == "avg":
            # Average metrics require additional count and sum fields
            stats[f"{key}_sum"] = 0.0
            stats[f"{key}_count"] = 0
            stats[f"avg_{key}"] = 0.0
    
    return stats


def update_stats(stats: Dict, metrics: Dict, metric_keys: Dict[str, str]):
    """
    Updates statistics.
    
    Args:
        stats: Statistics dictionary.
        metrics: Metrics dictionary.
        metric_keys: Mapping of metric keys to their types.
    """
    for key, key_type in metric_keys.items():
        if key in metrics and metrics[key] is not None:
            if key_type in ["sum", "count"]:
                stats[key] += metrics[key]
            elif key_type == "avg":
                # Special handling for average metrics
                stats[f"{key}_sum"] += metrics[key]
                stats[f"{key}_count"] += 1


def finalize_stats(stats: Dict, metric_keys: Dict[str, str], accuracy_pairs: List[Tuple[str, str]]):
    """
    Finalizes statistical calculations, computing final ratios and averages.
    
    Args:
        stats: Statistics dictionary.
        metric_keys: Mapping of metric keys to their types.
        accuracy_pairs: Pairs for accuracy calculation.
    """
    # Calculate accuracy
    for numerator, denominator in accuracy_pairs:
        if denominator in stats and stats[denominator] > 0:
            # Dynamically generate accuracy key name, no hardcoding special handling
            accuracy_key = f"{numerator}_accuracy"
            stats[accuracy_key] = stats.get(numerator, 0) / stats[denominator]
    
    # Calculate averages
    for key, key_type in metric_keys.items():
        if key_type == "avg":
            count_key = f"{key}_count"
            if count_key in stats and stats[count_key] > 0:
                sum_key = f"{key}_sum"
                avg_key = f"avg_{key}"
                stats[avg_key] = stats[sum_key] / stats[count_key]
    
    # Recursively process subgroups
    for subgroup_stats in stats["subgroups"].values():
        finalize_stats(subgroup_stats, metric_keys, accuracy_pairs)


def reorder_stats_for_output(stats: Dict) -> Dict:
    """
    Reorders statistics, moving subgroups to the end.
    
    Args:
        stats: Original statistics dictionary.
        
    Returns:
        Reordered statistics dictionary.
    """
    ordered_stats = {}
    
    # Add all non-subgroup keys first
    for key, value in stats.items():
        if key != "subgroups":
            ordered_stats[key] = value
    
    # Add subgroups last, and recursively process subgroups
    if "subgroups" in stats:
        ordered_subgroups = {}
        for group_name, group_stats in stats["subgroups"].items():
            ordered_subgroups[group_name] = reorder_stats_for_output(group_stats)
        ordered_stats["subgroups"] = ordered_subgroups
    
    return ordered_stats


def calculate_hierarchical_statistics(results: Dict, dataset: List[Dict], prompt_processor) -> Dict:
    """
    Calculates hierarchical statistics.
    
    Args:
        results: Results dictionary.
        dataset: Dataset.
        prompt_processor: Prompt processor, used to get metric definitions.
    """
    # Get metric definitions from prompt processor
    metric_keys = prompt_processor.get_metric_keys()
    accuracy_pairs = prompt_processor.get_accuracy_pairs()
    
    overall_stats = create_hierarchical_stats(metric_keys)
    
    # Update statistics for each sample
    for idx, sample in enumerate(dataset):
        if idx not in results:
            continue
            
        result = results[idx]
        groups = sample.get("group", ["default"])
        
        # Extract metrics from result, handle possible list format (for compatibility)
        metrics = {}
        for key in metric_keys.keys():
            value = result.get(key, 0)
            if isinstance(value, list) and len(value) > 0:
                value = value[0]
            metrics[key] = value
        
        # Update overall statistics
        update_stats(overall_stats, metrics, metric_keys)
        
        # Recursively update hierarchical statistics
        _update_hierarchical_stats(overall_stats, groups, metrics, metric_keys, 0)
    
    # Finalize all statistical calculations
    finalize_stats(overall_stats, metric_keys, accuracy_pairs)
    
    return overall_stats


def _update_hierarchical_stats(stats: Dict, groups: List[Union[str, List[str]]], metrics: Dict, metric_keys: Dict[str, str], level: int):
    """
    Recursively updates hierarchical statistics.
    Supports multi-category cases, i.e., groups[level] can be a string or a list of strings.
    """
    if level >= len(groups):
        return
    
    current_level = groups[level]
    
    # Handle multi-category cases: if the current level is a list, perform the same operation for each element
    if isinstance(current_level, list):
        group_names = current_level
    else:
        group_names = [current_level]
    
    # Update statistics for each category at the current level
    for group_name in group_names:
        # Ensure subgroup exists
        if group_name not in stats["subgroups"]:
            stats["subgroups"][group_name] = create_hierarchical_stats(metric_keys)
        
        # Update statistics for the current level
        update_stats(stats["subgroups"][group_name], metrics, metric_keys)
        
        # Recursively process the next level
        _update_hierarchical_stats(stats["subgroups"][group_name], groups, metrics, metric_keys, level + 1)


def print_hierarchical_stats(stats: Dict, indent: int = 0, metric_keys: Dict[str, str] = None, accuracy_pairs: List[Tuple[str, str]] = None):
    """
    Prints hierarchical statistics results.
    """
    prefix = "  " * indent
    
    # If metric definitions are not provided, try to infer from stats
    if metric_keys is None:
        # Print basic numerical metrics
        for key, value in stats.items():
            if key != "subgroups" and not key.endswith("_sum") and not key.endswith("_count"):
                if isinstance(value, (int, float)):
                    if key.endswith("_accuracy"):
                        print(f"{prefix}{key.title()}: {value:.4f}")
                    elif key.startswith("avg_"):
                        if value > 0:
                            print(f"{prefix}{key.replace('_', ' ').title()}: {value:.2f}")
                    else:
                        print(f"{prefix}{key.title()}: {value}")
    else:
        # Print basic metrics
        for key, key_type in metric_keys.items():
            if key in stats:
                print(f"{prefix}{key.title()}: {stats[key]}")
        
        # Print accuracy
        if accuracy_pairs:
            for numerator, denominator in accuracy_pairs:
                accuracy_key = f"{numerator}_accuracy"
                if accuracy_key in stats:
                    print(f"{prefix}{accuracy_key.replace('_', ' ').title()}: {stats[accuracy_key]:.4f}")
        
        # Print averages
        for key, key_type in metric_keys.items():
            if key_type == "avg":
                avg_key = f"avg_{key}"
                if avg_key in stats and stats[avg_key] > 0:
                    print(f"{prefix}{avg_key.replace('_', ' ').title()}: {stats[avg_key]:.2f}")
    
    if stats.get("subgroups"):
        for group_name, group_stats in stats["subgroups"].items():
            print(f"{prefix}{group_name}:")
            print_hierarchical_stats(group_stats, indent + 1, metric_keys, accuracy_pairs) 
