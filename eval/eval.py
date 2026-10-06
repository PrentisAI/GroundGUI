import argparse
import os
import json
import random
from datetime import datetime
from typing import Dict
from tqdm import tqdm
from PIL import Image
from qwen_vl_utils import smart_resize

# Import custom modules

from data import (
    load_benchmark_info, 
    load_dataset, 
    standardize_sample,
    resize_image,
    calculate_hierarchical_statistics,
    print_hierarchical_stats,
    reorder_stats_for_output
)
from prompts import get_prompt_processor, PROMPT_PROCESSORS


def prepare_sample_data(sample: Dict, benchmark_info: Dict, max_image_pixels: int, prompt_processor, think_mode: bool) -> tuple:
    """
    Prepares data for a single sample.
    """
    # Load image
    image_path = os.path.join(benchmark_info["image_root"], sample["images"])
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image not found: {image_path}")
    
    image = Image.open(image_path)
    original_width, original_height = image.size
    
    # Process bounding box coordinates
    bbox = sample["bbox"].copy()
    
    # Resize image
    # new_width, new_height = resize_image(original_width, original_height, max_image_pixels)
    
    # Further resize using smart_resize. factor = patch_size * spatial_merge_size:
    # 28 for Qwen2.5-VL (14*2), 32 for Qwen3-VL (16*2). Using the wrong factor forces
    # the vLLM processor to re-resize (double resize -> blur on tiny targets) and puts
    # the gt/pred comparison in a slightly different grid. Override via IMAGE_FACTOR
    # (set 32 for Qwen3-VL models, e.g. GUI-Owl / UI-Venus).
    factor = int(os.environ.get("IMAGE_FACTOR", 28))
    new_height, new_width = smart_resize(original_height, original_width, max_pixels=max_image_pixels, factor=factor)
    image = image.resize((new_width, new_height))
    
    # Update bounding box coordinates to the new image dimensions
    scale_x = new_width / original_width
    scale_y = new_height / original_height
    if all(key in bbox.keys() for key in ["x1", "y1", "x2", "y2"]):
        bbox["x1"] = bbox["x1"] * scale_x
        bbox["y1"] = bbox["y1"] * scale_y
        bbox["x2"] = bbox["x2"] * scale_x
        bbox["y2"] = bbox["y2"] * scale_y
    elif all(key in bbox.keys() for key in ["polygon"]):
        bbox["polygon"] = [[p[0] * scale_x, p[1] * scale_y] for p in bbox["polygon"]]
    
    # Generate prompt
    messages = prompt_processor.generate_prompt(sample, new_width, new_height, think_mode)
    
    return image, messages, bbox


def process_response(sample: Dict, response: str, bbox: Dict, prompt_processor, think_mode: bool) -> Dict:
    """
    Processes the model's response.
    """
    # Extract coordinates
    predictions = prompt_processor.extract_coordinates(response, think_mode)

    # Map predicted coordinates into the (resized) image-pixel space used by the GT.
    #   COORD_SPACE=norm1000 : model emits Qwen3-VL-native [0,1000] normalized coords;
    #                          rescale pred/1000 * processed_imgsize. Only for prompt
    #                          processors whose calculate_metrics does NOT map 0-1000
    #                          itself: the guiowl / qwen3vl / scalecua_toolcall family
    #                          already does, and setting it there rescales twice.
    #   otherwise            : upstream GroundCUA behaviour — only [0,1]-normalized preds are
    #                          scaled up; absolute-pixel preds pass through unchanged.
    coord_space = os.environ.get("COORD_SPACE", "").strip().lower()
    if (
        predictions is not None
        and coord_space in ("norm1000", "1000")
        and predictions[0] >= 0
        and predictions[1] >= 0
    ):
        processed_imgsize = sample.get("processed_imgsize")
        if processed_imgsize:
            predictions = [
                predictions[0] / 1000.0 * processed_imgsize[0],
                predictions[1] / 1000.0 * processed_imgsize[1],
            ]
    elif predictions is not None and 0 < predictions[0] < 1 and 0 < predictions[1] < 1:
        # If normalized coordinates, convert to absolute
        processed_imgsize = sample["processed_imgsize"]
        if processed_imgsize:
            predictions[0] *= processed_imgsize[0]
            predictions[1] *= processed_imgsize[1]

    metrics = prompt_processor.calculate_metrics(sample, predictions, bbox)
    
    # Construct result
    result = {
        "image": sample["images"],
        "instruction": sample["instruction"],
        "gt_bbox": bbox,
        "response": response,
        **metrics
    }
    
    return result


def main():
    parser = argparse.ArgumentParser(description="General UI Element Localization Evaluation Framework")
    
    parser.add_argument("model_path", type=str, help="Path to the model")
    parser.add_argument("--benchmark", "-b", type=str, default="screenspot-pro", help="Name of the benchmark to evaluate")
    parser.add_argument("--prompt", type=str, default="groundcua", help="Name of the prompt processor to use (see prompts.PROMPT_PROCESSORS)")
    parser.add_argument("--engine", type=str, default="vllm", choices=['vllm', 'hf', 'api', 'anthropic'], help="Name of the engine to use ('api' = OpenAI-compatible, 'anthropic' = Claude)")
    parser.add_argument("--reasoning-effort", type=str, default="", help="OpenAI reasoning effort for gpt-5/o-series (minimal|low|medium|high). Empty = API default.")
    parser.add_argument("--thinking-budget", type=int, default=0, help="Anthropic extended-thinking budget_tokens (0 = off)")
    parser.add_argument("--computer-use", type=int, default=1, help="Anthropic: use the computer-use tool for grounding (1=on, 0=free-form)")
    parser.add_argument("--allow-refusal", type=int, default=0, help="Anthropic CUA: allow the model to abstain/refuse (for OSWorld-G refusal cases)")
    parser.add_argument("--max-workers", type=int, default=8, help="Concurrent requests for the 'api'/'anthropic' engine")
    parser.add_argument("--base-url", type=str, default="", help="Override base_url (else uses OPENAI_BASE_URL / ANTHROPIC_BASE_URL / provider default)")
    parser.add_argument("--tensor-parallel", "-tp", type=int, default=4, help="Tensor parallelism size")
    parser.add_argument("--batch-size", type=int, default=16, help="Batch size")
    parser.add_argument("--max-num-seqs", type=int, default=16, help="Maximum number of sequences in a batch")
    parser.add_argument("--max-tokens", type=int, default=1024, help="Maximum number of tokens to generate")
    parser.add_argument("--no-cache", action="store_true", default=False, help="Whether to disable caching")
    parser.add_argument("--max-image-tokens", "-mit", type=int, default=5600, help="Maximum image tokens")
    parser.add_argument("--think-mode", type=int, default=0, help="Whether to enable thinking mode (1=enable, 0=disable)")
    parser.add_argument("--debug-mode", type=int, default=0, help="Whether to enable debug mode (1=enable, 0=disable)")
    parser.add_argument("--model-type", type=str, default='qwen2.5vl', choices=['qwen2.5vl', 'qwen2vl', 'molmopoint'], help="Output model name, extracted from model path if not specified")
    parser.add_argument("--temperature", "-t", type=float, default=0.0, help="Generation temperature")
    parser.add_argument("--output-dir", type=str, default=None, help="Output directory for evaluation results (default: ./output/{model_name}/{benchmark})")
    parser.add_argument("--cache-dir", type=str, default=None, help="Cache directory for processed data (default: ./cache)")
    
    args = parser.parse_args()
    
    # Convert flag arguments to boolean
    think_mode = bool(args.think_mode)
    debug_mode = bool(args.debug_mode)
    
    print(f"Starting evaluation - Benchmark: {args.benchmark}, Prompt: {args.prompt}")
    
    # Load benchmark information
    benchmark_info = load_benchmark_info(args.benchmark)
    print(f"Loaded benchmark: {benchmark_info['name']}")

    
    # Get prompt processor
    prompt_processor = get_prompt_processor(args.prompt)
    print(f"Using prompt processor: {args.prompt}")

    
    # Set output directory
    
    if os.path.basename(args.model_path).startswith('checkpoint'):
        model_name = os.path.basename(os.path.dirname(args.model_path)) + '_chkp' + os.path.basename(args.model_path).split('-')[-1] 
    elif os.path.basename(args.model_path) == 'huggingface':
        model_name = os.path.basename(os.path.dirname(os.path.dirname(os.path.dirname(args.model_path)))) + '_chkp' + os.path.basename(os.path.dirname(os.path.dirname(args.model_path))).split('_')[-1]
    else:
        model_name = os.path.basename(os.path.normpath(args.model_path))
        
    
    if args.output_dir:
        output_dir = os.path.expanduser(args.output_dir)
    else:
        output_dir = os.path.join("output", model_name, args.benchmark)
        if args.engine == 'hf':
            output_dir += '_hf'
    output_dir = os.path.abspath(output_dir)
    print('Output directory:', output_dir)
    os.makedirs(output_dir, exist_ok=True)
    
    # Initialize model
    max_image_pixels = int(os.environ.get("MAX_IMAGE_PIXELS", 12845056))  # default Qwen2.5-VL native max; override via env (e.g. ScaleCUA needs 2109744)
    print(f"Initializing model, max image pixels: {max_image_pixels}")
    
    if args.engine == 'api':
        print('Using OpenAI API engine')
        from models.openai_api import OpenAIModel
        llm = OpenAIModel(
            model_name=args.model_path,  # e.g. "gpt-5"
            base_url=(args.base_url or None),
            max_workers=args.max_workers,
            reasoning_effort=(args.reasoning_effort or None),
        )
    elif args.engine == 'anthropic':
        print('Using Anthropic (Claude) engine')
        from models.anthropic_api import AnthropicModel
        llm = AnthropicModel(
            model_name=args.model_path,  # e.g. "claude-opus-4-1-20250805"
            base_url=(args.base_url or None),
            max_workers=args.max_workers,
            thinking_budget=args.thinking_budget,
            computer_use=bool(args.computer_use),
            allow_refusal=bool(args.allow_refusal),
        )
    elif args.model_type == 'molmopoint':
        # allenai/MolmoPoint-GUI-8B (MolmoPointForConditionalGeneration). vLLM is ruled
        # out for two independent reasons: the architecture is not in the vLLM registry
        # (only Molmo2ForConditionalGeneration is), and it encodes points as special
        # tokens rather than text coordinates, which must be decoded with preprocessor
        # metadata + extract_image_points. Full rationale in models/molmopoint_hf.py.
        if args.engine != 'hf':
            raise SystemExit(
                f"--model-type molmopoint requires --engine hf (got '{args.engine}'): "
                "vLLM has no MolmoPoint support. See models/molmopoint_hf.py."
            )
        print('Using HuggingFace engine (MolmoPoint-GUI)')
        from models.molmopoint_hf import MolmoPoint
        llm = MolmoPoint(model_path=args.model_path, max_new_tokens=args.max_tokens)
    elif args.model_type in ['qwen2.5vl', 'qwen2vl']:
        if args.engine == 'vllm':
            print('Using VLLM engine')
            from models.qwen2vl import Qwen2VL as Qwen2VL_VLLM
            print(f"Using {args.model_path} VLLM model")
            # Fraction of *each* GPU vLLM may claim on startup. Default 0.9, but
            # on a shared node where other jobs already hold memory this can
            # exceed what's free and abort with "Free memory ... less than
            # desired GPU memory utilization". Override via GPU_MEMORY_UTILIZATION
            # (e.g. 0.5) to fit alongside other processes.
            gpu_mem_util = float(os.environ.get("GPU_MEMORY_UTILIZATION", 0.9))
            print(f"vLLM gpu_memory_utilization: {gpu_mem_util}")
            llm = Qwen2VL_VLLM(
                model_path=args.model_path,
                max_model_len=max_image_pixels//28//28 + 1024,
                tensor_parallel_size=args.tensor_parallel,
                max_num_seqs=args.max_num_seqs,
                min_pixels=4*28*28,
                max_pixels=max_image_pixels,  # keep the vLLM processor resize == the eval pre-resize, so predicted coords stay in the bbox space
                gpu_memory_utilization=gpu_mem_util,
                enforce_eager=True,
            )
        elif args.engine == 'hf':
            print('Using HuggingFace engine')
            from models.qwen2vl_hf import Qwen25VL as Qwen25VL_HF
            if args.model_type.startswith('qwen2.5vl'):
                print("Using Qwen2.5VL HF model")
                llm = Qwen25VL_HF(
                    model_path=args.model_path,
                    min_pixels=4*28*28,
                    max_pixels=max_image_pixels,
                    max_new_tokens=args.max_tokens
                )
            elif args.model_type.startswith('qwen2vl'):
                print("Using Qwen2VL HF model")
                pass

    # Load dataset
    print("Loading dataset...")
    raw_dataset = load_dataset(benchmark_info)
    print(f"Raw dataset size: {len(raw_dataset)}")

    
    # Standardize dataset
    dataset = []
    for sample in raw_dataset:
        standardized = standardize_sample(sample, args.benchmark)
        dataset.append(standardized)
    
    # Random sampling in debug mode
    random.seed(42)
    if debug_mode:
        dataset = random.sample(dataset, len(dataset)//50 if len(dataset) >= 10 else 1)
        print(f"Debug mode enabled, using {len(dataset)} samples.")

    # ---- optional data-parallel sharding -----------------------------------
    # EVAL_NUM_SHARDS / EVAL_SHARD_ID cut the dataset into contiguous slices so N
    # single-GPU jobs can cover ONE benchmark concurrently; merge_shards.py then
    # stitches the per-shard files back into a normal result file. Two details
    # make the shards faithful:
    #   * detailed_results stays keyed by the ORIGINAL global index, so a merged
    #     file is indistinguishable from an unsharded one downstream;
    #   * the slice length is rounded UP to a multiple of --batch-size, so every
    #     generate() call holds exactly the samples it would have held unsharded.
    #     vLLM output depends on batch composition, so without that rounding the
    #     shards would silently re-batch and a few rows would flip.
    # Unset (the default) -> no sharding, the whole benchmark runs in one process.
    num_shards = int(os.environ.get("EVAL_NUM_SHARDS", "0") or 0)
    shard_id, shard_offset = -1, 0
    if num_shards > 1:
        shard_id = int(os.environ["EVAL_SHARD_ID"])
        assert 0 <= shard_id < num_shards, f"EVAL_SHARD_ID={shard_id} out of range for {num_shards}"
        n_full = len(dataset)
        per = (n_full + num_shards - 1) // num_shards
        per = (per + args.batch_size - 1) // args.batch_size * args.batch_size
        shard_offset = shard_id * per
        dataset = dataset[shard_offset:shard_offset + per]
        print(f"Shard {shard_id}/{num_shards}: samples [{shard_offset}, "
              f"{shard_offset + len(dataset)}) of {n_full} (slice={per})")
        if not dataset:
            print("Shard is empty (more shards than batches); nothing to do.")
    
    # Prepare all data
    print("Preparing data...")
    messages_list = []
    images = []
    processed_bboxes = []


    if args.cache_dir:
        cache_base_dir = os.path.expanduser(args.cache_dir)
    else:
        cache_base_dir = "cache"
    
    cache_base_dir = os.path.abspath(cache_base_dir)
    processed_cache_dir = os.path.join(cache_base_dir, f"b-{benchmark_info['name']}_p-{args.prompt}_mit-{args.max_image_tokens}.pkl")
    if not args.no_cache:
        os.makedirs(cache_base_dir, exist_ok=True)

    if os.path.exists(processed_cache_dir) and not debug_mode and not args.no_cache:
        print(f"Loading processed data cache from {processed_cache_dir}")
        with open(processed_cache_dir, "rb") as f:
            import pickle
            cache_data = pickle.load(f)
            messages_list = cache_data["messages_list"]
            images = cache_data["images"]
            processed_bboxes = cache_data["processed_bboxes"]
            dataset = cache_data["dataset"]
            print(f"Loaded {len(messages_list)} samples from cache")
    else:
        for i, sample in enumerate(tqdm(dataset, desc="Preparing samples")):
            image, messages, bbox = prepare_sample_data(
                sample, benchmark_info, max_image_pixels, prompt_processor, think_mode
            )
            messages_list.append(messages)
            images.append(image)
            processed_bboxes.append(bbox)
            
            # Update bbox information in the sample (for subsequent processing)
            dataset[i]["processed_bbox"] = bbox
            dataset[i]["processed_imgsize"] = image.size

        if not debug_mode and not args.no_cache:
            cache_data = {
                "messages_list": messages_list,
                "images": images,
                "processed_bboxes": processed_bboxes,
                "dataset": dataset}
            
            with open(processed_cache_dir, "wb") as f:
                import pickle
                pickle.dump(cache_data, f)
                print(f"Saved processed data cache to {processed_cache_dir}")
    
    # Batch generate responses
    print("Generating responses...")
    
    responses = llm.get_responses(args, messages_list=messages_list, images=images)
    
    # Process response results
    print("Processing response results...")
    results = {}
    
    for idx, (sample, response) in enumerate(tqdm(zip(dataset, responses), total=len(dataset), desc="Processing responses")):
        bbox = sample["processed_bbox"]
        # print(response)
        try:
            result = process_response(sample, response, bbox, prompt_processor, think_mode)
            results[idx] = result
        except Exception as e:
            print(f"Failed to process sample {idx}: {e}")
            # Add default result using prompt processor's metric definition
            default_metrics = prompt_processor.calculate_metrics(sample, None, bbox)
            results[idx] = {
                "image": sample.get("images", ""),
                "instruction": sample.get("instruction", ""),
                "gt_bbox": bbox,
                "response": response,
                **default_metrics
            }
    
    # Calculate hierarchical statistics
    print("Calculating statistics...")
    statistics = calculate_hierarchical_statistics(results, dataset, prompt_processor)
    
    # Print statistics
    print("\n=== Evaluation Results ===")
    metric_keys = prompt_processor.get_metric_keys()
    accuracy_pairs = prompt_processor.get_accuracy_pairs()
    print_hierarchical_stats(statistics, 0, metric_keys, accuracy_pairs)
    
    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Reorder statistics, moving subgroups to the end
    ordered_statistics = reorder_stats_for_output(statistics)
    
    # results are keyed by LOCAL index (calculate_hierarchical_statistics enumerates
    # the shard); re-key to the GLOBAL index on the way out so shards merge cleanly.
    out_results = ({str(shard_offset + k): v for k, v in results.items()}
                   if num_shards > 1 else results)

    output_data = {
        "benchmark": args.benchmark,
        "prompt": args.prompt,
        "model_path": args.model_path,
        "args": vars(args),
        "statistics": ordered_statistics,
        "detailed_results": out_results
    }
    if num_shards > 1:
        output_data["shard"] = {"shard_id": shard_id, "num_shards": num_shards,
                                "offset": shard_offset, "count": len(dataset)}
    
    output_file = os.path.join(
        output_dir, 
        f"{timestamp}{'_t'+str(args.temperature).replace('.', '-') if args.temperature else ''}"
        f"{'_debug' if debug_mode else ''}"
        f"{f'_shard{shard_id}of{num_shards}' if num_shards > 1 else ''}_{args.prompt}.json"
    )
    
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)
    
    print(f"\nEvaluation results saved to: {output_file}")
    print_dict = {}
    for k, v in ordered_statistics.items():
        if k.endswith('_accuracy'):
            print_dict[k] = v
    print(f"Overall accuracy: {print_dict}")


if __name__ == "__main__":
    main()
