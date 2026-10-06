import os
from typing import List, Union, Dict, Any, Optional
from openai import responses
from tqdm import tqdm
from transformers import AutoConfig
from PIL import Image
from vllm import LLM, SamplingParams

class Qwen2VL:
    def __init__(
        self,
        model_path: str = "",
        tensor_parallel_size: int = 1,
        gpu_memory_utilization: float = 0.9,
        enforce_eager: bool = False,
        max_model_len: int = 8192,
        limit_mm_per_prompt: Dict[str, int] = {"image": 1, "video": 0},
        min_pixels: Optional[int] = None,
        max_pixels: Optional[int] = None,
        max_num_seqs: Optional[int] = None,
    ):
        """
        Initialize Qwen2VL/Qwen2.5VL inference class
        
        Args:
            model_path: Path to the model
            tensor_parallel_size: Number of tensor parallel processes
            gpu_memory_utilization: GPU memory utilization ratio
            enforce_eager: Whether to enforce eager execution
            max_model_len: Maximum sequence length
            limit_mm_per_prompt: Limit multimodal inputs per prompt
            min_pixels: Minimum number of pixels for image processing
            max_pixels: Maximum number of pixels for image processing
            max_num_seqs: Maximum number of sequences
        """
        kwargs = {
            "model": model_path,
            "tensor_parallel_size": tensor_parallel_size,
            "gpu_memory_utilization": gpu_memory_utilization,
            "enforce_eager": enforce_eager,
            "max_model_len": max_model_len,
            "limit_mm_per_prompt": limit_mm_per_prompt,
        }
        
        if max_num_seqs:
            kwargs["max_num_seqs"] = max_num_seqs
            
        if min_pixels or max_pixels:
            kwargs["mm_processor_kwargs"] = {
                "min_pixels": min_pixels if min_pixels else 4 * 28 * 28,
                "max_pixels": max_pixels if max_pixels else 16384 * 28 * 28,
            }
            
        
        # IMPORTANT: do NOT delete `text_config` and save_pretrained back to disk.
        # On transformers>=5 the flat Qwen2.5-VL text params are migrated into
        # `text_config`; stripping it and saving permanently corrupts config.json
        # (wipes hidden_size/intermediate_size/..), which breaks vLLM weight loading
        # with shape-mismatch errors. vLLM reads the config from disk itself, so no
        # in-place mutation is needed here.

        # Blackwell (sm_103): force the prebuilt flash-attn backend. By default vLLM
        # picks FlashInfer and JIT-compiles its kernels with nvcc for
        # arch=compute_103a, which a CUDA 12.x toolkit does not support
        # (nvcc fatal: Unsupported gpu architecture 'compute_103a'). Guarded so it is
        # a no-op on vLLM builds that don't expose the `attention_backend` arg.
        try:
            from vllm.engine.arg_utils import EngineArgs
            if "attention_backend" in getattr(EngineArgs, "__dataclass_fields__", {}):
                kwargs.setdefault(
                    "attention_backend", os.environ.get("VLLM_ATTN_BACKEND", "FLASH_ATTN")
                )
        except Exception:
            pass

        # Optional dtype override (e.g. VLLM_DTYPE=bfloat16). Needed for checkpoints
        # whose config.json torch_dtype is float32 (a verl/actor_hf export artifact):
        # vLLM "auto" downcasts such a config to float16, which is unsafe for
        # bf16-native Qwen2.5-VL (fp16 overflow -> NaN). Empty/unset => unchanged.
        _dtype = os.environ.get("VLLM_DTYPE", "").strip()
        if _dtype:
            kwargs["dtype"] = _dtype

        self.llm = LLM(**kwargs)
        self.system_prompt = "You are a helpful assistant."
        
        
    def _format_prompt(self, messages: List[Dict[str, Any]], images: List[Image.Image]) -> str:
        """
        Format conversation history and handle custom image positions
        
        Args:
            messages: Conversation history
            images: List of images
            
        Returns:
            Formatted prompt string
        """
        prompt = ""
        if messages[0]["role"] != "system":
            prompt = f"<|im_start|>system\n{self.system_prompt}<|im_end|>\n"
        
        image_idx = 0
        
        for message in messages:
            role = message["role"]
            content = message["content"]
            if role == "system":
                # content=None means "emit NO system block at all" (not an empty
                # one). Holo2's localization cookbook sends a single user turn with
                # no system message, and Holo2's own chat template injects no
                # default -- so prepending "You are a helpful assistant." here would
                # put the model off-distribution. A processor asks for that by
                # returning {"role": "system", "content": None}, which still
                # suppresses the default injection above.
                if content is None:
                    continue
                prompt += f"<|im_start|>system\n{content}<|im_end|>\n"
            elif role in ["user", "human"]:
                # Count required images
                required_images = content.count("<image>")
                if image_idx + required_images > len(images):
                    raise ValueError(f"Not enough images provided. Required: {image_idx + required_images}, Provided: {len(images)}")
                
                # Replace all <image> tokens
                for _ in range(required_images):
                    content = content.replace("<image>", "<|vision_start|><|image_pad|><|vision_end|>", 1)
                    image_idx += 1
                    
                prompt += f"<|im_start|>user\n{content}<|im_end|>\n"
            elif role in ["assistant", "gpt"]:
                prompt += f"<|im_start|>assistant\n{content}<|im_end|>\n"
                
        # Assistant-side prefill. Qwen3-VL *Thinking* derivatives (Holo2) always
        # open a <think> block in their chat template; apply_chat_template(...,
        # thinking=False) closes it immediately with "<think>\n\n</think>\n\n".
        # This function builds ChatML by hand and never calls the template, so
        # without this hook a Thinking model gets a bare "assistant\n" prefix,
        # starts reasoning on its own and burns max_tokens before emitting the
        # answer. Set ASSISTANT_PREFIX to the exact prefill the template would emit.
        prompt += "<|im_start|>assistant\n" + os.environ.get("ASSISTANT_PREFIX", "")
        
        # Verify all images are used
        if image_idx < len(images):
            raise ValueError(f"Too many images provided. Used: {image_idx}, Provided: {len(images)}")
            
        return prompt

    def chat(
        self,
        messages: Union[List[Dict[str, Any]], List[List[Dict[str, Any]]]],
        images: Optional[Union[Image.Image, List[Image.Image], List[List[Image.Image]]]] = None,
        temperature: float = 0.0,
        max_tokens: int = 2048,
        top_p: float = 1.0,
        **kwargs
    ) -> Union[str, List[str]]:
        """
        Chat interface supporting single or batch conversations
        
        Args:
            messages: Single conversation history or list of conversation histories
            images: Single image, list of images, or batch list of images
            temperature: Sampling temperature
            max_tokens: Maximum generation length
            top_p: Top-p sampling parameter
            **kwargs: Additional sampling parameters
            
        Returns:
            Generated response or list of responses
        """
        # Check if this is a batch request
        is_batch = isinstance(messages[0], list)
        if not is_batch:
            messages = [messages]
            images = [images] if isinstance(images, Image.Image) else ([images] if images is not None else [None])
            
        # Prepare inputs
        inputs = []
        assert len(messages) == len(images)
        for msg, img_list in zip(messages, images):
            if img_list is not None:
                img_list = [img_list] if isinstance(img_list, Image.Image) else img_list
                prompt = self._format_prompt(msg, img_list)
                
                # Build multimodal data dictionary
                input_data = {
                    "prompt": prompt,
                    "multi_modal_data": {
                        "image": img_list
                    }
                }
            else:
                prompt = self._format_prompt(msg, [])
                input_data = {
                    "prompt": prompt,
                    "multi_modal_data": {}
                }
                
            inputs.append(input_data)
            
        # Set sampling parameters
        sampling_params = SamplingParams(
            temperature=temperature,
            max_tokens=max_tokens,
            top_p=top_p,
            **kwargs
        )
        
        # Execute inference
        outputs = self.llm.generate(inputs, sampling_params=sampling_params)
        responses = [output.outputs[0].text for output in outputs]
        
        return responses[0] if not is_batch else responses

    def get_responses(self, args, messages_list, images):
        responses = []
        total_samples = len(messages_list)
        progress_bar = tqdm(total=total_samples, desc="Generating responses")
        
        for i in range(0, len(messages_list), args.batch_size):
            batch_messages = messages_list[i:i+args.batch_size] 
            batch_images = images[i:i+args.batch_size]
            
            # Generate responses
            batch_responses = self.chat(
                batch_messages, 
                batch_images, 
                max_tokens=args.max_tokens, 
                temperature=args.temperature
            )
            responses.extend(batch_responses)
            # print([responses[-1]])
            
            progress_bar.update(len(batch_messages))
        
        progress_bar.close()
    
        return responses
