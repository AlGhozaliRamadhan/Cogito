"""
Inference Engine: Model Lifecycle, Quantization, and Non-blocking Execution
"""

import os
import time
import queue
import logging
import threading
import asyncio
from pathlib import Path
from typing import Optional, Dict, Any, List, AsyncGenerator, Tuple

logger = logging.getLogger("cogito-engine")

try:
    import torch
    from transformers import (
        AutoModelForCausalLM,
        AutoTokenizer,
        TextIteratorStreamer,
        BitsAndBytesConfig,
        StoppingCriteriaList,
    )
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    logger.warning("transformers or torch not available in environment.")

from src.core.stop_criteria import WindowedStringStopCriteria
from src.core.prompt import DEFAULT_STOP_TOKENS

class InferenceEngine:
    """
    Manages model initialization, hardware quantization, and non-blocking generation.
    """

    def __init__(self, model_path: str, quant_mode: str = "auto", trust_remote_code: bool = True):
        self.model_path = Path(model_path)
        self.quant_mode = quant_mode
        self.trust_remote_code = trust_remote_code
        self.model: Optional[Any] = None
        self.tokenizer: Optional[Any] = None
        self.model_loaded: bool = False
        self.model_loading: bool = False
        self.lock = threading.Lock()
        self.ready_event = threading.Event()

    def load(self):
        """Loads tokenizer and safetensors model into GPU/CPU memory."""
        if self.model_loaded or self.model_loading:
            return
        self.model_loading = True
        logger.info(f"Loading model from: {self.model_path}")

        try:
            if not TRANSFORMERS_AVAILABLE:
                raise RuntimeError("Hugging Face transformers or PyTorch is not available.")

            self.tokenizer = AutoTokenizer.from_pretrained(
                str(self.model_path),
                trust_remote_code=self.trust_remote_code,
                local_files_only=self.model_path.exists() and (self.model_path / "tokenizer.json").exists(),
            )
            if self.tokenizer.pad_token_id is None:
                self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

            load_kwargs: Dict[str, Any] = {
                "trust_remote_code": self.trust_remote_code,
                "low_cpu_mem_usage": True,
            }

            has_cuda = torch.cuda.is_available()
            if has_cuda:
                load_kwargs["device_map"] = "auto"
                gpu_count = torch.cuda.device_count()
                total_vram_gb = sum(torch.cuda.get_device_properties(i).total_memory for i in range(gpu_count)) / (1024**3)
                compute_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
                logger.info(f"CUDA: {gpu_count} GPU(s), {total_vram_gb:.1f} GB VRAM. Target dtype: {compute_dtype}")

                if self.quant_mode in ("4bit", "4-bit", "q4", "bnb4"):
                    logger.info("Explicit 4-bit NF4 quantization enabled.")
                    load_kwargs["quantization_config"] = BitsAndBytesConfig(
                        load_in_4bit=True,
                        bnb_4bit_compute_dtype=compute_dtype,
                        bnb_4bit_quant_type="nf4",
                        bnb_4bit_use_double_quant=True,
                    )
                elif self.quant_mode in ("8bit", "8-bit", "q8", "bnb8"):
                    logger.info("Explicit 8-bit quantization enabled.")
                    load_kwargs["quantization_config"] = BitsAndBytesConfig(load_in_8bit=True)
                elif self.quant_mode in ("16bit", "fp16", "bf16", "none"):
                    logger.info(f"Full precision ({compute_dtype}) enabled.")
                    load_kwargs["torch_dtype"] = compute_dtype
                else:
                    if total_vram_gb < 28:
                        logger.info(f"VRAM ({total_vram_gb:.1f} GB) < 28 GB -> Auto 4-bit NF4 quantization")
                        try:
                            load_kwargs["quantization_config"] = BitsAndBytesConfig(
                                load_in_4bit=True,
                                bnb_4bit_compute_dtype=compute_dtype,
                                bnb_4bit_quant_type="nf4",
                                bnb_4bit_use_double_quant=True,
                            )
                        except Exception as e:
                            logger.warning(f"BitsAndBytes config failed ({e}), falling back to {compute_dtype}")
                            load_kwargs["torch_dtype"] = compute_dtype
                    else:
                        logger.info(f"VRAM ({total_vram_gb:.1f} GB) >= 28 GB -> Loading in {compute_dtype}")
                        load_kwargs["torch_dtype"] = compute_dtype
            else:
                logger.info("No CUDA detected. Loading on CPU in float32.")
                load_kwargs["device_map"] = "cpu"
                load_kwargs["torch_dtype"] = torch.float32

            self.model = AutoModelForCausalLM.from_pretrained(str(self.model_path), **load_kwargs)
            self.model.eval()
            self.model_loaded = True
            self.ready_event.set()
            logger.info("Model loaded and ready for inference!")
        except Exception as e:
            logger.error(f"Failed to load model: {e}", exc_info=True)
        finally:
            self.model_loading = False

    def is_ready(self) -> bool:
        return self.model_loaded and self.model is not None and self.tokenizer is not None

    def prepare_generation_args(
        self,
        prompt: str,
        max_tokens: int = 512,
        temperature: float = 0.7,
        top_p: float = 0.95,
        top_k: int = 40,
        repeat_penalty: float = 1.1,
        custom_stops: Optional[List[str]] = None,
    ) -> Tuple[Dict[str, Any], int, List[str]]:
        stop_list = (custom_stops or []) + DEFAULT_STOP_TOKENS
        inputs = self.tokenizer(prompt, return_tensors="pt")
        
        if TRANSFORMERS_AVAILABLE and torch.cuda.is_available() and hasattr(self.model, "device"):
            inputs = {k: v.to(self.model.device) for k, v in inputs.items()}

        input_len = inputs["input_ids"].shape[1]
        stopping_criteria = StoppingCriteriaList([
            WindowedStringStopCriteria(self.tokenizer, stop_list, input_len)
        ])

        gen_kwargs = {
            **inputs,
            "max_new_tokens": max_tokens,
            "temperature": max(temperature, 1e-4) if temperature > 0 else 1e-4,
            "top_p": top_p if (top_p is not None and temperature > 0) else 1.0,
            "do_sample": bool(temperature > 0),
            "pad_token_id": self.tokenizer.pad_token_id or self.tokenizer.eos_token_id,
            "eos_token_id": self.tokenizer.eos_token_id,
            "stopping_criteria": stopping_criteria,
        }
        if top_k and top_k > 0:
            gen_kwargs["top_k"] = top_k
        if repeat_penalty and repeat_penalty != 1.0:
            gen_kwargs["repetition_penalty"] = repeat_penalty

        return gen_kwargs, input_len, stop_list

    async def generate_non_streaming(self, gen_kwargs: Dict[str, Any], input_len: int, stop_list: List[str]) -> Tuple[str, int, int]:
        def _run():
            with torch.no_grad():
                with self.lock:
                    return self.model.generate(**gen_kwargs)

        out_ids = await asyncio.to_thread(_run)
        gen_tokens = out_ids[0][input_len:]
        raw_text = self.tokenizer.decode(gen_tokens, skip_special_tokens=False)

        clean_text = raw_text
        for sw in stop_list:
            if sw in clean_text:
                clean_text = clean_text.split(sw)[0]

        completion_tokens = len(gen_tokens)
        return clean_text, input_len, completion_tokens
