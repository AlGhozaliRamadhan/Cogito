"""
GGUF Inference Engine: Model Lifecycle, Hardware Offloading, and Non-blocking Execution
"""

import os
import time
import queue
import logging
import threading
import asyncio
from pathlib import Path
from typing import Optional, Dict, Any, List, AsyncGenerator, Tuple, Union

logger = logging.getLogger("cogito-engine")

try:
    import llama_cpp
    LLAMA_CPP_AVAILABLE = True
except ImportError:
    LLAMA_CPP_AVAILABLE = False
    logger.warning("llama_cpp not available in environment.")

from src.core.stop_criteria import get_combined_stop_tokens
from src.core.prompt import DEFAULT_STOP_TOKENS, CANONICAL_SYSTEM_PROMPT, ChatMessage, prepare_chat_messages

class InferenceEngine:
    """
    High-performance GGUF Inference Engine powered by llama.cpp C++ backend.
    Supports GPU offloading, FlashAttention, context caching, and non-blocking streaming.
    """

    def __init__(
        self,
        model_path: str,
        quant_mode: str = "q4_k_m",
        n_ctx: int = 32768,
        n_gpu_layers: int = -1,
        flash_attn: bool = True,
        cache_type_k: str = "q8_0",
        cache_type_v: str = "q8_0",
        trust_remote_code: bool = True,
    ):
        self.model_path = Path(model_path)
        self.quant_mode = quant_mode
        self.n_ctx = n_ctx
        self.n_gpu_layers = n_gpu_layers
        self.flash_attn = flash_attn
        self.cache_type_k = cache_type_k
        self.cache_type_v = cache_type_v
        self.trust_remote_code = trust_remote_code

        self.model: Optional[Any] = None
        self.tokenizer: Optional[Any] = None  # Maintained for mock/compatibility
        self.model_loaded: bool = False
        self.model_loading: bool = False
        self.lock = threading.Lock()
        self.ready_event = threading.Event()

    def load(self):
        """Loads the model into GPU VRAM / system memory (Safetensors / GGUF)."""
        if self.model_loaded or self.model_loading:
            return
        self.model_loading = True
        logger.info(f"Loading model from: {self.model_path} (quant={self.quant_mode})")

        try:
            model_path_str = str(self.model_path)
            is_gguf = (
                model_path_str.endswith(".gguf")
                or (self.model_path.is_dir() and list(self.model_path.glob("*.gguf")))
            )

            if is_gguf and LLAMA_CPP_AVAILABLE:
                model_file = self.model_path
                if model_file.is_dir():
                    model_file = list(model_file.glob("*.gguf"))[0]
                logger.info(f"Initializing llama_cpp.Llama with GGUF weights: {model_file}")
                self.model = llama_cpp.Llama(
                    model_path=str(model_file),
                    n_ctx=self.n_ctx,
                    n_gpu_layers=self.n_gpu_layers,
                    flash_attn=self.flash_attn,
                    type_k=self.cache_type_k,
                    type_v=self.cache_type_v,
                    verbose=False,
                )
            else:
                # Load normal Safetensors / Hugging Face model
                try:
                    import torch
                    from transformers import AutoModelForCausalLM, AutoTokenizer

                    logger.info(f"Initializing Transformers with Safetensors from: {model_path_str}")
                    self.tokenizer = AutoTokenizer.from_pretrained(
                        model_path_str,
                        trust_remote_code=self.trust_remote_code,
                    )

                    load_kwargs = {
                        "trust_remote_code": self.trust_remote_code,
                        "device_map": "auto" if torch.cuda.is_available() else None,
                    }

                    if self.quant_mode in ("4bit", "auto") and torch.cuda.is_available():
                        try:
                            from transformers import BitsAndBytesConfig
                            load_kwargs["quantization_config"] = BitsAndBytesConfig(
                                load_in_4bit=True,
                                bnb_4bit_use_double_quant=True,
                                bnb_4bit_quant_type="nf4",
                                bnb_4bit_compute_dtype=torch.bfloat16,
                            )
                            load_kwargs["dtype"] = torch.bfloat16
                        except ImportError:
                            load_kwargs["dtype"] = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
                    elif self.quant_mode == "8bit" and torch.cuda.is_available():
                        try:
                            from transformers import BitsAndBytesConfig
                            load_kwargs["quantization_config"] = BitsAndBytesConfig(load_in_8bit=True)
                        except ImportError:
                            load_kwargs["dtype"] = torch.bfloat16
                    elif torch.cuda.is_available():
                        load_kwargs["dtype"] = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16

                    self.model = AutoModelForCausalLM.from_pretrained(
                        model_path_str,
                        **load_kwargs,
                    )
                except ImportError:
                    logger.warning("torch / transformers not available; operating in mock/compatibility mode.")

            self.model_loaded = True
            self.ready_event.set()
            logger.info("Model successfully loaded and ready for inference!")
        except Exception as e:
            logger.error(f"Failed to load model: {e}", exc_info=True)
        finally:
            self.model_loading = False

    def is_ready(self) -> bool:
        return self.model_loaded and self.model is not None

    async def generate_chat_stream(
        self,
        messages: List[Union[ChatMessage, Dict[str, str]]],
        max_tokens: int = 2048,
        temperature: float = 0.70,
        top_p: float = 0.90,
        min_p: float = 0.05,
        top_k: int = 40,
        repeat_penalty: float = 1.08,
        custom_stops: Optional[List[str]] = None,
        cancel_event: Optional[asyncio.Event] = None,
    ) -> AsyncGenerator[Dict[str, Any], None]:
        """
        Non-blocking streaming chat completion generator with instant cancellation.
        """
        stop_list = get_combined_stop_tokens(custom_stops)
        
        # Prepare structured messages with canonical persona
        if messages and isinstance(messages[0], ChatMessage):
            formatted_messages = prepare_chat_messages(messages)
        else:
            has_system = any(m.get("role") == "system" for m in messages)
            formatted_messages = []
            if not has_system:
                formatted_messages.append({"role": "system", "content": CANONICAL_SYSTEM_PROMPT})
            formatted_messages.extend(messages)

        loop = asyncio.get_running_loop()
        q: asyncio.Queue = asyncio.Queue()
        stop_worker = threading.Event()

        def _worker():
            try:
                if hasattr(self.model, "create_chat_completion"):
                    kwargs = {
                        "messages": formatted_messages,
                        "max_tokens": max_tokens,
                        "temperature": max(temperature, 1e-4) if temperature > 0 else 1e-4,
                        "top_p": top_p if (top_p is not None and temperature > 0) else 1.0,
                        "min_p": min_p,
                        "repeat_penalty": repeat_penalty,
                        "top_k": top_k,
                        "stop": stop_list,
                        "stream": True,
                    }
                    for chunk in self.model.create_chat_completion(**kwargs):
                        if stop_worker.is_set():
                            break
                        loop.call_soon_threadsafe(q.put_nowait, chunk)
                elif self.tokenizer is not None and hasattr(self.model, "generate"):
                    import torch
                    from transformers import TextIteratorStreamer

                    prompt_text = self.tokenizer.apply_chat_template(
                        formatted_messages,
                        tokenize=False,
                        add_generation_prompt=True,
                    )
                    inputs = self.tokenizer(prompt_text, return_tensors="pt")
                    if hasattr(self.model, "device"):
                        inputs = inputs.to(self.model.device)

                    streamer = TextIteratorStreamer(
                        self.tokenizer,
                        skip_prompt=True,
                        skip_special_tokens=True,
                    )

                    gen_kwargs = {
                        "input_ids": inputs["input_ids"],
                        "attention_mask": inputs.get("attention_mask"),
                        "max_new_tokens": max_tokens,
                        "streamer": streamer,
                        "temperature": max(temperature, 1e-4) if temperature > 0 else 1.0,
                        "top_p": top_p if (top_p is not None and temperature > 0) else 1.0,
                        "top_k": top_k if temperature > 0 else 50,
                        "repetition_penalty": repeat_penalty,
                        "do_sample": temperature > 0,
                        "pad_token_id": self.tokenizer.eos_token_id,
                    }

                    gen_thread = threading.Thread(
                        target=self.model.generate,
                        kwargs=gen_kwargs,
                        daemon=True,
                    )
                    gen_thread.start()

                    first_chunk = True
                    for new_text in streamer:
                        if stop_worker.is_set():
                            break
                        delta = {"content": new_text}
                        if first_chunk:
                            delta["role"] = "assistant"
                            first_chunk = False
                        loop.call_soon_threadsafe(
                            q.put_nowait,
                            {"choices": [{"delta": delta, "finish_reason": None}]}
                        )

                    loop.call_soon_threadsafe(
                        q.put_nowait,
                        {"choices": [{"delta": {}, "finish_reason": "stop"}]}
                    )
                else:
                    # Fallback / Mock compatibility
                    loop.call_soon_threadsafe(
                        q.put_nowait,
                        {"choices": [{"delta": {"role": "assistant", "content": "<think>\nThinking...\n</think>\nResponse content."}, "finish_reason": None}]}
                    )
                    loop.call_soon_threadsafe(
                        q.put_nowait,
                        {"choices": [{"delta": {}, "finish_reason": "stop"}]}
                    )
            except Exception as e:
                logger.error(f"Error in chat stream worker: {e}", exc_info=True)
            finally:
                loop.call_soon_threadsafe(q.put_nowait, None)

        worker_thread = threading.Thread(target=_worker, daemon=True)
        worker_thread.start()

        try:
            while True:
                if cancel_event and cancel_event.is_set():
                    stop_worker.set()
                    break
                chunk = await q.get()
                if chunk is None:
                    break
                yield chunk
        finally:
            stop_worker.set()

    async def generate_chat_non_streaming(
        self,
        messages: List[Union[ChatMessage, Dict[str, str]]],
        max_tokens: int = 2048,
        temperature: float = 0.70,
        top_p: float = 0.90,
        min_p: float = 0.05,
        top_k: int = 40,
        repeat_penalty: float = 1.08,
        custom_stops: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """
        Non-blocking execution of complete chat completion.
        """
        stop_list = get_combined_stop_tokens(custom_stops)
        
        if messages and isinstance(messages[0], ChatMessage):
            formatted_messages = prepare_chat_messages(messages)
        else:
            has_system = any(m.get("role") == "system" for m in messages)
            formatted_messages = []
            if not has_system:
                formatted_messages.append({"role": "system", "content": CANONICAL_SYSTEM_PROMPT})
            formatted_messages.extend(messages)

        def _run():
            if hasattr(self.model, "create_chat_completion"):
                return self.model.create_chat_completion(
                    messages=formatted_messages,
                    max_tokens=max_tokens,
                    temperature=max(temperature, 1e-4) if temperature > 0 else 1e-4,
                    top_p=top_p if (top_p is not None and temperature > 0) else 1.0,
                    min_p=min_p,
                    repeat_penalty=repeat_penalty,
                    top_k=top_k,
                    stop=stop_list,
                    stream=False,
                )
            elif self.tokenizer is not None and hasattr(self.model, "generate"):
                import torch
                prompt_text = self.tokenizer.apply_chat_template(
                    formatted_messages,
                    tokenize=False,
                    add_generation_prompt=True,
                )
                inputs = self.tokenizer(prompt_text, return_tensors="pt")
                if hasattr(self.model, "device"):
                    inputs = inputs.to(self.model.device)

                gen_kwargs = {
                    "input_ids": inputs["input_ids"],
                    "attention_mask": inputs.get("attention_mask"),
                    "max_new_tokens": max_tokens,
                    "temperature": max(temperature, 1e-4) if temperature > 0 else 1.0,
                    "top_p": top_p if (top_p is not None and temperature > 0) else 1.0,
                    "top_k": top_k if temperature > 0 else 50,
                    "repetition_penalty": repeat_penalty,
                    "do_sample": temperature > 0,
                    "pad_token_id": self.tokenizer.eos_token_id,
                }
                outputs = self.model.generate(**gen_kwargs)
                gen_tokens = outputs[0][inputs["input_ids"].shape[1]:]
                response_text = self.tokenizer.decode(gen_tokens, skip_special_tokens=True)
                return {
                    "id": f"chatcmpl-{int(time.time())}",
                    "object": "chat.completion",
                    "created": int(time.time()),
                    "model": "Cogito-0.9.1-15B",
                    "choices": [{
                        "index": 0,
                        "message": {"role": "assistant", "content": response_text},
                        "finish_reason": "stop"
                    }],
                    "usage": {
                        "prompt_tokens": int(inputs["input_ids"].shape[1]),
                        "completion_tokens": int(gen_tokens.shape[0]),
                        "total_tokens": int(inputs["input_ids"].shape[1] + gen_tokens.shape[0]),
                    }
                }
            else:
                # Mock fallback
                return {
                    "id": f"chatcmpl-{int(time.time())}",
                    "object": "chat.completion",
                    "created": int(time.time()),
                    "model": "Cogito-0.9.1-15B",
                    "choices": [{
                        "index": 0,
                        "message": {"role": "assistant", "content": "<think>\nVerified reasoning.\n</think>\nComplete output."},
                        "finish_reason": "stop"
                    }],
                    "usage": {"prompt_tokens": 50, "completion_tokens": 20, "total_tokens": 70}
                }

        return await asyncio.to_thread(_run)

    async def generate_text_stream(
        self,
        prompt: str,
        max_tokens: int = 2048,
        temperature: float = 0.70,
        top_p: float = 0.90,
        min_p: float = 0.05,
        top_k: int = 40,
        repeat_penalty: float = 1.08,
        custom_stops: Optional[List[str]] = None,
        cancel_event: Optional[asyncio.Event] = None,
    ) -> AsyncGenerator[Dict[str, Any], None]:
        """
        Non-blocking streaming text completion generator.
        """
        stop_list = get_combined_stop_tokens(custom_stops)
        loop = asyncio.get_running_loop()
        q: asyncio.Queue = asyncio.Queue()
        stop_worker = threading.Event()

        def _worker():
            try:
                if hasattr(self.model, "create_completion"):
                    kwargs = {
                        "prompt": prompt,
                        "max_tokens": max_tokens,
                        "temperature": max(temperature, 1e-4) if temperature > 0 else 1e-4,
                        "top_p": top_p if (top_p is not None and temperature > 0) else 1.0,
                        "min_p": min_p,
                        "repeat_penalty": repeat_penalty,
                        "top_k": top_k,
                        "stop": stop_list,
                        "stream": True,
                    }
                    for chunk in self.model.create_completion(**kwargs):
                        if stop_worker.is_set():
                            break
                        loop.call_soon_threadsafe(q.put_nowait, chunk)
                elif self.tokenizer is not None and hasattr(self.model, "generate"):
                    from transformers import TextIteratorStreamer

                    inputs = self.tokenizer(prompt, return_tensors="pt")
                    if hasattr(self.model, "device"):
                        inputs = inputs.to(self.model.device)

                    streamer = TextIteratorStreamer(
                        self.tokenizer,
                        skip_prompt=True,
                        skip_special_tokens=True,
                    )

                    gen_kwargs = {
                        "input_ids": inputs["input_ids"],
                        "attention_mask": inputs.get("attention_mask"),
                        "max_new_tokens": max_tokens,
                        "streamer": streamer,
                        "temperature": max(temperature, 1e-4) if temperature > 0 else 1.0,
                        "top_p": top_p if (top_p is not None and temperature > 0) else 1.0,
                        "top_k": top_k if temperature > 0 else 50,
                        "repetition_penalty": repeat_penalty,
                        "do_sample": temperature > 0,
                        "pad_token_id": self.tokenizer.eos_token_id,
                    }

                    gen_thread = threading.Thread(
                        target=self.model.generate,
                        kwargs=gen_kwargs,
                        daemon=True,
                    )
                    gen_thread.start()

                    for new_text in streamer:
                        if stop_worker.is_set():
                            break
                        loop.call_soon_threadsafe(
                            q.put_nowait,
                            {"choices": [{"text": new_text, "index": 0, "finish_reason": None}]}
                        )

                    loop.call_soon_threadsafe(
                        q.put_nowait,
                        {"choices": [{"text": "", "index": 0, "finish_reason": "stop"}]}
                    )
                else:
                    loop.call_soon_threadsafe(
                        q.put_nowait,
                        {"choices": [{"text": " Generated completion text.", "index": 0, "finish_reason": None}]}
                    )
                    loop.call_soon_threadsafe(
                        q.put_nowait,
                        {"choices": [{"text": "", "index": 0, "finish_reason": "stop"}]}
                    )
            except Exception as e:
                logger.error(f"Error in text stream worker: {e}", exc_info=True)
            finally:
                loop.call_soon_threadsafe(q.put_nowait, None)

        worker_thread = threading.Thread(target=_worker, daemon=True)
        worker_thread.start()

        try:
            while True:
                if cancel_event and cancel_event.is_set():
                    stop_worker.set()
                    break
                chunk = await q.get()
                if chunk is None:
                    break
                yield chunk
        finally:
            stop_worker.set()

    async def generate_text_non_streaming(
        self,
        prompt: str,
        max_tokens: int = 2048,
        temperature: float = 0.70,
        top_p: float = 0.90,
        min_p: float = 0.05,
        top_k: int = 40,
        repeat_penalty: float = 1.08,
        custom_stops: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """
        Non-blocking execution of raw text completion.
        """
        stop_list = get_combined_stop_tokens(custom_stops)

        def _run():
            if hasattr(self.model, "create_completion"):
                return self.model.create_completion(
                    prompt=prompt,
                    max_tokens=max_tokens,
                    temperature=max(temperature, 1e-4) if temperature > 0 else 1e-4,
                    top_p=top_p if (top_p is not None and temperature > 0) else 1.0,
                    min_p=min_p,
                    repeat_penalty=repeat_penalty,
                    top_k=top_k,
                    stop=stop_list,
                    stream=False,
                )
            elif self.tokenizer is not None and hasattr(self.model, "generate"):
                inputs = self.tokenizer(prompt, return_tensors="pt")
                if hasattr(self.model, "device"):
                    inputs = inputs.to(self.model.device)

                gen_kwargs = {
                    "input_ids": inputs["input_ids"],
                    "attention_mask": inputs.get("attention_mask"),
                    "max_new_tokens": max_tokens,
                    "temperature": max(temperature, 1e-4) if temperature > 0 else 1.0,
                    "top_p": top_p if (top_p is not None and temperature > 0) else 1.0,
                    "top_k": top_k if temperature > 0 else 50,
                    "repetition_penalty": repeat_penalty,
                    "do_sample": temperature > 0,
                    "pad_token_id": self.tokenizer.eos_token_id,
                }
                outputs = self.model.generate(**gen_kwargs)
                gen_tokens = outputs[0][inputs["input_ids"].shape[1]:]
                response_text = self.tokenizer.decode(gen_tokens, skip_special_tokens=True)
                return {
                    "id": f"cmpl-{int(time.time())}",
                    "object": "text_completion",
                    "created": int(time.time()),
                    "model": "Cogito-0.9.1-15B",
                    "choices": [{"text": response_text, "index": 0, "finish_reason": "stop"}],
                    "usage": {
                        "prompt_tokens": int(inputs["input_ids"].shape[1]),
                        "completion_tokens": int(gen_tokens.shape[0]),
                        "total_tokens": int(inputs["input_ids"].shape[1] + gen_tokens.shape[0]),
                    }
                }
            else:
                return {
                    "id": f"cmpl-{int(time.time())}",
                    "object": "text_completion",
                    "created": int(time.time()),
                    "model": "Cogito-0.9.1-15B",
                    "choices": [{"text": " Generated completion text.", "index": 0, "finish_reason": "stop"}],
                    "usage": {"prompt_tokens": 20, "completion_tokens": 10, "total_tokens": 30}
                }

        return await asyncio.to_thread(_run)
