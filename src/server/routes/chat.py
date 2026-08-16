"""
Chat Completions Endpoint (/v1/chat/completions)
"""

import time
import uuid
import json
import logging
import threading
import asyncio
from typing import Dict, Any
from fastapi import APIRouter, Request, Depends, HTTPException, status
from fastapi.responses import StreamingResponse
from transformers import TextIteratorStreamer

from src.config import settings
from src.core.prompt import build_chatml_prompt
from src.server.schemas import ChatCompletionRequest
from src.server.auth import get_api_key

logger = logging.getLogger("cogito-chat")
router = APIRouter(prefix="/v1", tags=["Chat"])

def ensure_engine_ready(engine):
    if not engine.is_ready():
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model is loading. Please retry in a few moments." if engine.model_loading else "Model is not loaded."
        )

@router.post("/chat/completions")
async def chat_completions(
    body: ChatCompletionRequest,
    request: Request,
    kd: Dict[str, Any] = Depends(get_api_key),
):
    engine = request.app.state.engine
    km = request.app.state.key_manager
    ensure_engine_ready(engine)

    prompt = build_chatml_prompt(body.messages) + "<think>\n"
    custom_stops = [body.stop] if isinstance(body.stop, str) else (body.stop or [])
    
    gen_kwargs, input_len, stop_list = engine.prepare_generation_args(
        prompt=prompt,
        max_tokens=body.max_tokens or settings.default_tokens,
        temperature=body.temperature if body.temperature is not None else 0.7,
        top_p=body.top_p if body.top_p is not None else 0.95,
        top_k=body.top_k or 40,
        repeat_penalty=body.repeat_penalty or 1.1,
        custom_stops=custom_stops,
    )

    request_id = f"chatcmpl-{uuid.uuid4().hex}"
    created_ts = int(time.time())

    # Streaming Execution
    if body.stream:
        streamer = TextIteratorStreamer(engine.tokenizer, skip_prompt=True, skip_special_tokens=False)
        gen_kwargs["streamer"] = streamer

        def run_gen():
            try:
                import torch
                with torch.no_grad():
                    with engine.lock:
                        engine.model.generate(**gen_kwargs)
            except Exception as e:
                logger.error(f"Generation error in streaming worker: {e}")

        gen_thread = threading.Thread(target=run_gen, daemon=True)
        gen_thread.start()

        async def stream_generator():
            tok_count = 0
            accumulated = ""
            stopped = False
            last_heartbeat = time.time()

            yield f"data: {json.dumps({'id': request_id, 'object': 'chat.completion.chunk', 'created': created_ts, 'model': body.model, 'choices': [{'index': 0, 'delta': {'role': 'assistant', 'content': ''}, 'finish_reason': None}]})}\n\n"
            yield f"data: {json.dumps({'id': request_id, 'object': 'chat.completion.chunk', 'created': created_ts, 'model': body.model, 'choices': [{'index': 0, 'delta': {'content': '<think>\n'}, 'finish_reason': None}]})}\n\n"

            def get_next_token():
                try:
                    return next(streamer)
                except StopIteration:
                    return None
                except Exception:
                    return None

            try:
                while True:
                    if await request.is_disconnected():
                        logger.info(f"Client disconnected: {request_id}")
                        break

                    text_chunk = await asyncio.to_thread(get_next_token)
                    if text_chunk is None:
                        break

                    now = time.time()
                    if now - last_heartbeat >= settings.sse_heartbeat_secs:
                        yield ": heartbeat\n\n"
                        last_heartbeat = now

                    accumulated += text_chunk
                    hit_stop = None
                    for sw in stop_list:
                        if sw in accumulated:
                            hit_stop = sw
                            break

                    if hit_stop:
                        pre_stop = accumulated.split(hit_stop)[0]
                        remaining_to_send = pre_stop[len(accumulated) - len(text_chunk) - len(hit_stop):]
                        if remaining_to_send:
                            tok_count += 1
                            yield f"data: {json.dumps({'id': request_id, 'object': 'chat.completion.chunk', 'created': created_ts, 'model': body.model, 'choices': [{'index': 0, 'delta': {'content': remaining_to_send}, 'finish_reason': None}]})}\n\n"
                        stopped = True
                        break
                    else:
                        tok_count += 1
                        yield f"data: {json.dumps({'id': request_id, 'object': 'chat.completion.chunk', 'created': created_ts, 'model': body.model, 'choices': [{'index': 0, 'delta': {'content': text_chunk}, 'finish_reason': None}]})}\n\n"

                yield f"data: {json.dumps({'id': request_id, 'object': 'chat.completion.chunk', 'created': int(time.time()), 'model': body.model, 'choices': [{'index': 0, 'delta': {}, 'finish_reason': 'stop'}]})}\n\n"
                yield "data: [DONE]\n\n"
            except Exception as e:
                logger.error(f"Streaming error: {e}")
                try:
                    yield "data: [DONE]\n\n"
                except Exception:
                    pass
            finally:
                km.record_usage(kd["key"], tok_count)

        resp = StreamingResponse(stream_generator(), media_type="text/event-stream")
        resp.headers["Connection"] = "close"
        resp.headers["X-Accel-Buffering"] = "no"
        return resp

    # Non-Streaming Execution
    clean_text, prompt_tokens, comp_tokens = await engine.generate_non_streaming(gen_kwargs, input_len, stop_list)
    content = "<think>\n" + clean_text
    total_tokens = prompt_tokens + comp_tokens
    km.record_usage(kd["key"], total_tokens)

    return {
        "id": request_id,
        "object": "chat.completion",
        "created": created_ts,
        "model": body.model,
        "choices": [{
            "index": 0,
            "message": {"role": "assistant", "content": content},
            "finish_reason": "stop"
        }],
        "usage": {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": comp_tokens,
            "total_tokens": total_tokens,
        }
    }
