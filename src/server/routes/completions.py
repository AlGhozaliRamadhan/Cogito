"""
Text Completions Endpoint (/v1/completions)
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
from src.server.schemas import CompletionRequest
from src.server.auth import get_api_key

logger = logging.getLogger("cogito-completions")
router = APIRouter(prefix="/v1", tags=["Completions"])

def ensure_engine_ready(engine):
    if not engine.is_ready():
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model is loading. Please retry in a few moments." if engine.model_loading else "Model is not loaded."
        )

@router.post("/completions")
async def text_completions(
    body: CompletionRequest,
    request: Request,
    kd: Dict[str, Any] = Depends(get_api_key),
):
    engine = request.app.state.engine
    km = request.app.state.key_manager
    ensure_engine_ready(engine)

    custom_stops = [body.stop] if isinstance(body.stop, str) else (body.stop or [])
    gen_kwargs, input_len, stop_list = engine.prepare_generation_args(
        prompt=body.prompt,
        max_tokens=body.max_tokens or settings.default_tokens,
        temperature=body.temperature if body.temperature is not None else 0.7,
        top_p=body.top_p if body.top_p is not None else 0.95,
        top_k=body.top_k or 40,
        repeat_penalty=body.repeat_penalty or 1.1,
        custom_stops=custom_stops,
    )

    request_id = f"cmpl-{uuid.uuid4().hex}"
    created_ts = int(time.time())

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
                        remaining = pre_stop[len(accumulated) - len(text_chunk) - len(hit_stop):]
                        if remaining:
                            tok_count += 1
                            chunk_data = {"id": request_id, "object": "text_completion", "created": created_ts, "model": body.model, "choices": [{"text": remaining, "index": 0, "finish_reason": None}]}
                            yield f"data: {json.dumps(chunk_data)}\n\n"
                        stopped = True
                        break
                    else:
                        tok_count += 1
                        chunk_data = {"id": request_id, "object": "text_completion", "created": created_ts, "model": body.model, "choices": [{"text": text_chunk, "index": 0, "finish_reason": None}]}
                        yield f"data: {json.dumps(chunk_data)}\n\n"

                yield f"data: {json.dumps({'id': request_id, 'object': 'text_completion', 'created': int(time.time()), 'model': body.model, 'choices': [{'text': '', 'index': 0, 'finish_reason': 'stop'}]})}\n\n"
                yield "data: [DONE]\n\n"
            except Exception as e:
                logger.error(f"Stream text error: {e}")
            finally:
                km.record_usage(kd["key"], tok_count)

        resp = StreamingResponse(stream_generator(), media_type="text/event-stream")
        resp.headers["Connection"] = "close"
        resp.headers["X-Accel-Buffering"] = "no"
        return resp

    clean_text, prompt_tokens, comp_tokens = await engine.generate_non_streaming(gen_kwargs, input_len, stop_list)
    total_tokens = prompt_tokens + comp_tokens
    km.record_usage(kd["key"], total_tokens)

    return {
        "id": request_id,
        "object": "text_completion",
        "created": created_ts,
        "model": body.model,
        "choices": [{"text": clean_text, "index": 0, "finish_reason": "stop"}],
        "usage": {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": comp_tokens,
            "total_tokens": total_tokens,
        }
    }
