# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

Cogito-0.9.1-15B is a self-hosted, OpenAI-compatible REST API that serves the `Cogito-0.9.1-15B-GGUF` model (HF repo `ozaa77/Cogito-0.9.1-15B-GGUF`) through high-performance `llama-cpp-python` and native C++ GGML/CUDA backends with FlashAttention and context caching. It is consumed by external clients -- including frontend AI agents that ask questions over this API -- so OpenAI wire-compatibility (paths, SSE streaming, response shapes) matters.

The model is a 15B parameter reasoning model in GGUF format. It supports:
- **`auto` mode**: defaults to Q4_K_M (~8.85 GB weight size), fitting comfortably in single 12-16 GB GPUs (Kaggle/Colab T4, RTX 3060/4060Ti/4070/4080) with full 32k context support.
- **`4bit` / `q4_k_m` mode**: 4-bit medium quantization (~8.85 GB VRAM).
- **`5bit` / `q5_k_m` mode**: 5-bit medium quantization (~10.6 GB VRAM).
- **`8bit` / `q8_0` mode**: 8-bit quantization (~16.1 GB VRAM, near lossless).
- **`16bit` / `f16` mode**: full precision float16 (~30.8 GB VRAM).

The critical engineering invariant: preserve all 9 directives of the abliterated system prompt, the `<think>` reasoning format, and the canonical stop tokens.

## Common commands

```bash
# Run the test suite
pytest -v

# Run the all-in-one manager
python cogito.py setup        # install deps + download GGUF model (auto/Q4_K_M profile)
python cogito.py setup q4_k_m # configure Q4_K_M quantization profile
python cogito.py setup q5_k_m # configure Q5_K_M quantization profile
python cogito.py setup q8_0   # configure Q8_0 quantization profile
python cogito.py start        # start server + tunnel; blocks and auto-restarts both
python cogito.py keys         # interactive key manager (needs a TTY)
python cogito.py status       # show URL, admin key, model, health
```

```bash
# Standalone server
pip install -r requirements.txt
python -m src.server.app
```

## Architecture

The codebase is organized into clean, modular packages under `src/`:

- `src/config.py`: Environment and GPU detection (`detect_env`), GGUF model profiles, typed settings.
- `src/core/prompt.py`: Canonical abliterated system prompt, 9 directives, stop tokens, ChatML prompt builder, and structured message formatter.
- `src/core/stop_criteria.py`: Native C++ stop list resolution and fallback stopping criteria.
- `src/core/key_manager.py`: Thread-safe `APIKeyManager` with in-memory dirty tracking, periodic 30s background flusher, and atomic disk persistence.
- `src/core/engine.py`: `InferenceEngine` handling GGUF lifecycle via `llama_cpp.Llama`, GPU layer offloading (`n_gpu_layers=-1`), FlashAttention, and non-blocking streaming.
- `src/server/app.py`: FastAPI application factory (`create_app`), CORS, GZip, and OpenAI-compatible error formatting.
- `src/server/schemas.py`: OpenAI-compatible Pydantic request/response schemas (with `min_p: 0.05`, `temperature: 0.70`, `top_p: 0.90`, `repeat_penalty: 1.08`).
- `src/server/routes/`: Modular route handlers (`health.py`, `models.py`, `chat.py`, `completions.py`, `admin.py`).
- `src/tunnel/cloudflare.py`: Cloudflare Quick Tunnel binary manager and public URL resolution.
- `src/supervisor/watchdog.py`: Keepalive thread and supervisor watchdog.
- `src/cli.py`: Unified CLI management interface for single-file GGUF management.

## Output Taming & Safety

In `src/core/prompt.py`, `prepare_chat_messages()` prepends the canonical system prompt and structures turns in ChatML format.

- Stop tokens: `<|im_end|>`, `<|im_start|>`, `NdrFc`, `⊋`, `الحوثي`, `:UIControl`, `*angstrom`, `(egt)`, `<|eot_id|>`, `<|end_of_text|>`, `<|end_of_turn|>`, `ãeste`, `çãeste`, `iVar`, `прекрасн`, `建档立`.
- Stop token matching is executed directly in `llama.cpp`'s high-speed C++ engine.
- SSE heartbeats (`: heartbeat\n\n`) and headers (`Connection: close`, `X-Accel-Buffering: no`) prevent Cloudflare 502 Bad Gateway timeouts.
- Active disconnection cancellation terminates background generation immediately if a client disconnects.

## Auth and Keys

- `APIKeyManager` tracks usage in-memory with zero disk latency on the request path, syncing to disk every 30 seconds.
- All non-health endpoints require a key via Bearer token or `x-api-key` header; admin endpoints additionally require `role == "admin"`. Per-key rate limiting is a fixed 60s sliding window, default 30 RPM.
