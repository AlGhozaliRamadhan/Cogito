# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

Cogito-0.9.1-15B is a self-hosted, OpenAI-compatible REST API that serves the `Cogito-0.9.1-15B` safetensors model (HF repo `ozaa77/Cogito-0.9.1-15B`) through Hugging Face `transformers`, `torch`, `accelerate`, and `bitsandbytes`. It is consumed by external clients -- including frontend AI agents that ask questions over this API -- so OpenAI wire-compatibility (paths, SSE streaming, response shapes) matters.

The model is a 15B parameter model in sharded safetensors format (`model-00001-of-00008.safetensors` to `model-00008-of-00008.safetensors`). It supports:
- **`auto` mode**: automatically detects total GPU VRAM and loads with 4-bit NF4 quantization (`BitsAndBytesConfig`) if VRAM < 28 GB, enabling it to run smoothly on single 15-16 GB GPUs (such as Kaggle/Colab T4 or P100).
- **`4bit` mode**: explicit NF4 4-bit quantization (~9 GB VRAM).
- **`8bit` mode**: 8-bit quantization (~15 GB VRAM).
- **`16bit` mode**: full precision bfloat16/float16 (~30 GB VRAM across multi-GPU or high-memory instances).

The critical engineering invariant: preserve all 9 directives of the abliterated system prompt, the `<think>` reasoning format, and the 15 canonical stop tokens.

## Common commands

```bash
# Run the test suite
pytest -v

# Run the all-in-one manager
python cogito.py setup        # install deps + snapshot_download safetensors model (auto profile)
python cogito.py setup 4bit   # configure 4-bit NF4 quantization profile
python cogito.py setup 8bit   # configure 8-bit quantization profile
python cogito.py setup 16bit  # configure 16-bit float16/bfloat16 profile
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

- `src/config.py`: Environment and GPU detection (`detect_env`), model profiles, typed settings.
- `src/core/prompt.py`: Canonical abliterated system prompt, 9 directives, stop tokens, ChatML prompt builder.
- `src/core/stop_criteria.py`: High-performance `WindowedStringStopCriteria` (O(1) per forward step).
- `src/core/key_manager.py`: Thread-safe, crash-resilient `APIKeyManager` with atomic JSON persistence and sliding-window rate tracking.
- `src/core/engine.py`: `InferenceEngine` handling model lifecycle, BitsAndBytes quantization, and non-blocking generation.
- `src/server/app.py`: FastAPI application factory (`create_app`), CORS, GZip, and error formatting.
- `src/server/schemas.py`: OpenAI-compatible Pydantic request/response schemas.
- `src/server/routes/`: Modular route handlers (`health.py`, `models.py`, `chat.py`, `completions.py`, `admin.py`).
- `src/tunnel/cloudflare.py`: Cloudflare Quick Tunnel binary manager and public URL resolution.
- `src/supervisor/watchdog.py`: Keepalive thread and supervisor watchdog.
- `src/cli.py`: Unified CLI management interface.

## Output Taming & Safety

In `src/core/prompt.py`, `build_chatml_prompt()` prepends the canonical system prompt and formats turns in ChatML format.

- `/v1/chat/completions` appends stop tokens: `<|im_end|>`, `<|im_start|>`, `NdrFc`, `⊋`, `الحوثي`, `:UIControl`, `*angstrom`, `(egt)`, `<|eot_id|>`, `<|end_of_text|>`, `<|end_of_turn|>`, `ãeste`, `çãeste`, `iVar`, `прекрасн`, `建档立`.
- `WindowedStringStopCriteria` checks trailing generated tokens on each forward step.
- SSE heartbeats (`: heartbeat\n\n`) and headers (`Connection: close`, `X-Accel-Buffering: no`) prevent Cloudflare 502 Bad Gateway timeouts.

## Auth and Keys

- `APIKeyManager` (thread-safe) loads/saves keys atomically to `cogito_keys.json`. An `admin` key is auto-created on first boot.
- All non-health endpoints require a key via Bearer token or `x-api-key` header; admin endpoints additionally require `role == "admin"`. Per-key rate limiting is a fixed 60s sliding window, default 30 RPM.
