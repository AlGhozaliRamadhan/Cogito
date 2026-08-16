# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

Cogito-0.9.1-15B is a self-hosted, OpenAI-compatible REST API that serves the `Cogito-0.9.1-15B` safetensors model (HF repo `ozaa77/Cogito-0.9.1-15B`) through Hugging Face `transformers`, `torch`, `accelerate`, and `bitsandbytes`. It is consumed by external clients — including frontend AI agents that "ask questions" over this API — so OpenAI wire-compatibility (paths, SSE streaming, response shapes) matters.

The model is a 15B parameter model in sharded safetensors format (`model-00001-of-00008.safetensors` to `model-00008-of-00008.safetensors`). It supports:
- **`auto` mode**: automatically detects total GPU VRAM and loads with 4-bit NF4 quantization (`BitsAndBytesConfig`) if VRAM < 28 GB, enabling it to run smoothly on single 15-16 GB GPUs (such as Kaggle/Colab T4 or P100).
- **`4bit` mode**: explicit NF4 4-bit quantization (~9 GB VRAM).
- **`8bit` mode**: 8-bit quantization (~15 GB VRAM).
- **`16bit` mode**: full precision bfloat16/float16 (~30 GB VRAM across multi-GPU or high-memory instances).

The hard engineering problem here is **not** just the FastAPI plumbing: it is **taming the model's generation** so responses stop cleanly instead of looping, spamming tokens, or leaking internal tags. Output-quality regressions must be guarded when touching the prompt or generation code.

## Common commands

There is no build step or linter. Development is: edit → run the server → smoke-test against the live endpoint.

```bash
# Run the all-in-one manager (recommended path — sets up deps, downloads model,
# starts FastAPI, starts Cloudflare tunnel, keepalive)
python cogito.py setup        # install deps + snapshot_download safetensors model (auto profile)
python cogito.py setup 4bit   # configure 4-bit NF4 quantization profile
python cogito.py setup 8bit   # configure 8-bit quantization profile
python cogito.py setup 16bit  # configure 16-bit float16/bfloat16 profile
python cogito.py start        # start server + tunnel; blocks and auto-restarts both
python cogito.py keys         # interactive key manager (needs a TTY)
python cogito.py test         # send a streamed test prompt
python cogito.py status       # show URL, admin key, model, health
```

```bash
# Standalone server (separate implementation — see Architecture)
pip install -r server/requirements.txt
python server/api_server.py
```

Smoke tests can be run using `python test_502_prevention.py` and `python cogito.py test`.

## Architecture

### `cogito.py` — the live all-in-one manager

`cogito.py start` is the deployment path the README documents and what actually runs on Kaggle/Colab. It:

1. **Detects environment** (`detect_env()`): local vs Kaggle vs Colab, GPU presence via `nvidia-smi` / `torch.cuda`. Paths and defaults change per environment.
2. **Downloads model shards** via `snapshot_download` from `huggingface_hub` into `models/Cogito-0.9.1-15B/`.
3. **Persists runtime state** to `.cogito_state.json` in the working dir (`model_key`, `model_path`, `admin_key`, `public_url`). The admin key is generated on first run and reused from state afterward.
4. **Writes `SERVER_CODE`** — a raw-string FastAPI app — to `_cogito_server.py` and runs it as a **subprocess**, injecting all config through env vars (`COGITO_MODEL_PATH`, `COGITO_ADMIN_KEY`, `COGITO_KEYS_FILE`, `COGITO_QUANT`, `PORT`, …). The parent process monitors both the server and `cloudflared` tunnel subprocesses and restarts them on death.
5. `start_tunnel()` downloads `cloudflared` and parses the `https://…trycloudflare.com` URL from its output. The URL changes every session; keys persist.
6. A keepalive thread pings `/ping` so the notebook session doesn't idle out.

### Two server implementations — know which one you're editing

- **`SERVER_CODE` inside `cogito.py`** is the **live** server. Its chat endpoint calls `model.generate(...)` with `TextIteratorStreamer`, `StringStopCriteria`, and `BitsAndBytesConfig`.
- **`server/api_server.py`** is a **standalone** server mirror implementing the same `transformers` / `torch` pipeline.

### The output-taming system (the critical part)

In `SERVER_CODE`, `build_messages()` prepends a **canonical system prompt** (9 directives: epistemic rigor, collaborative agency, boundaries, superiority, writing style, contextual awareness, reasoning inside `<think>`/`</think>`, clean termination, RAW OUTPUT) and formats messages in ChatML format.

- `/v1/chat/completions` appends stop tokens: `<|im_end|>`, `<|im_start|>`, `NdrFc`, `⊋`, `الحوثي`, `:UIControl`, `*angstrom`, `(egt)`, `<|eot_id|>`, `<|end_of_text|>`, `<|end_of_turn|>`, `ãeste`, `çãeste`, `iVar`, `прекрасн`, `建档立`.
- `StringStopCriteria` stops model generation immediately when stop tokens appear, and `TextIteratorStreamer` trims the stream before emitting to client.
- SSE heartbeats (`: heartbeat\n\n`) and connection termination headers (`Connection: close`, `X-Accel-Buffering: no`) prevent Cloudflare 502 Bad Gateway timeouts during long generational pauses.

### Auth and keys

- `KeyManager` (thread-safe, in `SERVER_CODE`) loads/saves keys as JSON to `cogito_keys.json` (working dir; path overridable). An `admin` key is auto-created on first boot and migrated if `COGITO_ADMIN_KEY` changes.
- All non-health endpoints require a key via Bearer token or `x-api-key` header; admin endpoints additionally require `role == "admin"`. Per-key rate limiting is a fixed 60s sliding window, default 30 RPM (`COGITO_RPM`).

## Gotchas

- **Never commit** `.cogito_state.json`, `cogito_keys.json`, or API keys.
- Model downloads are sharded `.safetensors` files (~30 GB total across 8 shards), gitignored, fetched from HF into `models/Cogito-0.9.1-15B`.
- Context/window defaults: `MAX_CTX` 8192, `DEFAULT_TOKENS` 512, `repeat_penalty` default 1.1.
