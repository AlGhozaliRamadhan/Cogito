# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

Cogito-0.9 is a self-hosted, OpenAI-compatible REST API that serves the `Cogito-0.9` GGUF model (HF repo `ozaa77/Cogito-0.9`) through `llama-cpp-python`. It is consumed by external clients — including frontend AI agents that "ask questions" over this API — so OpenAI wire-compatibility (paths, SSE streaming, response shapes) matters.

The hard engineering problem here is **not** the FastAPI plumbing: it is **taming the model's generation** so responses stop cleanly instead of looping, spamming tokens, or leaking internal tags. Nearly the entire git history is a sequence of stop-token and system-prompt fixes. Treat output-quality regressions as the primary risk when touching the prompt or generation code.

## Common commands

There is no build step, linter, or test suite. Development is: edit → run the server → smoke-test against the live endpoint.

```bash
# Run the all-in-one manager (recommended path — sets up deps, downloads model,
# starts FastAPI, starts Cloudflare tunnel, keepalive)
python cogito.py setup        # install deps + download model (q4_k_m default)
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

Smoke tests are the untracked `test_api*.py` scripts in the repo root. They `requests.post` to a hardcoded live `trycloudflare.com` URL with a hardcoded key, and write results to `test_out*.txt`. They only work while a `cogito.py start` session is running with the matching tunnel.

## Architecture

### `cogito.py` — the live all-in-one manager

`cogito.py start` is the deployment path the README documents and what actually runs on Kaggle/Colab. It:

1. **Detects environment** (`detect_env()`): local vs Kaggle vs Colab, GPU presence via `nvidia-smi`. Paths and defaults change per environment.
2. **Persists runtime state** to `.cogito_state.json` in the working dir (`model_key`, `model_path`, `admin_key`, `public_url`). The admin key is generated on first run and reused from state afterward — losing this file means losing the key.
3. **Writes `SERVER_CODE`** — a raw-string FastAPI app — to `_cogito_server.py` and runs it as a **subprocess**, injecting all config through env vars (`COGITO_MODEL_PATH`, `COGITO_ADMIN_KEY`, `COGITO_KEYS_FILE`, `COGITO_GPU_LAYERS`, `PORT`, …). The parent process monitors both the server and `cloudflared` tunnel subprocesses and restarts them on death.
4. `start_tunnel()` downloads `cloudflared` and parses the `https://…trycloudflare.com` URL from its output. The URL changes every session; keys persist.
5. A keepalive thread pings `/ping` so the notebook session doesn't idle out.

### Two server implementations — know which one you're editing

- **`SERVER_CODE` inside `cogito.py`** is the **live** server. Its chat endpoint calls `llm.create_chat_completion(messages=msgs, …)` (native chat formatting + EOS handling). This is the current one.
- **`server/api_server.py`** is a **separate, older-style standalone** server. It hand-assembles a `<|system|>/<|user|>/<|assistant|>` prompt string and calls `llm(prompt, …)`. It is **not** what `cogito.py start` runs. It has a different env-var surface (`MODEL_PATH`, `API_KEYS_FILE`, `ADMIN_KEY`, `MAX_CONTEXT`, `RATE_LIMIT_RPM`) and slightly different keys-file schema (`rate_limit_rpm`/`total_requests` vs `rpm`/`reqs`). Edits to prompt/stop-token logic almost always belong in `SERVER_CODE`; keep `api_server.py` in mind only if the standalone path is the deployment target.

### The output-taming system (the critical part)

In `SERVER_CODE`, `build_messages()` prepends a **canonical system prompt** (9 directives: epistemic rigor, collaborative agency, boundaries, superiority, writing style, contextual awareness, reasoning inside `<think>`/`</think>`, clean termination, RAW OUTPUT) and concatenates any user-provided system message onto it. The prompt and the stop-token list are tightly coupled:

- `/v1/chat/completions` appends hardcoded stop tokens to any user-supplied `stop`: `<|im_end|>`, `<|im_start|>`, `NdrFc`, `⊋`, `الحوثي`, `:UIControl`, `*angstrom`, `(egt)`, `<|eot_id|>`, `<|end_of_text|>`, `<|end_of_turn|>`, `ãeste`, `çãeste`, `iVar`.
- Each token exists because of a real observed failure: `:UIControl`/`*angstrom`/`(egt)` for token spam, `NdrFc`/`⊋`/`الحوثي`/`ãeste`/`çãeste`/`iVar` for weird single-token leaks, the EOS tokens for infinite loops, and the RAW OUTPUT rule to stop `<action>` tags and `<b>Response:</b>` headers leaking into output.
- When adding a new prompt directive to fight a new failure mode, also add its telltale output fragment to the stop list — and vice versa.

### Auth and keys

- `KeyManager` (thread-safe, in `SERVER_CODE`) loads/saves keys as JSON to `cogito_keys.json` (working dir; path overridable). An `admin` key is auto-created on first boot and migrated if `COGITO_ADMIN_KEY` changes.
- All non-health endpoints require a key via Bearer token or `x-api-key` header; admin endpoints additionally require `role == "admin"`. Per-key rate limiting is a fixed 60s sliding window, default 30 RPM (`COGITO_RPM`).

## Gotchas

- **Never commit** `.cogito_state.json`, `cogito_keys.json`, or the API keys in the `test_api*.py` scripts — `.gitignore` covers `*.keys.json` and `*.log` but the test scripts (which embed live keys) are currently untracked scratch files.
- Model downloads are 5–9 GB (`.gguf`), gitignored, fetched from HF into `models/`. Local runs require the model file present or a full re-download.
- The dashboard at `/` is a giant HTML/CSS/JS string constant embedded in the server file — edit it in place.
- Context/window defaults: `MAX_CTX` 4096, `DEFAULT_TOKENS` 512, `repeat_penalty` default 1.0 (raised from a higher default specifically to stop end-of-generation hallucination).
