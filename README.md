# 🧠 Cogito-0.9 API

> **Turn your [Cogito-0.9](https://huggingface.co/ozaa77/Cogito-0.9) model into a free, OpenAI-compatible API.**
> Runs on Kaggle or Google Colab. Free tunnel included — no account, no token, just works.

---

## ⚡ TL;DR — Just paste these two cells

Open a Kaggle or Colab notebook, paste **Cell 1** then **Cell 2**, run them top to bottom.

---

### 📋 Cell 1 — Download & Setup

```python
import urllib.request, os

# Download cogito.py (the single all-in-one manager script)
urllib.request.urlretrieve(
    "https://raw.githubusercontent.com/AlGhozaliRamadhan/Cogito/main/cogito.py",
    "cogito.py"
)

# Install the two packages needed to run the CLI itself
import subprocess, sys
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "fastapi", "uvicorn[standard]",
                "python-multipart", "huggingface_hub", "pydantic", "requests"], check=True)

# (Optional) GPU check
subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader"],
               capture_output=False)
```

---

### 📋 Cell 2 — Choose model, start server + free tunnel

```python
# This launches the interactive CLI.
# It will ask you to pick a model, then start the server and tunnel.
# Your public URL and admin key are printed at the end.
import subprocess, sys
subprocess.run([sys.executable, "cogito.py", "start"])
```

> **That's it.** You'll see a box like this:
> ```
> ╔══════════════════════════════════════════════════════════╗
> ║  🎉  Cogito-0.9 API is LIVE!                             ║
> ║  🌐  URL:       https://xxxx.trycloudflare.com           ║
> ║  🔑  Admin key: cg-xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx      ║
> ║  📖  Docs:      https://xxxx.trycloudflare.com/docs      ║
> ╚══════════════════════════════════════════════════════════╝
> ```

---

## 🔑 Cell 3 — Create API keys (in a new cell)

```python
# Create a user key for your app / share with others
import subprocess, sys
subprocess.run([sys.executable, "cogito.py", "keys"])
```

This opens an interactive menu:
```
  [1] List all keys
  [2] Create new key     ← pick this, enter a name + rate limit
  [3] Revoke a key
  [0] Exit
```

---

## 🧪 Cell 4 — Test the API

```python
import subprocess, sys
subprocess.run([sys.executable, "cogito.py", "test"])
```

This shows a prompt where you type a question. The answer streams back token-by-token, so you know it's working.

---

## 📊 Cell 5 — Check status anytime

```python
import subprocess, sys
subprocess.run([sys.executable, "cogito.py", "status"])
```

Shows: current public URL, admin key, model loaded status, uptime, total requests.

---

## 🌐 Using the API

The API is **fully OpenAI-compatible**. Swap your `base_url` and you're done.

### cURL
```bash
curl -X POST "https://YOURS.trycloudflare.com/v1/chat/completions" \
  -H "Authorization: Bearer cg-YOUR_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "cogito-0.9-q4_k_m",
    "messages": [{"role": "user", "content": "Hello!"}],
    "max_tokens": 200
  }'
```

### Python (requests)
```python
import requests

r = requests.post(
    "https://YOURS.trycloudflare.com/v1/chat/completions",
    headers={"Authorization": "Bearer cg-YOUR_KEY"},
    json={
        "model": "cogito-0.9-q4_k_m",
        "messages": [{"role": "user", "content": "Explain quantum entanglement simply."}],
        "max_tokens": 300,
        "temperature": 0.7,
    }
)
print(r.json()["choices"][0]["message"]["content"])
```

### OpenAI Python SDK (drop-in)
```python
from openai import OpenAI

client = OpenAI(
    base_url="https://YOURS.trycloudflare.com/v1",
    api_key="cg-YOUR_KEY",
)

# Streaming
for chunk in client.chat.completions.create(
    model="cogito-0.9-q4_k_m",
    messages=[{"role": "user", "content": "Write a haiku about AI."}],
    stream=True,
):
    print(chunk.choices[0].delta.content or "", end="", flush=True)
```

### LangChain
```python
from langchain_openai import ChatOpenAI

llm = ChatOpenAI(
    base_url="https://YOURS.trycloudflare.com/v1",
    api_key="cg-YOUR_KEY",
    model="cogito-0.9-q4_k_m",
)
print(llm.invoke("What is the Cogito model?").content)
```

---

## 🔑 Admin API (key management via HTTP)

All admin calls use `Authorization: Bearer ADMIN_KEY`.

### Create a key
```bash
curl -X POST "https://YOURS.trycloudflare.com/v1/admin/keys/create" \
  -H "Authorization: Bearer ADMIN_KEY" \
  -H "Content-Type: application/json" \
  -d '{"name": "my-app", "role": "user", "rpm": 20}'
```
**Response:**
```json
{
  "success": true,
  "key": {
    "key": "cg-aBcDeFgH...",
    "name": "my-app",
    "role": "user",
    "rpm": 20,
    "reqs": 0,
    "active": true
  }
}
```

### List keys
```bash
curl "https://YOURS.trycloudflare.com/v1/admin/keys/list" \
  -H "Authorization: Bearer ADMIN_KEY"
```

### Revoke a key
```bash
curl -X POST "https://YOURS.trycloudflare.com/v1/admin/keys/revoke" \
  -H "Authorization: Bearer ADMIN_KEY" \
  -H "Content-Type: application/json" \
  -d '{"key": "cg-aBcDeFgH..."}'
```

### View usage stats
```bash
curl "https://YOURS.trycloudflare.com/v1/admin/stats" \
  -H "Authorization: Bearer ADMIN_KEY"
```

---

## 📡 API Endpoints

| Method | Path | Auth | Description |
|---|---|---|---|
| GET | `/` | None | Web dashboard |
| GET | `/health` | None | Server & model status |
| GET | `/ping` | None | Liveness probe |
| GET | `/v1/models` | API Key | List available models |
| POST | `/v1/chat/completions` | API Key | Chat completion (streaming ✓) |
| POST | `/v1/completions` | API Key | Text completion (streaming ✓) |
| POST | `/v1/admin/keys/create` | Admin | Create API key |
| GET | `/v1/admin/keys/list` | Admin | List all keys |
| POST | `/v1/admin/keys/revoke` | Admin | Revoke a key |
| GET | `/v1/admin/stats` | Admin | Usage statistics |
| GET | `/docs` | None | Swagger UI |

---

## 🛡️ Rate Limiting

Each API key has its own rate limit (requests per minute). When exceeded, the server returns `HTTP 429`.

| Key Role | Default RPM | Can be changed |
|---|---|---|
| `user` | 30 req/min | Yes, per-key |
| `admin` | Unlimited | — |

Create a high-rate key: `{"name": "power-user", "rpm": 120}`

---

## 🌐 The Tunnel

Uses **Cloudflare Quick Tunnel** — completely free, no account, no token needed.

- Binary is downloaded automatically from GitHub releases
- Creates a random `https://xxxx.trycloudflare.com` URL
- URL changes each session (just copy the new one from the output)
- Backed by Cloudflare's global CDN

The URL changes every time you restart. Users with existing API keys continue to work — just update the base URL.

---

## 🎛️ CLI Reference

```
python cogito.py              → interactive menu
python cogito.py setup        → install deps + pick & download model
python cogito.py setup q4_k_m → setup with specific model (no prompt)
python cogito.py setup q8_0   → setup with q8_0 model
python cogito.py start        → start server + tunnel
python cogito.py keys         → manage API keys (create/list/revoke)
python cogito.py test         → test the running API interactively
python cogito.py status       → show URL, keys, health
python cogito.py help         → show this help
```

---

## 🤖 Models

| File | Size | Speed | Quality |
|---|---|---|---|
| `cogito-0.9-q4_k_m.gguf` | ~5 GB | ⚡ Fast | Good |
| `cogito-0.9-q8_0.gguf` | ~9 GB | Medium | 🎯 Best |

---

## ⏱️ Session Limits

| Platform | Max Session | GPU |
|---|---|---|
| **Kaggle** | 12h (GPU) / 9h (CPU) | T4 x2 — Free |
| **Google Colab** | ~4h free / 12h Pro | T4 — Free tier |

**Tip:** Download `cogito_keys.json` before a session ends. Upload it next session to keep the same user keys.

---

## 📁 Repo Structure

```
Cogito/
├── cogito.py          ← Everything: CLI + embedded server + tunnel
├── server/
│   ├── api_server.py  ← Standalone FastAPI server (separate use)
│   └── requirements.txt
└── README.md
```
