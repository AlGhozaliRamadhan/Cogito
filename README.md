# Cogito-0.9.1-15B API

Self-hosted, OpenAI-compatible REST API for [Cogito-0.9.1-15B](https://huggingface.co/ozaa77/Cogito-0.9.1-15B) (Safetensors).
Runs on Kaggle or Google Colab. Free Cloudflare tunnel — no account, no token required.

---

## Run

Paste the corresponding cell into your Kaggle or Colab notebook and run it.
It always pulls the latest version of `cogito.py` from this repo before starting.

### For Kaggle

```python
import os

# 1. Always start from the base Kaggle working directory
%cd /kaggle/working

# 2. Clone if it doesn't exist, otherwise force-pull the latest changes
if not os.path.exists("Cogito"):
    print("Repository not found. Cloning...")
    !git clone https://github.com/AlGhozaliRamadhan/Cogito.git
else:
    print("Repository found. Checking for updates...")
    %cd /kaggle/working/Cogito
    # Stash any local accidental changes, pull latest, and clear stash
    !git stash
    !git pull origin main
    !git stash drop
    %cd /kaggle/working

# 3. Enter the directory for setup
%cd /kaggle/working/Cogito

# 4. Install dependencies and start the server
!pip install -q fastapi "uvicorn[standard]" python-multipart huggingface_hub pydantic requests transformers accelerate safetensors bitsandbytes sentencepiece tiktoken
!python cogito.py start
```

### For Google Colab

```python
import os

# 1. Always start from the base Colab working directory
%cd /content

# 2. Clone if it doesn't exist, otherwise force-pull the latest changes
if not os.path.exists("Cogito"):
    print("Repository not found. Cloning...")
    !git clone https://github.com/AlGhozaliRamadhan/Cogito.git
else:
    print("Repository found. Checking for updates...")
    %cd /content/Cogito
    # Stash any local accidental changes, pull latest, and clear stash
    !git stash
    !git pull origin main
    !git stash drop
    %cd /content

# 3. Enter the directory for setup
%cd /content/Cogito

# 4. Install dependencies and start the server
!pip install -q fastapi "uvicorn[standard]" python-multipart huggingface_hub pydantic requests transformers accelerate safetensors bitsandbytes sentencepiece tiktoken
!python cogito.py start
```

When the server is ready you will see:

```
  +----------------------------------------------------------+
  |  Cogito-0.9.1-15B API is LIVE                            |
  |  URL:       https://xxxx.trycloudflare.com               |
  |  Admin key: cg-xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx          |
  |  Docs:      https://xxxx.trycloudflare.com/docs          |
  +----------------------------------------------------------+
```

The cell keeps the server running. Open a second cell for key management or testing.

---

## Other commands (run in separate cells)

```python
# Create, list, and revoke API keys
!python cogito.py keys
```

```python
# Send a test prompt and stream the response
!python cogito.py test
```

```python
# Show current URL, admin key, model status, uptime
!python cogito.py status
```

---

## Using the API

The API is fully OpenAI-compatible. Change `base_url` in any existing client and it works.

### curl

```bash
curl -X POST "https://YOURS.trycloudflare.com/v1/chat/completions" \
  -H "Authorization: Bearer cg-YOUR_KEY" \
  -H "Content-Type: application/json" \
  -d '{"model":"Cogito-0.9.1-15B","messages":[{"role":"user","content":"Hello"}],"max_tokens":200}'
```

### Python

```python
import requests

r = requests.post(
    "https://YOURS.trycloudflare.com/v1/chat/completions",
    headers={"Authorization": "Bearer cg-YOUR_KEY"},
    json={
        "model": "Cogito-0.9.1-15B",
        "messages": [{"role": "user", "content": "Explain quantum entanglement simply."}],
        "max_tokens": 300,
        "temperature": 0.7,
    }
)
print(r.json()["choices"][0]["message"]["content"])
```

### OpenAI SDK

```python
from openai import OpenAI

client = OpenAI(
    base_url="https://YOURS.trycloudflare.com/v1",
    api_key="cg-YOUR_KEY",
)

for chunk in client.chat.completions.create(
    model="Cogito-0.9.1-15B",
    messages=[{"role": "user", "content": "What can you do?"}],
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
    model="Cogito-0.9.1-15B",
)
print(llm.invoke("Who are you?").content)
```

---

## Admin API

All admin endpoints require `Authorization: Bearer ADMIN_KEY`.

### Create a key

```bash
curl -X POST "https://YOURS.trycloudflare.com/v1/admin/keys/create" \
  -H "Authorization: Bearer ADMIN_KEY" \
  -H "Content-Type: application/json" \
  -d '{"name": "my-app", "role": "user", "rpm": 20}'
```

Response:

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

### Stats

```bash
curl "https://YOURS.trycloudflare.com/v1/admin/stats" \
  -H "Authorization: Bearer ADMIN_KEY"
```

---

## Endpoints

| Method | Path | Auth | Description |
|---|---|---|---|
| GET | `/` | None | Web dashboard |
| GET | `/health` | None | Server and model status |
| GET | `/ping` | None | Liveness probe |
| GET | `/v1/models` | API Key | List models |
| POST | `/v1/chat/completions` | API Key | Chat completion, streaming supported |
| POST | `/v1/completions` | API Key | Text completion, streaming supported |
| POST | `/v1/admin/keys/create` | Admin | Create API key |
| GET | `/v1/admin/keys/list` | Admin | List all keys |
| POST | `/v1/admin/keys/revoke` | Admin | Revoke a key |
| GET | `/v1/admin/stats` | Admin | Usage statistics |
| GET | `/docs` | None | Swagger UI |

---

## Rate limiting

Each API key has its own rate limit in requests per minute. Exceeding it returns HTTP 429.

| Role | Default RPM |
|---|---|
| user | 30 |
| admin | unlimited |

Set a custom limit when creating a key: `"rpm": 120`

---

## Tunnel

Uses Cloudflare Quick Tunnel (`trycloudflare.com`). Completely free, no account needed.
The binary is downloaded automatically. The URL is random and changes each session.
Existing API keys keep working — only the base URL needs updating.

---

## Models & Quantization

The model is loaded from [ozaa77/Cogito-0.9.1-15B](https://huggingface.co/ozaa77/Cogito-0.9.1-15B) in **safetensors** format.

| Profile | Format | Target VRAM | Notes |
|---|---|---|---|
| `auto` (default) | Safetensors | Auto | Auto-detects VRAM & applies 4-bit NF4 if VRAM < 28GB |
| `4bit` | Safetensors (NF4) | ~9 GB VRAM | Fits single 15-16GB GPU (Kaggle/Colab T4/P100) |
| `8bit` | Safetensors (Int8) | ~15 GB VRAM | Balanced quality & speed |
| `16bit` | Safetensors (FP16/BF16) | ~30 GB VRAM | Full precision (Multi-GPU 2xT4 / A100) |

---

## CLI reference

```
python cogito.py              interactive menu
python cogito.py setup        install deps and download model (auto profile)
python cogito.py setup 4bit   skip prompt, download & configure 4-bit NF4 profile
python cogito.py setup 8bit   skip prompt, download & configure 8-bit profile
python cogito.py setup 16bit  skip prompt, download & configure full precision profile
python cogito.py start        start server and tunnel
python cogito.py keys         create, list, revoke keys
python cogito.py test         send a test prompt
python cogito.py status       show URL, keys, health
```

---

## Session limits

| Platform | GPU | Max session |
|---|---|---|
| Kaggle | T4 x2, free | 12h (GPU) / 9h (CPU) |
| Google Colab | T4, free tier | ~4h free, 12h Pro |

To keep API keys between sessions, download `cogito_keys.json` before the session ends and upload it at the start of the next one.

---

## Repo structure

```
Cogito/
├── cogito.py          all-in-one: CLI, embedded server, tunnel manager
├── server/
│   ├── api_server.py  standalone FastAPI server
│   └── requirements.txt
└── README.md
```
