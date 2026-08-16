# Cogito-0.9.1-15B API

Self-hosted, OpenAI-compatible REST API for [Cogito-0.9.1-15B](https://huggingface.co/ozaa77/Cogito-0.9.1-15B) (Safetensors).
Runs on Kaggle, Google Colab, or local Linux/Windows systems. Automatic Cloudflare tunnel included -- no account or token required.

---

## Run

Paste the corresponding cell into your Kaggle or Colab notebook and run it.
It pulls the latest version from this repository before starting.

### For Kaggle

```python
import os

# 1. Start from base Kaggle working directory
%cd /kaggle/working

# 2. Clone or pull latest
if not os.path.exists("Cogito"):
    print("Repository not found. Cloning...")
    !git clone https://github.com/AlGhozaliRamadhan/Cogito.git
else:
    print("Repository found. Checking for updates...")
    %cd /kaggle/working/Cogito
    !git stash
    !git pull origin main
    !git stash drop
    %cd /kaggle/working

# 3. Enter directory and launch
%cd /kaggle/working/Cogito
!pip install -q fastapi "uvicorn[standard]" python-multipart huggingface_hub pydantic requests transformers accelerate safetensors bitsandbytes sentencepiece tiktoken pytest pytest-asyncio httpx
!python cogito.py start
```

### For Google Colab

```python
import os

# 1. Start from base Colab working directory
%cd /content

# 2. Clone or pull latest
if not os.path.exists("Cogito"):
    print("Repository not found. Cloning...")
    !git clone https://github.com/AlGhozaliRamadhan/Cogito.git
else:
    print("Repository found. Checking for updates...")
    %cd /content/Cogito
    !git stash
    !git pull origin main
    !git stash drop
    %cd /content

# 3. Enter directory and launch
%cd /content/Cogito
!pip install -q fastapi "uvicorn[standard]" python-multipart huggingface_hub pydantic requests transformers accelerate safetensors bitsandbytes sentencepiece tiktoken pytest pytest-asyncio httpx
!python cogito.py start
```

When the server is ready you will see:

```text
  +----------------------------------------------------------+
  |  Cogito-0.9.1-15B API is LIVE                            |
  |  URL:       https://xxxx.trycloudflare.com               |
  |  Admin key: cg-xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx          |
  |  Docs:      https://xxxx.trycloudflare.com/docs          |
  +----------------------------------------------------------+
```

---

## CLI Commands

Run these commands in a terminal or separate notebook cells:

```bash
# Manage API keys
python cogito.py keys

# Show current URL, admin key, model status, and uptime
python cogito.py status

# Run setup (download model and configure profile)
python cogito.py setup
python cogito.py setup 4bit
python cogito.py setup 8bit
python cogito.py setup 16bit

# Start server and Cloudflare tunnel
python cogito.py start
```

---

## Running the Automated Test Suite

Run the full pytest suite:

```bash
pytest -v
```

---

## Using the API

The API is fully OpenAI-compatible. Set the base URL in any standard client.

### cURL

```bash
curl -X POST "https://YOURS.trycloudflare.com/v1/chat/completions" \
  -H "Authorization: Bearer cg-YOUR_KEY" \
  -H "Content-Type: application/json" \
  -d '{"model":"Cogito-0.9.1-15B","messages":[{"role":"user","content":"Hello"}],"max_tokens":200}'
```

### Python (Requests)

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

### OpenAI Python SDK

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
| DELETE | `/v1/admin/keys/{key}` | Admin | Delete a key |
| GET | `/v1/admin/stats` | Admin | Usage statistics |
| GET | `/docs` | None | Swagger UI |

---

## Rate Limiting

Each API key has its own sliding-window rate limit in requests per minute (RPM). Exceeding it returns HTTP 429.

| Role | Default RPM |
|---|---|
| user | 30 |
| admin | unlimited |

---

## Models and Quantization

The model is loaded from [ozaa77/Cogito-0.9.1-15B](https://huggingface.co/ozaa77/Cogito-0.9.1-15B) in safetensors format.

| Profile | Format | Target VRAM | Notes |
|---|---|---|---|
| `auto` (default) | Safetensors | Auto | Auto-detects VRAM and applies 4-bit NF4 if VRAM < 28GB |
| `4bit` | Safetensors (NF4) | ~9 GB VRAM | Fits single 15-16GB GPU (Kaggle/Colab T4/P100) |
| `8bit` | Safetensors (Int8) | ~15 GB VRAM | Balanced quality and speed |
| `16bit` | Safetensors (FP16/BF16) | ~30 GB VRAM | Full precision (Multi-GPU 2xT4 / A100) |

---

## Repository Structure

```text
Cogito/
├── src/
│   ├── config.py              configuration and environment detection
│   ├── cli.py                 CLI command management
│   ├── core/                  inference engine, prompt builder, stop criteria, key manager
│   ├── server/                FastAPI application factory, routers, schemas, auth
│   ├── tunnel/                Cloudflare tunnel management
│   └── supervisor/            keepalive and watchdog supervision
├── tests/                     comprehensive pytest test suite
├── requirements.txt           pinned server and testing dependencies
├── cogito.py                  CLI and notebook runner
└── README.md
```

