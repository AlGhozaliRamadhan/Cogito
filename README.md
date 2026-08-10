# 🧠 Cogito-0.9 API

> **OpenAI-Compatible REST API for the [Cogito-0.9](https://huggingface.co/ozaa77/Cogito-0.9) language model — free, self-hosted on Kaggle or Google Colab.**

[![Open In Kaggle](https://kaggle.com/static/images/open-in-kaggle.svg)](https://kaggle.com)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com)

---

## ✨ Features

| Feature | Details |
|---|---|
| 🔌 **OpenAI Compatible** | Drop-in replacement for any OpenAI client |
| 🌊 **Streaming** | SSE streaming for real-time token output |
| 🔑 **API Key Auth** | Per-key rate limiting & usage tracking |
| 🌐 **Free Tunnel** | Cloudflared → ngrok → localtunnel → serveo (auto-failover) |
| 💓 **KeepAlive** | Prevents Kaggle/Colab from going idle |
| 📊 **Dashboard** | Beautiful web UI at your tunnel URL |
| ⚡ **CUDA Accelerated** | llama.cpp with full GPU offload |
| 📖 **Swagger UI** | Interactive API docs at `/docs` |

---

## 🚀 Quick Start

### Option A: Kaggle (Recommended — 12h GPU)
1. Upload `cogito_api_kaggle.ipynb` to Kaggle
2. Set **Runtime → GPU T4 x2**
3. Enable **Internet access**
4. Run all cells
5. Copy the public URL from the output

### Option B: Google Colab
1. Upload `cogito_api_colab.ipynb` to Colab
2. Set **Runtime → Change runtime type → T4 GPU**
3. Fill in the config form
4. Run all cells

---

## 📡 API Endpoints

### Chat Completion (OpenAI-compatible)
```bash
curl -X POST "https://YOUR-TUNNEL-URL/v1/chat/completions" \
  -H "Authorization: Bearer YOUR_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "cogito-0.9-q4_k_m",
    "messages": [
      {"role": "system", "content": "You are a helpful assistant."},
      {"role": "user", "content": "Hello!"}
    ],
    "max_tokens": 512,
    "temperature": 0.7
  }'
```

### With OpenAI Python SDK
```python
from openai import OpenAI

client = OpenAI(
    base_url="https://YOUR-TUNNEL-URL/v1",
    api_key="YOUR_API_KEY",
)

response = client.chat.completions.create(
    model="cogito-0.9-q4_k_m",
    messages=[{"role": "user", "content": "Explain quantum computing"}],
    stream=True,
)

for chunk in response:
    print(chunk.choices[0].delta.content or "", end="", flush=True)
```

### With LangChain
```python
from langchain_openai import ChatOpenAI

llm = ChatOpenAI(
    base_url="https://YOUR-TUNNEL-URL/v1",
    api_key="YOUR_API_KEY",
    model="cogito-0.9-q4_k_m",
)
```

---

## 🔑 API Key Management

All key management is done via the Admin API. You need the **admin key** (printed when the notebook starts).

### Create a Key
```bash
curl -X POST "https://YOUR-URL/v1/admin/keys/create" \
  -H "Authorization: Bearer ADMIN_KEY" \
  -H "Content-Type: application/json" \
  -d '{"name": "my-app", "role": "user", "rate_limit_rpm": 30}'
```

**Response:**
```json
{
  "success": true,
  "key": {
    "key": "cg-aBcD...",
    "name": "my-app",
    "role": "user",
    "rate_limit_rpm": 30,
    "created_at": "2026-08-10T...",
    "active": true
  }
}
```

### List All Keys
```bash
curl "https://YOUR-URL/v1/admin/keys/list" \
  -H "Authorization: Bearer ADMIN_KEY"
```

### Revoke a Key
```bash
curl -X POST "https://YOUR-URL/v1/admin/keys/revoke" \
  -H "Authorization: Bearer ADMIN_KEY" \
  -H "Content-Type: application/json" \
  -d '{"key": "cg-aBcD..."}'
```

### View Stats
```bash
curl "https://YOUR-URL/v1/admin/stats" \
  -H "Authorization: Bearer ADMIN_KEY"
```

---

## 🛡️ Rate Limiting

Each API key has its own rate limit (requests per minute). Default is 30 RPM.

- **Exceeded**: Returns `HTTP 429 Too Many Requests`
- **Admin keys**: No rate limit
- **Custom limits**: Set `rate_limit_rpm` when creating keys

---

## 🌐 Tunnel Providers

The system tries these providers in order and automatically falls back:

| Priority | Provider | Notes |
|---|---|---|
| 1st | **Cloudflared** | Most stable, Cloudflare CDN |
| 2nd | **ngrok** | Requires free account for longer sessions |
| 3rd | **localtunnel** | No account needed |
| 4th | **serveo** | SSH-based, no install needed |

> **Tip**: Get a free [ngrok authtoken](https://dashboard.ngrok.com/get-started/your-authtoken) for more reliable tunnels.

---

## ⚠️ Limitations

| Platform | Session Limit | GPU |
|---|---|---|
| Kaggle | 12h GPU / 9h CPU | T4 x2 (free) |
| Google Colab | ~3-4h free / 12h Pro | T4 (free) |

**Workaround**: Re-run the notebook when the session expires. API keys are persisted to `api_keys.json` — download and re-upload between sessions.

---

## 📁 Repository Structure

```
Cogito/
├── cogito_api_kaggle.ipynb     # Kaggle notebook (recommended)
├── cogito_api_colab.ipynb      # Google Colab notebook
└── server/
    ├── api_server.py            # FastAPI server (OpenAI-compatible)
    ├── tunnel_manager.py        # Multi-provider tunnel with failover
    └── requirements.txt         # Python dependencies
```

---

## 🔧 Models

| File | Size | Quality | Speed |
|---|---|---|---|
| `cogito-0.9-q4_k_m.gguf` | ~5 GB | Good | Fast ⚡ |
| `cogito-0.9-q8_0.gguf` | ~9 GB | Best | Medium |

---

## 📄 License

This API server code is open-source. The Cogito-0.9 model is hosted by [@ozaa77](https://huggingface.co/ozaa77) on HuggingFace.
