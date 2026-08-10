#!/usr/bin/env python3
"""
╔══════════════════════════════════════════════════════════════╗
║            🧠 Cogito-0.9 API Manager                        ║
║  One script to rule them all: setup → serve → tunnel → keys ║
╚══════════════════════════════════════════════════════════════╝

Usage (paste in Kaggle/Colab cell, or run locally):
    python cogito.py            → interactive menu
    python cogito.py setup      → install deps + download model
    python cogito.py start      → start server + tunnel
    python cogito.py keys       → manage API keys
    python cogito.py test       → test the API
    python cogito.py status     → show URL, keys, health
"""

import os, sys, json, time, socket, signal, secrets, threading, subprocess
import urllib.request, urllib.error, shutil, platform, textwrap
from pathlib import Path
from typing import Optional


# ══════════════════════════════════════════════════════════════════════════════
# ENVIRONMENT DETECTION
# ══════════════════════════════════════════════════════════════════════════════

def detect_env() -> dict:
    """Detect runtime environment and available hardware."""
    env = {
        "name": "local",
        "is_kaggle": False,
        "is_colab": False,
        "is_gpu": False,
        "gpu_name": None,
        "work_dir": str(Path.cwd()),
        "model_dir": str(Path.cwd() / "models"),
    }

    if os.path.exists("/kaggle"):
        env["name"] = "kaggle"
        env["is_kaggle"] = True
        env["work_dir"] = "/kaggle/working"
        env["model_dir"] = "/kaggle/working/models"
    elif os.path.exists("/content") and os.path.exists("/usr/local/lib/python3"):
        env["name"] = "colab"
        env["is_colab"] = True
        env["work_dir"] = "/content"
        env["model_dir"] = "/content/models"

    # GPU detection
    try:
        r = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=5,
        )
        if r.returncode == 0 and r.stdout.strip():
            env["is_gpu"] = True
            env["gpu_name"] = r.stdout.strip().split("\n")[0].split(",")[0].strip()
    except Exception:
        pass

    return env


ENV = detect_env()
WORK_DIR   = Path(ENV["work_dir"])
MODEL_DIR  = Path(ENV["model_dir"])
KEYS_FILE  = WORK_DIR / "cogito_keys.json"
SERVER_LOG = WORK_DIR / "cogito_server.log"
STATE_FILE = WORK_DIR / ".cogito_state.json"
PORT       = int(os.environ.get("COGITO_PORT", "8000"))

MODELS = {
    "q4_k_m": {
        "file": "cogito-0.9-q4_k_m.gguf",
        "size": "~5 GB",
        "description": "Q4_K_M · Faster, lower VRAM, good quality ⚡",
        "gpu_layers_default": -1,
    },
    "q8_0": {
        "file": "cogito-0.9-q8_0.gguf",
        "size": "~9 GB",
        "description": "Q8_0 · Best quality, needs more VRAM 🎯",
        "gpu_layers_default": -1,
    },
}
HF_REPO = "ozaa77/Cogito-0.9"


# ══════════════════════════════════════════════════════════════════════════════
# TERMINAL COLORS (no external deps)
# ══════════════════════════════════════════════════════════════════════════════

USE_COLOR = sys.stdout.isatty() or ENV["is_kaggle"] or ENV["is_colab"]

def c(text, *codes):
    if not USE_COLOR:
        return text
    return f"\033[{';'.join(str(x) for x in codes)}m{text}\033[0m"

def bold(t):     return c(t, 1)
def dim(t):      return c(t, 2)
def green(t):    return c(t, 32)
def yellow(t):   return c(t, 33)
def blue(t):     return c(t, 34)
def cyan(t):     return c(t, 36)
def red(t):      return c(t, 31)
def magenta(t):  return c(t, 35)
def b_green(t):  return c(t, 1, 32)
def b_cyan(t):   return c(t, 1, 36)
def b_yellow(t): return c(t, 1, 33)
def b_red(t):    return c(t, 1, 31)

def header(title: str):
    w = 62
    print()
    print(cyan("╔" + "═" * w + "╗"))
    print(cyan("║") + bold(f"  {title}".ljust(w)) + cyan("║"))
    print(cyan("╚" + "═" * w + "╝"))

def info(msg):    print(f"  {cyan('ℹ')} {msg}")
def ok(msg):      print(f"  {green('✓')} {b_green(msg)}")
def warn(msg):    print(f"  {yellow('⚠')} {yellow(msg)}")
def err(msg):     print(f"  {red('✗')} {b_red(msg)}")
def step(n, msg): print(f"\n  {bold(b_cyan(f'[{n}]'))} {bold(msg)}")
def rule():       print(dim("  " + "─" * 58))


# ══════════════════════════════════════════════════════════════════════════════
# STATE PERSISTENCE
# ══════════════════════════════════════════════════════════════════════════════

def save_state(data: dict):
    try:
        existing = load_state()
        existing.update(data)
        STATE_FILE.write_text(json.dumps(existing, indent=2))
    except Exception:
        pass

def load_state() -> dict:
    try:
        if STATE_FILE.exists():
            return json.loads(STATE_FILE.read_text())
    except Exception:
        pass
    return {}


# ══════════════════════════════════════════════════════════════════════════════
# BANNER
# ══════════════════════════════════════════════════════════════════════════════

def print_banner():
    env_label = {
        "kaggle": f"{b_cyan('Kaggle')} {'GPU ' + ENV['gpu_name'] if ENV['is_gpu'] else 'CPU'}",
        "colab":  f"{b_cyan('Google Colab')} {'GPU ' + ENV['gpu_name'] if ENV['is_gpu'] else 'CPU'}",
        "local":  f"{b_cyan('Local')} {'GPU ' + ENV['gpu_name'] if ENV['is_gpu'] else 'CPU'}",
    }[ENV["name"]]

    print()
    print(cyan("  ╔══════════════════════════════════════════════════════════╗"))
    print(cyan("  ║") + bold("   🧠  Cogito-0.9 API Manager                           ") + cyan("║"))
    print(cyan("  ║") + f"   Environment: {env_label}".ljust(62) + cyan("║"))
    print(cyan("  ╚══════════════════════════════════════════════════════════╝"))
    print()


# ══════════════════════════════════════════════════════════════════════════════
# DEPENDENCY INSTALLATION
# ══════════════════════════════════════════════════════════════════════════════

def run_pip(packages: list, extra_args: list = None, env_vars: dict = None):
    cmd = [sys.executable, "-m", "pip", "install", "-q"] + (extra_args or []) + packages
    run_env = os.environ.copy()
    if env_vars:
        run_env.update(env_vars)
    r = subprocess.run(cmd, capture_output=True, text=True, env=run_env)
    return r.returncode == 0

def install_deps():
    header("📦 Installing Dependencies")

    step(1, "Core packages (fastapi, uvicorn, huggingface_hub)")
    ok_1 = run_pip(["fastapi", "uvicorn[standard]", "python-multipart",
                    "huggingface_hub", "pydantic", "requests"])
    if ok_1:
        ok("Core packages installed")
    else:
        warn("Some core packages may have failed — continuing")

    step(2, "llama-cpp-python")
    if ENV["is_gpu"]:
        info("GPU detected → trying CUDA-enabled wheel...")
        # Try pre-built CUDA wheel (fastest)
        ok_llama = run_pip(
            ["llama-cpp-python"],
            extra_args=["--extra-index-url", "https://abetlen.github.io/llama-cpp-python/whl/cu121"],
        )
        if not ok_llama:
            info("Pre-built wheel unavailable → compiling from source (may take 3-5 min)...")
            ok_llama = run_pip(
                ["llama-cpp-python", "--force-reinstall", "--no-cache-dir"],
                env_vars={"CMAKE_ARGS": "-DGGML_CUDA=on"},
            )
    else:
        info("No GPU → installing CPU build...")
        ok_llama = run_pip(["llama-cpp-python"])

    if ok_llama:
        ok("llama-cpp-python installed")
    else:
        err("llama-cpp-python install failed")
        print("  Try manually: pip install llama-cpp-python")

    ok("Dependencies ready")


# ══════════════════════════════════════════════════════════════════════════════
# MODEL SELECTION & DOWNLOAD
# ══════════════════════════════════════════════════════════════════════════════

def choose_model(auto: Optional[str] = None) -> dict:
    """Interactive model picker. Returns model config dict."""
    if auto and auto in MODELS:
        return MODELS[auto]

    header("🎛️  Choose Your Model")
    print()
    for i, (key, m) in enumerate(MODELS.items(), 1):
        gpu_tag = green(" [GPU recommended]") if "q8" in key else ""
        print(f"  {bold(f'[{i}]')} {bold(m['description'])}")
        print(f"      Size: {cyan(m['size'])}{gpu_tag}")
        print()

    default = "1"
    while True:
        choice = input(f"  Pick model {dim(f'[1-{len(MODELS)}] (default={default})')}: ").strip() or default
        if choice.isdigit() and 1 <= int(choice) <= len(MODELS):
            key = list(MODELS.keys())[int(choice) - 1]
            info(f"Selected: {bold(MODELS[key]['description'])}")
            save_state({"model_key": key})
            return MODELS[key]
        warn("Invalid choice, try again")


def download_model(model: dict) -> Path:
    """Download model GGUF from HuggingFace. Returns path."""
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    dest = MODEL_DIR / model["file"]

    if dest.exists():
        size_gb = dest.stat().st_size / 1e9
        ok(f"Model already present: {dest.name} ({size_gb:.2f} GB)")
        return dest

    header("⬇️  Downloading Model")
    info(f"File : {model['file']}")
    info(f"Size : {model['size']}")
    info(f"Repo : {HF_REPO}")
    print()

    # Method 1: huggingface_hub
    try:
        from huggingface_hub import hf_hub_download
        info("Downloading via huggingface_hub...")
        path = hf_hub_download(
            repo_id=HF_REPO,
            filename=model["file"],
            local_dir=str(MODEL_DIR),
            local_dir_use_symlinks=False,
        )
        size_gb = Path(path).stat().st_size / 1e9
        ok(f"Downloaded: {Path(path).name} ({size_gb:.2f} GB)")
        return Path(path)
    except Exception as e:
        warn(f"huggingface_hub failed ({e}), trying wget...")

    # Method 2: wget / curl
    url = f"https://huggingface.co/{HF_REPO}/resolve/main/{model['file']}"
    downloader = shutil.which("wget") or shutil.which("curl")
    if downloader:
        if "wget" in downloader:
            cmd = ["wget", "-q", "--show-progress", "-O", str(dest), url]
        else:
            cmd = ["curl", "-L", "--progress-bar", "-o", str(dest), url]
        info(f"Downloading via {Path(downloader).name}...")
        r = subprocess.run(cmd)
        if r.returncode == 0 and dest.exists():
            ok(f"Downloaded via {Path(downloader).name}")
            return dest

    err("All download methods failed. Please download manually:")
    print(f"     {cyan(url)}")
    print(f"     Save to: {dest}")
    sys.exit(1)


# ══════════════════════════════════════════════════════════════════════════════
# API SERVER (embedded — written to disk on start)
# ══════════════════════════════════════════════════════════════════════════════

SERVER_CODE = r'''"""Cogito-0.9 FastAPI Server — auto-generated by cogito.py"""
import os, sys, json, time, uuid, secrets, logging, threading
from datetime import datetime
from typing import Optional, List, Dict, Union
from pathlib import Path
from fastapi import FastAPI, Request, HTTPException, Depends, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import JSONResponse, StreamingResponse, HTMLResponse
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel
import uvicorn

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("cogito")

MODEL_PATH     = os.environ["COGITO_MODEL_PATH"]
ADMIN_KEY      = os.environ["COGITO_ADMIN_KEY"]
KEYS_FILE      = os.environ.get("COGITO_KEYS_FILE", "/tmp/cogito_keys.json")
MAX_CTX        = int(os.environ.get("COGITO_CTX", "4096"))
N_GPU_LAYERS   = int(os.environ.get("COGITO_GPU_LAYERS", "-1"))
N_THREADS      = int(os.environ.get("COGITO_THREADS", "4"))
DEFAULT_TOKENS = int(os.environ.get("COGITO_MAX_TOKENS", "512"))
DEFAULT_RPM    = int(os.environ.get("COGITO_RPM", "30"))
MODEL_ID       = Path(MODEL_PATH).stem

llm = None
model_loaded   = False
model_loading  = False
start_time     = datetime.utcnow()

# ── Key Manager ───────────────────────────────────────────────────────────────
class KeyManager:
    def __init__(self):
        self.keys: Dict[str, dict] = {}
        self.rate: Dict[str, list] = {}
        self.lock = threading.Lock()
        self._load()
        self._ensure_admin()

    def _load(self):
        try:
            if Path(KEYS_FILE).exists():
                self.keys = json.loads(Path(KEYS_FILE).read_text())
                log.info(f"Loaded {len(self.keys)} API keys")
        except Exception: pass

    def _save(self):
        try: Path(KEYS_FILE).write_text(json.dumps(self.keys, indent=2, default=str))
        except Exception: pass

    def _ensure_admin(self):
        if not any(v.get("role") == "admin" for v in self.keys.values()):
            self.create(name="admin", role="admin", key_override=ADMIN_KEY)
            log.info(f"Admin key: {ADMIN_KEY}")

    def create(self, name, role="user", rpm=DEFAULT_RPM, key_override=None):
        with self.lock:
            key = key_override or f"cg-{secrets.token_urlsafe(32)}"
            record = {"key": key, "name": name, "role": role, "rpm": rpm,
                      "created": datetime.utcnow().isoformat(), "last_used": None,
                      "reqs": 0, "tokens": 0, "active": True}
            self.keys[key] = record
            self._save()
            return record

    def validate(self, key):
        with self.lock:
            r = self.keys.get(key)
            return r if r and r.get("active") else None

    def check_rate(self, key):
        now = time.time()
        with self.lock:
            r = self.keys.get(key)
            if not r: return False
            self.rate.setdefault(key, [])
            self.rate[key] = [t for t in self.rate[key] if now - t < 60]
            if len(self.rate[key]) >= r.get("rpm", DEFAULT_RPM): return False
            self.rate[key].append(now)
            return True

    def record(self, key, tokens=0):
        with self.lock:
            if key in self.keys:
                self.keys[key]["last_used"] = datetime.utcnow().isoformat()
                self.keys[key]["reqs"] = self.keys[key].get("reqs", 0) + 1
                self.keys[key]["tokens"] = self.keys[key].get("tokens", 0) + tokens
                self._save()

    def revoke(self, key):
        with self.lock:
            if key in self.keys:
                self.keys[key]["active"] = False
                self._save()
                return True
            return False

    def list(self, reveal=False):
        with self.lock:
            result = []
            for k, v in self.keys.items():
                entry = v.copy()
                if not reveal:
                    entry["key"] = k[:10] + "..." + k[-4:]
                result.append(entry)
            return result

km = KeyManager()
security = HTTPBearer(auto_error=False)

async def auth(creds: Optional[HTTPAuthorizationCredentials] = Depends(security),
               req: Request = None):
    token = (creds.credentials if creds else None) or (req.headers.get("x-api-key") if req else None)
    if not token:
        raise HTTPException(401, "Missing API key. Use: Authorization: Bearer cg-xxx")
    r = km.validate(token)
    if not r:
        raise HTTPException(401, "Invalid or revoked API key.")
    if not km.check_rate(token):
        raise HTTPException(429, f"Rate limit exceeded ({r['rpm']} req/min).")
    return r

async def admin_auth(key_data=Depends(auth)):
    if key_data.get("role") != "admin":
        raise HTTPException(403, "Admin access required.")
    return key_data

# ── Pydantic Models ───────────────────────────────────────────────────────────
class Msg(BaseModel):
    role: str
    content: str

class ChatReq(BaseModel):
    model: str = MODEL_ID
    messages: List[Msg]
    max_tokens: Optional[int] = DEFAULT_TOKENS
    temperature: Optional[float] = 0.7
    top_p: Optional[float] = 0.95
    top_k: Optional[int] = 40
    repeat_penalty: Optional[float] = 1.1
    stream: Optional[bool] = False
    stop: Optional[Union[str, List[str]]] = None

class CompReq(BaseModel):
    model: str = MODEL_ID
    prompt: str
    max_tokens: Optional[int] = DEFAULT_TOKENS
    temperature: Optional[float] = 0.7
    top_p: Optional[float] = 0.95
    stream: Optional[bool] = False
    stop: Optional[Union[str, List[str]]] = None

class KeyReq(BaseModel):
    name: str
    role: str = "user"
    rpm: int = DEFAULT_RPM

class RevokeReq(BaseModel):
    key: str

# ── App ───────────────────────────────────────────────────────────────────────
app = FastAPI(title="Cogito-0.9 API", version="1.0.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])
app.add_middleware(GZipMiddleware, minimum_size=1000)

def load_model():
    global llm, model_loaded, model_loading
    model_loading = True
    log.info(f"Loading model: {MODEL_PATH}")
    try:
        from llama_cpp import Llama
        llm = Llama(model_path=MODEL_PATH, n_ctx=MAX_CTX, n_gpu_layers=N_GPU_LAYERS,
                    n_threads=N_THREADS, verbose=False)
        model_loaded = True
        log.info("✅ Model loaded!")
    except Exception as e:
        log.error(f"❌ Model load failed: {e}")
    finally:
        model_loading = False

@app.on_event("startup")
async def startup():
    if Path(MODEL_PATH).exists():
        threading.Thread(target=load_model, daemon=True).start()
    else:
        log.warning(f"Model not found: {MODEL_PATH}")

def ts(): return int(time.time())

def build_prompt(messages):
    p = ""
    for m in messages:
        r = m.role.lower()
        if r == "system":    p += f"<|system|>\n{m.content}\n"
        elif r == "user":    p += f"<|user|>\n{m.content}\n"
        elif r == "assistant": p += f"<|assistant|>\n{m.content}\n"
    return p + "<|assistant|>\n"

def not_ready():
    if not model_loaded:
        raise HTTPException(503, "Model loading..." if model_loading else "Model unavailable.")

# ── Routes ────────────────────────────────────────────────────────────────────
@app.get("/", include_in_schema=False)
async def root(): return HTMLResponse(DASHBOARD)

@app.get("/health")
async def health():
    return {"ok": True, "model_loaded": model_loaded, "model_loading": model_loading,
            "model": MODEL_ID, "uptime": (datetime.utcnow()-start_time).total_seconds()}

@app.get("/ping")
async def ping(): return {"pong": True}

@app.get("/v1/models")
async def models(kd=Depends(auth)):
    return {"object":"list","data":[
        {"id": MODEL_ID, "object":"model","created":1700000000,"owned_by":"ozaa77"},
        {"id":"cogito-0.9","object":"model","created":1700000000,"owned_by":"ozaa77"},
    ]}

@app.post("/v1/chat/completions")
async def chat(body: ChatReq, req: Request, kd=Depends(auth)):
    not_ready()
    prompt = build_prompt(body.messages)
    stop = body.stop if isinstance(body.stop, list) else ([body.stop] if body.stop else ["<|user|>","<|system|>"])
    rid = f"chatcmpl-{uuid.uuid4().hex}"

    if body.stream:
        async def gen():
            tok = 0
            created = ts()
            yield f"data: {json.dumps({'id':rid,'object':'chat.completion.chunk','created':created,'model':body.model,'choices':[{'index':0,'delta':{'role':'assistant','content':''},'finish_reason':None}]})}\n\n"
            try:
                for chunk in llm(prompt,max_tokens=body.max_tokens,temperature=body.temperature,
                                  top_p=body.top_p,top_k=body.top_k,repeat_penalty=body.repeat_penalty,
                                  stop=stop,stream=True):
                    txt = chunk["choices"][0]["text"]; tok+=1
                    yield f"data: {json.dumps({'id':rid,'object':'chat.completion.chunk','created':created,'model':body.model,'choices':[{'index':0,'delta':{'content':txt},'finish_reason':None}]})}\n\n"
            except Exception as e: log.error(f"stream err: {e}")
            yield f"data: {json.dumps({'id':rid,'object':'chat.completion.chunk','created':ts(),'model':body.model,'choices':[{'index':0,'delta':{},'finish_reason':'stop'}]})}\n\n"
            yield "data: [DONE]\n\n"
            km.record(kd["key"], tok)
        return StreamingResponse(gen(), media_type="text/event-stream")

    out = llm(prompt,max_tokens=body.max_tokens,temperature=body.temperature,
              top_p=body.top_p,top_k=body.top_k,repeat_penalty=body.repeat_penalty,stop=stop)
    content = out["choices"][0]["text"].strip()
    u = out.get("usage",{}); total = u.get("total_tokens",0)
    km.record(kd["key"], total)
    return {"id":rid,"object":"chat.completion","created":ts(),"model":body.model,
            "choices":[{"index":0,"message":{"role":"assistant","content":content},"finish_reason":"stop"}],
            "usage":{"prompt_tokens":u.get("prompt_tokens",0),"completion_tokens":u.get("completion_tokens",0),"total_tokens":total}}

@app.post("/v1/completions")
async def complete(body: CompReq, req: Request, kd=Depends(auth)):
    not_ready()
    stop = body.stop if isinstance(body.stop,list) else ([body.stop] if body.stop else [])
    rid = f"cmpl-{uuid.uuid4().hex}"
    if body.stream:
        async def gen():
            tok=0; created=ts()
            try:
                for chunk in llm(body.prompt,max_tokens=body.max_tokens,temperature=body.temperature,
                                  top_p=body.top_p,stop=stop,stream=True):
                    txt=chunk["choices"][0]["text"]; tok+=1
                    yield f"data: {json.dumps({'id':rid,'object':'text_completion','created':created,'model':body.model,'choices':[{'text':txt,'index':0,'finish_reason':None}]})}\n\n"
            except Exception as e: log.error(f"stream err: {e}")
            yield "data: [DONE]\n\n"; km.record(kd["key"], tok)
        return StreamingResponse(gen(), media_type="text/event-stream")
    out = llm(body.prompt,max_tokens=body.max_tokens,temperature=body.temperature,top_p=body.top_p,stop=stop)
    u = out.get("usage",{}); km.record(kd["key"], u.get("total_tokens",0))
    return {"id":rid,"object":"text_completion","created":ts(),"model":body.model,
            "choices":[{"text":out["choices"][0]["text"],"index":0,"finish_reason":"stop"}],
            "usage":u}

# ── Admin Routes ──────────────────────────────────────────────────────────────
@app.post("/v1/admin/keys/create")
async def key_create(body: KeyReq, adm=Depends(admin_auth)):
    return {"success": True, "key": km.create(body.name, body.role, body.rpm)}

@app.get("/v1/admin/keys/list")
async def key_list(adm=Depends(admin_auth)):
    keys = km.list(reveal=True)
    return {"keys": keys, "count": len(keys)}

@app.post("/v1/admin/keys/revoke")
async def key_revoke(body: RevokeReq, adm=Depends(admin_auth)):
    return {"success": km.revoke(body.key)}

@app.get("/v1/admin/stats")
async def stats(adm=Depends(admin_auth)):
    keys = km.list()
    return {"uptime": (datetime.utcnow()-start_time).total_seconds(),
            "model_loaded": model_loaded, "model": MODEL_ID,
            "total_keys": len(keys),
            "active_keys": sum(1 for k in keys if k.get("active")),
            "total_requests": sum(k.get("reqs",0) for k in keys),
            "total_tokens": sum(k.get("tokens",0) for k in keys)}

DASHBOARD = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Cogito-0.9 API</title>
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;500&display=swap" rel="stylesheet">
<style>
:root{--bg:#07070f;--s1:#0f0f1a;--s2:#161624;--bd:#22223a;--acc:#7c6fff;--ac2:#00d4ff;--ac3:#ff6b9d;--tx:#e2e8f0;--muted:#6b7a99;--ok:#00e5a0;--warn:#ffb347;--err:#ff5252}
*{box-sizing:border-box;margin:0;padding:0}
body{background:var(--bg);color:var(--tx);font-family:'Inter',sans-serif;min-height:100vh}
body::before{content:'';position:fixed;inset:0;background:radial-gradient(ellipse at 15% 50%,rgba(124,111,255,.06),transparent 55%),radial-gradient(ellipse at 85% 20%,rgba(0,212,255,.05),transparent 55%);pointer-events:none}
.wrap{max-width:1100px;margin:0 auto;padding:0 20px;position:relative}
header{padding:18px 0;border-bottom:1px solid var(--bd);background:rgba(7,7,15,.85);backdrop-filter:blur(20px);position:sticky;top:0;z-index:100}
.hrow{display:flex;align-items:center;justify-content:space-between;gap:12px}
.logo{display:flex;align-items:center;gap:10px}
.licon{width:38px;height:38px;background:linear-gradient(135deg,var(--acc),var(--ac2));border-radius:9px;display:flex;align-items:center;justify-content:center;font-size:18px;box-shadow:0 0 24px rgba(124,111,255,.4);animation:glow 3s ease-in-out infinite}
@keyframes glow{0%,100%{box-shadow:0 0 20px rgba(124,111,255,.4)}50%{box-shadow:0 0 40px rgba(124,111,255,.7),0 0 60px rgba(0,212,255,.3)}}
.ltxt h1{font-size:17px;font-weight:700;background:linear-gradient(135deg,#fff,var(--ac2));-webkit-background-clip:text;-webkit-text-fill-color:transparent}
.ltxt p{font-size:11px;color:var(--muted)}
.badge{display:flex;align-items:center;gap:7px;padding:6px 14px;border-radius:999px;font-size:12px;font-weight:500;border:1px solid}
.online{background:rgba(0,229,160,.08);border-color:rgba(0,229,160,.3);color:var(--ok)}
.loading{background:rgba(255,179,71,.08);border-color:rgba(255,179,71,.3);color:var(--warn)}
.dot{width:7px;height:7px;border-radius:50%;animation:pulse 2s ease-in-out infinite}
.online .dot{background:var(--ok)}.loading .dot{background:var(--warn)}
@keyframes pulse{0%,100%{transform:scale(1);opacity:1}50%{transform:scale(1.4);opacity:.6}}
.hero{padding:52px 0 36px;text-align:center}
.hero h2{font-size:clamp(28px,5vw,50px);font-weight:800;line-height:1.1;margin-bottom:14px;background:linear-gradient(135deg,#fff 0%,var(--ac2) 50%,var(--acc) 100%);-webkit-background-clip:text;-webkit-text-fill-color:transparent}
.hero p{color:var(--muted);font-size:16px;max-width:540px;margin:0 auto 24px;line-height:1.6}
.tags{display:flex;gap:8px;justify-content:center;flex-wrap:wrap}
.tag{padding:5px 12px;border-radius:999px;font-size:11px;font-weight:600;text-transform:uppercase;letter-spacing:.5px;border:1px solid}
.tb{background:rgba(0,212,255,.08);border-color:rgba(0,212,255,.25);color:var(--ac2)}
.tp{background:rgba(124,111,255,.08);border-color:rgba(124,111,255,.25);color:var(--acc)}
.tk{background:rgba(255,107,157,.08);border-color:rgba(255,107,157,.25);color:var(--ac3)}
.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(220px,1fr));gap:16px;margin-bottom:36px}
.card{background:var(--s1);border:1px solid var(--bd);border-radius:11px;padding:20px;transition:all .25s;position:relative;overflow:hidden}
.card::before{content:'';position:absolute;top:0;left:0;right:0;height:2px;background:linear-gradient(90deg,var(--acc),var(--ac2));opacity:0;transition:opacity .25s}
.card:hover{border-color:rgba(124,111,255,.4);transform:translateY(-2px);box-shadow:0 0 28px rgba(124,111,255,.12)}
.card:hover::before{opacity:1}
.cico{font-size:24px;margin-bottom:10px}
.ctit{font-size:11px;color:var(--muted);font-weight:500;text-transform:uppercase;letter-spacing:.5px;margin-bottom:5px}
.cval{font-size:22px;font-weight:700}
.cval.sm{font-size:14px;word-break:break-all}
.csub{font-size:11px;color:var(--muted);margin-top:5px}
.sec{margin-bottom:36px}
.sec-title{font-size:18px;font-weight:700;margin-bottom:16px;display:flex;align-items:center;gap:10px}
.sec-title::after{content:'';flex:1;height:1px;background:var(--bd)}
.tabs{display:flex;gap:3px;margin-bottom:0;border-bottom:1px solid var(--bd)}
.tab{padding:9px 18px;border-radius:8px 8px 0 0;font-size:12px;font-weight:500;cursor:pointer;color:var(--muted);border:1px solid transparent;border-bottom:none;transition:all .2s;background:transparent;position:relative;bottom:-1px}
.tab.active{color:var(--tx);background:var(--s1);border-color:var(--bd)}
.tc{display:none}.tc.active{display:block}
pre{background:var(--s1);border:1px solid var(--bd);border-radius:11px;padding:18px;overflow-x:auto;font-family:'JetBrains Mono',monospace;font-size:12.5px;line-height:1.65;position:relative}
.cp{position:absolute;top:10px;right:10px;background:var(--s2);border:1px solid var(--bd);color:var(--muted);padding:4px 10px;border-radius:6px;font-size:11px;cursor:pointer;font-family:'Inter',sans-serif;transition:all .2s}
.cp:hover{background:var(--acc);color:#fff;border-color:var(--acc)}
.kw{color:#c678dd}.str{color:#98c379}.num{color:#d19a66}.cmt{color:#4b5263}.ac2c{color:var(--ac2)}
.etb{width:100%;border-collapse:separate;border-spacing:0}
.etb th{text-align:left;padding:11px 14px;font-size:10px;text-transform:uppercase;letter-spacing:.5px;color:var(--muted);background:var(--s1);border-bottom:1px solid var(--bd)}
.etb th:first-child{border-radius:8px 0 0 0}.etb th:last-child{border-radius:0 8px 0 0}
.etb td{padding:12px 14px;border-bottom:1px solid var(--bd);font-size:13px;background:var(--s1);transition:background .15s}
.etb tr:hover td{background:var(--s2)}.etb tr:last-child td{border-bottom:none}
.mt{padding:3px 9px;border-radius:5px;font-size:10px;font-weight:700;font-family:'JetBrains Mono',monospace;text-transform:uppercase}
.get{background:rgba(0,229,160,.12);color:var(--ok)}.post{background:rgba(124,111,255,.12);color:var(--acc)}.del{background:rgba(255,82,82,.12);color:var(--err)}
.path{font-family:'JetBrains Mono',monospace;font-size:12px;color:var(--ac2)}
.ab{padding:2px 7px;border-radius:4px;font-size:10px;font-weight:600}
.au{background:rgba(124,111,255,.12);color:var(--acc)}.aa{background:rgba(255,107,157,.12);color:var(--ac3)}.an{background:rgba(107,122,153,.12);color:var(--muted)}
footer{padding:36px 0;border-top:1px solid var(--bd);text-align:center;color:var(--muted);font-size:12px}
footer a{color:var(--acc);text-decoration:none}footer a:hover{text-decoration:underline}
@media(max-width:600px){.grid{grid-template-columns:1fr 1fr}.hrow{flex-direction:column;align-items:flex-start}}
</style>
</head>
<body>
<header><div class="wrap"><div class="hrow">
  <div class="logo">
    <div class="licon">🧠</div>
    <div class="ltxt"><h1>Cogito-0.9 API</h1><p>OpenAI-Compatible REST API</p></div>
  </div>
  <div id="sb" class="badge loading"><div class="dot"></div><span id="st">Checking…</span></div>
</div></div></header>

<main><div class="wrap">
<div class="hero">
  <h2>Cogito AI as an API</h2>
  <p>Fully OpenAI-compatible. Drop it into any app that uses the OpenAI SDK — zero changes needed.</p>
  <div class="tags">
    <span class="tag tp">OpenAI Compatible</span>
    <span class="tag tb">Streaming SSE</span>
    <span class="tag tk">API Key Auth</span>
    <span class="tag tb">Rate Limiting</span>
    <span class="tag tp">GGUF · llama.cpp</span>
  </div>
</div>

<div class="grid">
  <div class="card"><div class="cico">⚡</div><div class="ctit">Model</div><div class="cval" id="ms" style="font-size:16px">–</div><div class="csub" id="mn">–</div></div>
  <div class="card"><div class="cico">🌐</div><div class="ctit">Base URL</div><div class="cval sm" id="bu">–</div><div class="csub">Use with any OpenAI client</div></div>
  <div class="card"><div class="cico">🔑</div><div class="ctit">Auth</div><div class="cval sm">Bearer Token</div><div class="csub">Authorization: Bearer cg-xxx</div></div>
  <div class="card"><div class="cico">🕒</div><div class="ctit">Uptime</div><div class="cval sm" id="up">–</div><div class="csub" id="upt">–</div></div>
</div>

<div class="sec">
<div class="sec-title">⚡ Quick Start</div>
<div class="tabs">
  <button class="tab active" onclick="sw(event,'tcurl')">cURL</button>
  <button class="tab" onclick="sw(event,'tpy')">Python</button>
  <button class="tab" onclick="sw(event,'toai')">OpenAI SDK</button>
  <button class="tab" onclick="sw(event,'tjs')">JavaScript</button>
</div>
<div id="tcurl" class="tc active"><pre><button class="cp" onclick="cp(this)">Copy</button><span class="kw">curl</span> -X POST <span class="str">"BASE_URL/v1/chat/completions"</span> \
  -H <span class="str">"Authorization: Bearer cg-YOUR_KEY"</span> \
  -H <span class="str">"Content-Type: application/json"</span> \
  -d <span class="str">'{"model":"cogito-0.9-q4_k_m","messages":[{"role":"user","content":"Hello!"}],"max_tokens":200}'</span></pre></div>
<div id="tpy" class="tc"><pre><button class="cp" onclick="cp(this)">Copy</button><span class="kw">import</span> requests
r = requests.post(<span class="str">"BASE_URL/v1/chat/completions"</span>,
    headers={<span class="str">"Authorization"</span>: <span class="str">"Bearer cg-YOUR_KEY"</span>},
    json={<span class="str">"model"</span>:<span class="str">"cogito-0.9-q4_k_m"</span>,<span class="str">"messages"</span>:[{<span class="str">"role"</span>:<span class="str">"user"</span>,<span class="str">"content"</span>:<span class="str">"Hi!"</span>}],<span class="str">"max_tokens"</span>:<span class="num">200</span>})
<span class="kw">print</span>(r.json()[<span class="str">"choices"</span>][<span class="num">0</span>][<span class="str">"message"</span>][<span class="str">"content"</span>])</pre></div>
<div id="toai" class="tc"><pre><button class="cp" onclick="cp(this)">Copy</button><span class="kw">from</span> openai <span class="kw">import</span> OpenAI
client = OpenAI(base_url=<span class="str">"BASE_URL/v1"</span>, api_key=<span class="str">"cg-YOUR_KEY"</span>)
<span class="kw">for</span> chunk <span class="kw">in</span> client.chat.completions.create(
    model=<span class="str">"cogito-0.9-q4_k_m"</span>,
    messages=[{<span class="str">"role"</span>:<span class="str">"user"</span>,<span class="str">"content"</span>:<span class="str">"Explain AI in 3 sentences."</span>}],
    stream=<span class="kw">True</span>,
):
    <span class="kw">print</span>(chunk.choices[<span class="num">0</span>].delta.content <span class="kw">or</span> <span class="str">""</span>, end=<span class="str">""</span>, flush=<span class="kw">True</span>)</pre></div>
<div id="tjs" class="tc"><pre><button class="cp" onclick="cp(this)">Copy</button><span class="kw">const</span> r = <span class="kw">await</span> fetch(<span class="str">"BASE_URL/v1/chat/completions"</span>, {
  method: <span class="str">"POST"</span>,
  headers: {<span class="str">"Authorization"</span>:<span class="str">"Bearer cg-YOUR_KEY"</span>,<span class="str">"Content-Type"</span>:<span class="str">"application/json"</span>},
  body: JSON.stringify({model:<span class="str">"cogito-0.9-q4_k_m"</span>,messages:[{role:<span class="str">"user"</span>,content:<span class="str">"Hi!"</span>}],stream:<span class="kw">true</span>})
});</pre></div>
</div>

<div class="sec">
<div class="sec-title">📡 Endpoints</div>
<div style="overflow-x:auto;border-radius:10px;border:1px solid var(--bd)">
<table class="etb"><thead><tr><th>Method</th><th>Path</th><th>Description</th><th>Auth</th></tr></thead><tbody>
<tr><td><span class="mt get">GET</span></td><td><span class="path">/health</span></td><td style="color:var(--muted);font-size:12px">Health &amp; model status</td><td><span class="ab an">None</span></td></tr>
<tr><td><span class="mt get">GET</span></td><td><span class="path">/v1/models</span></td><td style="color:var(--muted);font-size:12px">List models</td><td><span class="ab au">API Key</span></td></tr>
<tr><td><span class="mt post">POST</span></td><td><span class="path">/v1/chat/completions</span></td><td style="color:var(--muted);font-size:12px">Chat (streaming ✓)</td><td><span class="ab au">API Key</span></td></tr>
<tr><td><span class="mt post">POST</span></td><td><span class="path">/v1/completions</span></td><td style="color:var(--muted);font-size:12px">Text completion (streaming ✓)</td><td><span class="ab au">API Key</span></td></tr>
<tr><td><span class="mt post">POST</span></td><td><span class="path">/v1/admin/keys/create</span></td><td style="color:var(--muted);font-size:12px">Create API key</td><td><span class="ab aa">Admin</span></td></tr>
<tr><td><span class="mt get">GET</span></td><td><span class="path">/v1/admin/keys/list</span></td><td style="color:var(--muted);font-size:12px">List all keys</td><td><span class="ab aa">Admin</span></td></tr>
<tr><td><span class="mt post">POST</span></td><td><span class="path">/v1/admin/keys/revoke</span></td><td style="color:var(--muted);font-size:12px">Revoke a key</td><td><span class="ab aa">Admin</span></td></tr>
<tr><td><span class="mt get">GET</span></td><td><span class="path">/v1/admin/stats</span></td><td style="color:var(--muted);font-size:12px">Usage statistics</td><td><span class="ab aa">Admin</span></td></tr>
<tr><td><span class="mt get">GET</span></td><td><span class="path">/docs</span></td><td style="color:var(--muted);font-size:12px">Swagger UI</td><td><span class="ab an">None</span></td></tr>
</tbody></table>
</div></div>
</div></main>

<footer><div class="wrap">
  <p>Cogito-0.9 API &bull; <a href="https://huggingface.co/ozaa77/Cogito-0.9" target="_blank">ozaa77/Cogito-0.9</a> &bull; <a href="/docs">Swagger UI</a> &bull; <a href="https://github.com/ggerganov/llama.cpp" target="_blank">llama.cpp</a></p>
</div></footer>

<script>
document.getElementById('bu').textContent = location.origin;
async function health(){
  try{
    const d=await(await fetch('/health')).json();
    const sb=document.getElementById('sb'),st=document.getElementById('st'),ms=document.getElementById('ms'),mn=document.getElementById('mn'),up=document.getElementById('up');
    if(d.model_loaded){sb.className='badge online';st.textContent='Online';ms.textContent='Ready ✓';ms.style.color='var(--ok)'}
    else if(d.model_loading){sb.className='badge loading';st.textContent='Loading model…';ms.textContent='Loading…';ms.style.color='var(--warn)'}
    else{sb.className='badge loading';st.textContent='Starting up'}
    mn.textContent=d.model||'–';
    const u=Math.floor(d.uptime||0),h=Math.floor(u/3600),m=Math.floor((u%3600)/60),s=u%60;
    up.textContent=`${h}h ${m}m ${s}s`;
  }catch(e){}
}
health();setInterval(health,5000);
function sw(e,id){document.querySelectorAll('.tab').forEach(t=>t.classList.remove('active'));document.querySelectorAll('.tc').forEach(t=>t.classList.remove('active'));e.target.classList.add('active');document.getElementById(id).classList.add('active')}
function cp(btn){const code=btn.parentElement.innerText.replace('Copy','').trim();navigator.clipboard.writeText(code).then(()=>{btn.textContent='Copied!';setTimeout(()=>btn.textContent='Copy',2000)})}
</script>
</body></html>"""

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", "8000"))
    uvicorn.run(app, host="0.0.0.0", port=port, log_level="info")
'''


# ══════════════════════════════════════════════════════════════════════════════
# TUNNEL — Cloudflare Quick Tunnel (FREE, NO AUTH, JUST WORKS)
# ══════════════════════════════════════════════════════════════════════════════

CLOUDFLARED_URLS = {
    "linux_amd64":  "https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64",
    "linux_arm64":  "https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-arm64",
    "darwin_amd64": "https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-darwin-amd64",
    "windows":      "https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-windows-amd64.exe",
}

def _cf_binary_path() -> Path:
    suffix = ".exe" if platform.system().lower() == "windows" else ""
    return Path("/tmp") / f"cloudflared{suffix}"

def _download_cloudflared() -> Path:
    dest = _cf_binary_path()
    if dest.exists():
        return dest
    system = platform.system().lower()
    arch = platform.machine().lower()
    if system == "windows":
        key = "windows"
    elif "arm" in arch or "aarch" in arch:
        key = "linux_arm64"
    elif system == "darwin":
        key = "darwin_amd64"
    else:
        key = "linux_amd64"
    url = CLOUDFLARED_URLS[key]
    info(f"Downloading cloudflared ({key})...")
    urllib.request.urlretrieve(url, str(dest))
    if system != "windows":
        os.chmod(str(dest), 0o755)
    ok("cloudflared ready")
    return dest

def start_tunnel(port: int) -> tuple[Optional[subprocess.Popen], Optional[str]]:
    """
    Start Cloudflare Quick Tunnel — completely free, no account, no token.
    Returns (process, public_url) or (None, None) on failure.
    """
    try:
        cf = _download_cloudflared()
        proc = subprocess.Popen(
            [str(cf), "tunnel", "--url", f"http://localhost:{port}", "--no-autoupdate"],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        deadline = time.time() + 35
        while time.time() < deadline:
            line = proc.stdout.readline()
            if not line:
                break
            if "trycloudflare.com" in line or ".cfargotunnel.com" in line:
                for part in line.split():
                    if part.startswith("https://") and ("trycloudflare" in part or "cfargotunnel" in part):
                        return proc, part.strip()
        err("cloudflared: could not parse tunnel URL from output")
        proc.terminate()
        return None, None
    except Exception as e:
        err(f"cloudflared failed: {e}")
        return None, None


# ══════════════════════════════════════════════════════════════════════════════
# KEEPALIVE
# ══════════════════════════════════════════════════════════════════════════════

def start_keepalive(port: int):
    """Prevents Kaggle/Colab from killing the session via idle timeout."""
    def _loop():
        while True:
            try:
                time.sleep(50)
                # CPU burst
                _ = sum(i * i for i in range(300_000))
                # HTTP ping
                urllib.request.urlopen(f"http://localhost:{port}/ping", timeout=5)
            except Exception:
                pass
    t = threading.Thread(target=_loop, daemon=True)
    t.start()
    info("KeepAlive active (prevents idle shutdown)")


# ══════════════════════════════════════════════════════════════════════════════
# SERVER MANAGEMENT
# ══════════════════════════════════════════════════════════════════════════════

_server_proc: Optional[subprocess.Popen] = None
_tunnel_proc: Optional[subprocess.Popen] = None

def wait_for_port(port: int, timeout: float = 60.0) -> bool:
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            with socket.create_connection(("127.0.0.1", port), timeout=1):
                return True
        except (ConnectionRefusedError, OSError):
            time.sleep(0.5)
    return False

def write_server_file() -> Path:
    dest = WORK_DIR / "_cogito_server.py"
    dest.write_text(SERVER_CODE)
    return dest

def start_server(model_path: Path, admin_key: str, model_cfg: dict) -> bool:
    global _server_proc
    server_file = write_server_file()
    env = os.environ.copy()
    env.update({
        "COGITO_MODEL_PATH": str(model_path),
        "COGITO_ADMIN_KEY": admin_key,
        "COGITO_KEYS_FILE": str(KEYS_FILE),
        "PORT": str(PORT),
        "COGITO_CTX":       os.environ.get("COGITO_CTX", "4096"),
        "COGITO_GPU_LAYERS": str(model_cfg.get("gpu_layers_default", -1)),
        "COGITO_THREADS":   os.environ.get("COGITO_THREADS", "4"),
        "COGITO_MAX_TOKENS":os.environ.get("COGITO_MAX_TOKENS", "512"),
        "COGITO_RPM":       os.environ.get("COGITO_RPM", "30"),
    })
    _server_proc = subprocess.Popen(
        [sys.executable, str(server_file)],
        env=env,
        stdout=open(SERVER_LOG, "w"),
        stderr=subprocess.STDOUT,
    )
    info(f"Server starting (PID {_server_proc.pid})...")
    return wait_for_port(PORT, timeout=60)


# ══════════════════════════════════════════════════════════════════════════════
# API KEY HELPERS (via HTTP)
# ══════════════════════════════════════════════════════════════════════════════

def api(method: str, path: str, admin_key: str, data: dict = None) -> Optional[dict]:
    url = f"http://localhost:{PORT}{path}"
    req = urllib.request.Request(url, method=method.upper())
    req.add_header("Authorization", f"Bearer {admin_key}")
    req.add_header("Content-Type", "application/json")
    if data:
        req.data = json.dumps(data).encode()
    try:
        with urllib.request.urlopen(req, timeout=10) as r:
            return json.loads(r.read())
    except urllib.error.HTTPError as e:
        body = e.read().decode()
        err(f"HTTP {e.code}: {body[:200]}")
        return None
    except Exception as e:
        err(str(e))
        return None


# ══════════════════════════════════════════════════════════════════════════════
# CLI COMMANDS
# ══════════════════════════════════════════════════════════════════════════════

def cmd_setup(args: list = None):
    """Install dependencies and download the model."""
    print_banner()
    header("🛠️  Setup")

    # 1. Install deps
    install_deps()

    # 2. Choose + download model
    auto_model = (args[0] if args else None)
    model_cfg = choose_model(auto=auto_model)
    model_path = download_model(model_cfg)
    save_state({"model_path": str(model_path), "model_key": list(
        k for k, v in MODELS.items() if v["file"] == model_cfg["file"]
    )[0]})
    print()
    ok(f"Setup complete! Run: {bold('python cogito.py start')}")


def cmd_start(args: list = None):
    """Start server + tunnel. Blocks until Ctrl-C."""
    global _tunnel_proc
    print_banner()
    state = load_state()

    # Resolve model
    model_key  = state.get("model_key") or (args[0] if args else None)
    model_path_str = state.get("model_path")

    if not model_key and not model_path_str:
        warn("No model configured. Running setup first...")
        cmd_setup()
        state = load_state()
        model_key = state.get("model_key")
        model_path_str = state.get("model_path")

    if model_key and model_key in MODELS:
        model_cfg = MODELS[model_key]
    else:
        model_cfg = list(MODELS.values())[0]

    if model_path_str:
        model_path = Path(model_path_str)
    else:
        model_path = MODEL_DIR / model_cfg["file"]

    if not model_path.exists():
        warn(f"Model not found at {model_path}")
        model_path = download_model(model_cfg)

    # Admin key
    admin_key = state.get("admin_key") or secrets.token_urlsafe(32)
    save_state({"admin_key": admin_key})

    header("🚀 Starting API Server")
    info(f"Model : {model_path.name}")
    info(f"Port  : {PORT}")
    info(f"GPU   : {'Yes — ' + ENV['gpu_name'] if ENV['is_gpu'] else 'No (CPU mode)'}")

    step(1, "Starting FastAPI server...")
    ready = start_server(model_path, admin_key, model_cfg)
    if ready:
        ok(f"Server listening on port {PORT}")
    else:
        err("Server failed to start. Check logs:")
        print(f"     {SERVER_LOG}")
        try:
            print(SERVER_LOG.read_text()[-1500:])
        except Exception:
            pass
        return

    step(2, "Starting Cloudflare Quick Tunnel (free, no account)...")
    _tunnel_proc, public_url = start_tunnel(PORT)

    if public_url:
        save_state({"public_url": public_url})
        ok(f"Tunnel active!")
    else:
        warn("Tunnel failed. API is available locally only.")
        public_url = f"http://localhost:{PORT}"

    step(3, "Starting keepalive...")
    start_keepalive(PORT)

    # Print the big box
    print()
    bw = 62
    def bline(content=""): print(cyan("  ║") + f" {content}".ljust(bw) + cyan("║"))
    print(cyan("  ╔" + "═"*bw + "╗"))
    bline(bold("🎉  Cogito-0.9 API is LIVE!"))
    bline()
    bline(f"🌐  URL:       {b_cyan(public_url)}")
    bline(f"🔑  Admin key: {yellow(admin_key)}")
    bline(f"📖  Docs:      {blue(public_url + '/docs')}")
    bline(f"📊  Dashboard: {blue(public_url + '/')}")
    bline()
    bline(dim("The model is loading in background (~1-3 min)."))
    bline(dim("Use: python cogito.py keys   ← create user keys"))
    bline(dim("Use: python cogito.py test   ← test the API"))
    bline(dim("Press Ctrl-C to stop."))
    print(cyan("  ╚" + "═"*bw + "╝"))
    print()

    # Block + monitor
    try:
        while True:
            time.sleep(15)
            # Check server still alive
            if _server_proc and _server_proc.poll() is not None:
                warn("Server died! Restarting...")
                start_server(model_path, admin_key, model_cfg)
    except KeyboardInterrupt:
        print()
        info("Shutting down...")
        if _server_proc:  _server_proc.terminate()
        if _tunnel_proc:  _tunnel_proc.terminate()
        ok("Stopped.")


def cmd_keys(args: list = None):
    """Interactive API key management."""
    print_banner()
    state = load_state()
    admin_key = state.get("admin_key")
    public_url = state.get("public_url", f"http://localhost:{PORT}")

    if not admin_key:
        err("No admin key found. Run: python cogito.py start  first.")
        return

    header("🔑  API Key Manager")

    def do_list():
        data = api("GET", "/v1/admin/keys/list", admin_key)
        if not data:
            return
        keys = data.get("keys", [])
        rule()
        print(f"  {'NAME':<20} {'ROLE':<8} {'RPM':<6} {'REQS':<7} {'ACTIVE':<7} {'KEY'}")
        rule()
        for k in keys:
            active_str = green("yes") if k.get("active") else red("no")
            key_val = k.get("key", "")
            print(f"  {k.get('name',''):<20} {k.get('role',''):<8} {k.get('rpm',0):<6} {k.get('reqs',0):<7} {active_str:<18} {cyan(key_val)}")
        rule()
        print(f"  Total: {bold(str(len(keys)))} keys")

    def do_create():
        print()
        name = input(f"  Key name {dim('(e.g. my-app)')}: ").strip()
        if not name:
            warn("Name is required"); return
        role_in = input(f"  Role {dim('[user/admin] (default=user)')}: ").strip() or "user"
        rpm_in  = input(f"  Rate limit {dim('[req/min] (default=30)')}: ").strip() or "30"
        try:
            rpm = int(rpm_in)
        except ValueError:
            rpm = 30
        data = api("POST", "/v1/admin/keys/create", admin_key,
                   {"name": name, "role": role_in, "rpm": rpm})
        if data and data.get("success"):
            new_key = data["key"]["key"]
            print()
            ok(f"Key created!")
            print(f"  {bold('Name:')} {name}")
            print(f"  {bold('Role:')} {role_in}")
            print(f"  {bold('Key: ')} {b_cyan(new_key)}")
            print()
            print(f"  Use it:")
            print(f"  {dim('curl -H \"Authorization: Bearer ' + new_key + '\" ' + public_url + '/v1/models')}")

    def do_revoke():
        print()
        key_val = input(f"  Key to revoke {dim('(paste full key)')}: ").strip()
        if not key_val:
            warn("Key is required"); return
        data = api("POST", "/v1/admin/keys/revoke", admin_key, {"key": key_val})
        if data:
            if data.get("success"):
                ok("Key revoked.")
            else:
                warn("Key not found.")

    while True:
        print()
        print(f"  {bold('[1]')} List all keys")
        print(f"  {bold('[2]')} Create new key")
        print(f"  {bold('[3]')} Revoke a key")
        print(f"  {bold('[0]')} Back / Exit")
        print()
        choice = input("  Choice: ").strip()
        if choice == "1":   do_list()
        elif choice == "2": do_create()
        elif choice == "3": do_revoke()
        elif choice == "0": break
        else: warn("Invalid choice")


def cmd_test(args: list = None):
    """Run a test completion against the running server."""
    print_banner()
    state = load_state()
    admin_key = state.get("admin_key")
    public_url = state.get("public_url", f"http://localhost:{PORT}")

    if not admin_key:
        err("No admin key found. Run: python cogito.py start  first.")
        return

    header("🧪  API Test")

    step(1, "Health check...")
    data = api("GET", "/health", admin_key)
    if not data:
        err("Server not responding. Is it running?")
        info(f"Run: python cogito.py start")
        return

    info(f"Server up | model_loaded={green(str(data['model_loaded']))} | uptime={data.get('uptime',0):.0f}s")

    if not data.get("model_loaded"):
        warn("Model is still loading. Waiting up to 3 minutes...")
        for i in range(180):
            time.sleep(2)
            d = api("GET", "/health", admin_key)
            if d and d.get("model_loaded"):
                ok("Model ready!")
                break
            print(f"  [{i*2}s] Loading...", end="\r")
        else:
            err("Model did not load in time.")
            return

    step(2, "Chat completion test...")
    prompt = input(f"\n  Enter a prompt {dim('(or press Enter for default)')}: ").strip()
    if not prompt:
        prompt = "In exactly one sentence, what is your name and what can you do?"

    print()
    info(f"Prompt: {dim(prompt)}")
    print()
    print(f"  {cyan('🤖 Cogito says:')}")
    print(f"  ", end="", flush=True)

    # Streaming test
    url = f"http://localhost:{PORT}/v1/chat/completions"
    req = urllib.request.Request(url, method="POST")
    req.add_header("Authorization", f"Bearer {admin_key}")
    req.add_header("Content-Type", "application/json")
    req.data = json.dumps({
        "model": state.get("model_key", "cogito-0.9-q4_k_m"),
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": 200,
        "temperature": 0.7,
        "stream": True,
    }).encode()

    try:
        with urllib.request.urlopen(req, timeout=120) as r:
            for line in r:
                line = line.decode().strip()
                if line.startswith("data: ") and line != "data: [DONE]":
                    try:
                        chunk = json.loads(line[6:])
                        token = chunk["choices"][0]["delta"].get("content", "")
                        print(token, end="", flush=True)
                    except Exception:
                        pass
        print("\n")
        ok("Test passed!")
    except Exception as e:
        print()
        err(f"Streaming test failed: {e}")

    step(3, "Summary")
    info(f"API URL:    {b_cyan(public_url)}")
    info(f"Admin key:  {yellow(admin_key)}")
    info(f"Docs:       {blue(public_url + '/docs')}")


def cmd_status(args: list = None):
    """Show current server status, URL, and keys."""
    print_banner()
    state = load_state()
    admin_key  = state.get("admin_key")
    public_url = state.get("public_url")
    model_key  = state.get("model_key")

    header("📊  Status")
    print()

    if public_url:
        print(f"  {'Public URL:':<15} {b_cyan(public_url)}")
    else:
        print(f"  {'Public URL:':<15} {yellow('Not started')}")

    if admin_key:
        print(f"  {'Admin Key:':<15} {yellow(admin_key)}")
    else:
        print(f"  {'Admin Key:':<15} {dim('Not set — run: python cogito.py start')}")

    if model_key:
        m = MODELS.get(model_key, {})
        print(f"  {'Model:':<15} {cyan(m.get('file', model_key))}")

    print()

    # Live health
    if admin_key:
        data = api("GET", "/health", admin_key)
        if data:
            loaded = data.get("model_loaded", False)
            uptime = data.get("uptime", 0)
            h, m_, s = int(uptime)//3600, (int(uptime)%3600)//60, int(uptime)%60
            print(f"  {'Server:':<15} {green('Running')}")
            print(f"  {'Model:':<15} {green('Loaded ✓') if loaded else yellow('Loading...')}")
            print(f"  {'Uptime:':<15} {h}h {m_}m {s}s")
            print()
            stats = api("GET", "/v1/admin/stats", admin_key)
            if stats:
                print(f"  {'Total keys:':<15} {stats.get('total_keys',0)}")
                print(f"  {'Active keys:':<15} {stats.get('active_keys',0)}")
                print(f"  {'Total reqs:':<15} {stats.get('total_requests',0)}")
                print(f"  {'Total tokens:':<15} {stats.get('total_tokens',0)}")
        else:
            print(f"  {'Server:':<15} {red('Not running')}")
            print(f"  Run: {bold('python cogito.py start')}")


def cmd_menu():
    """Interactive main menu."""
    print_banner()
    while True:
        print(f"  {bold('What would you like to do?')}")
        print()
        print(f"  {bold('[1]')} {green('Setup')}       — install deps + download model")
        print(f"  {bold('[2]')} {cyan('Start')}       — start API server + public tunnel")
        print(f"  {bold('[3]')} {yellow('Keys')}        — create / list / revoke API keys")
        print(f"  {bold('[4]')} {magenta('Test')}        — send a test prompt to the API")
        print(f"  {bold('[5]')} {blue('Status')}      — show URL, keys, health")
        print(f"  {bold('[0]')} Exit")
        print()
        choice = input("  > ").strip()
        print()
        if   choice == "1": cmd_setup()
        elif choice == "2": cmd_start()
        elif choice == "3": cmd_keys()
        elif choice == "4": cmd_test()
        elif choice == "5": cmd_status()
        elif choice == "0": break
        else: warn("Invalid choice")
        print()


# ══════════════════════════════════════════════════════════════════════════════
# ENTRY POINT
# ══════════════════════════════════════════════════════════════════════════════

COMMANDS = {
    "setup":  (cmd_setup,  "Install deps + download model"),
    "start":  (cmd_start,  "Start server + free tunnel"),
    "keys":   (cmd_keys,   "Manage API keys"),
    "test":   (cmd_test,   "Test the API with a prompt"),
    "status": (cmd_status, "Show status & current URL"),
    "help":   (None,       "Show this message"),
}

def print_help():
    print_banner()
    print(f"  {bold('Usage:')}  python cogito.py [command] [args]")
    print()
    print(f"  {bold('Commands:')}")
    for cmd, (fn, desc) in COMMANDS.items():
        print(f"    {cyan(cmd):<12} {desc}")
    print()
    print(f"  {bold('Examples:')}")
    print(f"    python cogito.py              {dim('# interactive menu')}")
    print(f"    python cogito.py setup q4_k_m {dim('# setup with specific model')}")
    print(f"    python cogito.py start        {dim('# start everything')}")
    print(f"    python cogito.py keys         {dim('# manage keys')}")
    print(f"    python cogito.py test         {dim('# test the API')}")
    print()

if __name__ == "__main__":
    args = sys.argv[1:]
    if not args:
        cmd_menu()
    elif args[0] == "help" or args[0] == "--help" or args[0] == "-h":
        print_help()
    elif args[0] in COMMANDS:
        fn, _ = COMMANDS[args[0]]
        if fn:
            fn(args[1:])
    else:
        err(f"Unknown command: {args[0]}")
        print_help()
        sys.exit(1)
