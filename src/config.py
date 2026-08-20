"""
Configuration and Environment Detection for Cogito
"""

import os
import sys
import secrets
import subprocess
from pathlib import Path
from dataclasses import dataclass
from typing import Dict, Any, Optional

def detect_env() -> Dict[str, Any]:
    """Detect runtime platform (Kaggle, Colab, Local) and GPU availability."""
    env = {
        "name": "local",
        "is_kaggle": False,
        "is_colab": False,
        "is_gpu": False,
        "gpu_name": None,
        "gpu_count": 0,
        "work_dir": str(Path.cwd()),
        "model_dir": str(Path.cwd() / "models"),
    }

    if os.path.exists("/kaggle"):
        env["name"] = "kaggle"
        env["is_kaggle"] = True
        env["work_dir"] = "/kaggle/working"
        env["model_dir"] = "/kaggle/working/models"
    elif os.path.exists("/content") and ("COLAB_RELEASE_TAG" in os.environ or "COLAB_GPU" in os.environ or os.path.exists("/env/python")):
        env["name"] = "colab"
        env["is_colab"] = True
        env["work_dir"] = "/content"
        env["model_dir"] = "/content/models"

    try:
        r = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=5,
        )
        if r.returncode == 0 and r.stdout.strip():
            lines = [line.strip() for line in r.stdout.strip().split("\n") if line.strip()]
            env["is_gpu"] = True
            env["gpu_count"] = len(lines)
            env["gpu_name"] = lines[0].split(",")[0].strip()
    except Exception:
        pass

    return env


ENV = detect_env()
WORK_DIR = Path(ENV["work_dir"])
MODEL_DIR = Path(ENV["model_dir"])
KEYS_FILE = WORK_DIR / "cogito_keys.json"
SERVER_LOG = WORK_DIR / "cogito_server.log"
STATE_FILE = WORK_DIR / ".cogito_state.json"

PORT = int(os.environ.get("PORT", os.environ.get("COGITO_PORT", "8000")))
QUIET = os.environ.get("COGITO_QUIET", "").lower() in ("1", "true", "yes")

HF_REPO = "ozaa77/Cogito-0.9.1-15B"
MODEL_NAME = "Cogito-0.9.1-15B"
MODEL_ID = "Cogito-0.9.1-15B"

MODELS: Dict[str, Dict[str, str]] = {
    "auto": {
        "name": "Cogito-0.9.1-15B-4bit",
        "file": "model.safetensors.index.json",
        "dir": "Cogito-0.9.1-15B",
        "size": "~8.85 GB (4-bit NF4)",
        "description": "Cogito-0.9.1-15B 4-bit NF4 (Optimal for 12GB-16GB GPUs like T4/RTX 3060/4070)",
        "quant": "4bit",
    },
    "4bit": {
        "name": "Cogito-0.9.1-15B-4bit",
        "file": "model.safetensors.index.json",
        "dir": "Cogito-0.9.1-15B",
        "size": "~8.85 GB VRAM (4-bit NF4)",
        "description": "Cogito-0.9.1-15B 4-bit NF4 (Fits single 12-16GB GPU)",
        "quant": "4bit",
    },
    "8bit": {
        "name": "Cogito-0.9.1-15B-8bit",
        "file": "model.safetensors.index.json",
        "dir": "Cogito-0.9.1-15B",
        "size": "~16.10 GB VRAM (8-bit)",
        "description": "Cogito-0.9.1-15B 8-bit (Higher precision, 24GB GPU)",
        "quant": "8bit",
    },
    "16bit": {
        "name": "Cogito-0.9.1-15B-16bit",
        "file": "model.safetensors.index.json",
        "dir": "Cogito-0.9.1-15B",
        "size": "~30.80 GB VRAM (bfloat16)",
        "description": "Cogito-0.9.1-15B Full Precision (Multi-GPU or High-VRAM)",
        "quant": "16bit",
    },
    "q4_k_m": {
        "name": "Cogito-0.9.1-15B-Q4_K_M",
        "file": "cogito-0.9.1-15b-q4_k_m.gguf",
        "dir": "cogito-0.9.1-15b-q4_k_m.gguf",
        "size": "~8.85 GB (GGUF Q4_K_M)",
        "description": "Cogito-0.9.1-15B GGUF Q4_K_M (llama.cpp format)",
        "quant": "q4_k_m",
    },
}


@dataclass
class Settings:
    """Typed runtime settings parsed from environment variables."""
    model_path: str = os.environ.get("COGITO_MODEL_PATH", os.environ.get("MODEL_PATH", str(MODEL_DIR / "Cogito-0.9.1-15B")))
    admin_key: str = os.environ.get("COGITO_ADMIN_KEY", os.environ.get("ADMIN_KEY", secrets.token_urlsafe(32)))
    keys_file: str = os.environ.get("COGITO_KEYS_FILE", os.environ.get("API_KEYS_FILE", str(KEYS_FILE)))
    quant_mode: str = os.environ.get("COGITO_QUANT", os.environ.get("QUANT_MODE", "q4_k_m")).lower()
    max_context: int = int(os.environ.get("COGITO_CTX", os.environ.get("MAX_CONTEXT", "32768")))
    default_tokens: int = int(os.environ.get("COGITO_MAX_TOKENS", os.environ.get("MAX_TOKENS_DEFAULT", "2048")))
    default_temperature: float = float(os.environ.get("COGITO_TEMPERATURE", "0.70"))
    default_top_p: float = float(os.environ.get("COGITO_TOP_P", "0.90"))
    default_min_p: float = float(os.environ.get("COGITO_MIN_P", "0.05"))
    default_top_k: int = int(os.environ.get("COGITO_TOP_K", "40"))
    default_repetition_penalty: float = float(os.environ.get("COGITO_REPEAT_PENALTY", "1.08"))
    n_gpu_layers: int = int(os.environ.get("COGITO_N_GPU_LAYERS", "-1"))
    flash_attn: bool = os.environ.get("COGITO_FLASH_ATTN", "1").lower() in ("1", "true", "yes")
    default_rpm: int = int(os.environ.get("COGITO_RPM", os.environ.get("RATE_LIMIT_RPM", "30")))
    sse_heartbeat_secs: float = float(os.environ.get("COGITO_SSE_HEARTBEAT", os.environ.get("SSE_HEARTBEAT_SECS", "5.0")))
    trust_remote_code: bool = os.environ.get("COGITO_TRUST_REMOTE_CODE", "1").lower() in ("1", "true", "yes")
    port: int = PORT
    quiet: bool = QUIET

    def __post_init__(self):
        if not self.admin_key.startswith("cg-"):
            self.admin_key = f"cg-{self.admin_key}"

settings = Settings()
