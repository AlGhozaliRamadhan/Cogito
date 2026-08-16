"""
Cogito CLI Management Interface
"""

import os
import sys
import json
import time
import secrets
import subprocess
import urllib.request
from pathlib import Path
from typing import Optional, Dict, Any

import src.config as config
from src.config import (
    ENV,
    WORK_DIR,
    MODEL_DIR,
    KEYS_FILE,
    SERVER_LOG,
    PORT,
    QUIET,
    HF_REPO,
    MODEL_NAME,
    MODELS,
    settings,
)
from src.tunnel.cloudflare import start_tunnel, download_cloudflared
from src.supervisor.watchdog import (
    start_keepalive,
    is_server_healthy,
    public_health_ok,
    wait_for_port,
)

_server_proc: Optional[subprocess.Popen] = None
_tunnel_proc: Optional[subprocess.Popen] = None

def _get_attr(name: str, fallback: Any) -> Any:
    mod = sys.modules.get("cogito", sys.modules[__name__])
    return getattr(mod, name, fallback)

def header(title: str):    print(f"\n--- {title.upper()} ---")
def info(msg: str):       print(f"[INFO] {msg}")
def ok(msg: str):         print(f"[OK]   {msg}")
def warn(msg: str):       print(f"[WARN] {msg}")
def err(msg: str):        print(f"[ERR]  {msg}")
def step(n: int, msg: str): print(f"\n[{n}] {msg}")
def rule():               print("-" * 60)

def print_banner():
    gpu_info = f"GPU ({ENV['gpu_count']}x {ENV['gpu_name']})" if ENV['is_gpu'] else "CPU"
    env_label = f"{ENV['name']} - {gpu_info}"
    print("\n============================================================")
    print(f" {MODEL_NAME} API Manager (Safetensors)")
    print(f" Environment: {env_label}")
    print("============================================================\n")

def save_state(data: Dict[str, Any]):
    try:
        existing = load_state()
        existing.update(data)
        state_file = Path(config.STATE_FILE)
        state_file.parent.mkdir(parents=True, exist_ok=True)
        temp_state = state_file.with_suffix(f".tmp.{secrets.token_hex(4)}")
        temp_state.write_text(json.dumps(existing, indent=2), encoding="utf-8")
        temp_state.replace(state_file)
    except Exception as e:
        warn(f"Failed to save state: {e}")

def load_state() -> Dict[str, Any]:
    try:
        state_file = Path(config.STATE_FILE)
        if state_file.exists():
            content = state_file.read_text(encoding="utf-8").strip()
            if content:
                return json.loads(content)
    except Exception:
        pass
    return {}

def run_pip(packages: list, extra_args: list = None) -> bool:
    cmd = [sys.executable, "-m", "pip", "install"] + (extra_args or []) + packages
    r = subprocess.run(cmd)
    return r.returncode == 0

def install_deps():
    header("Installing Dependencies")
    step(1, "Core server packages (fastapi, uvicorn, huggingface_hub, etc.)")
    run_pip(["fastapi>=0.111.0", "uvicorn[standard]>=0.29.0", "python-multipart>=0.0.9", "huggingface_hub>=0.23.0", "pydantic>=2.0.0", "requests>=2.31.0"])

    step(2, "Inference engine (transformers, accelerate, safetensors, bitsandbytes, torch)")
    packages = ["transformers>=4.40.0", "accelerate>=0.28.0", "safetensors>=0.4.0", "sentencepiece>=0.2.0", "tiktoken>=0.7.0"]
    if ENV["is_gpu"]:
        packages.append("bitsandbytes>=0.43.0")

    run_pip(packages)
    ok("Dependencies ready")

def choose_model(auto: Optional[str] = None) -> Dict[str, str]:
    key = auto if auto and auto in MODELS else "auto"
    info(f"Selected profile: {MODELS[key]['description']}")
    save_state({"model_key": key})
    return MODELS[key]

def download_model(model_cfg: Dict[str, str]) -> Path:
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    dest_dir = MODEL_DIR / model_cfg.get("dir", "Cogito-0.9.1-15B")

    if dest_dir.exists():
        safetensor_files = list(dest_dir.glob("*.safetensors"))
        config_file = dest_dir / "config.json"
        if config_file.exists() and len(safetensor_files) > 0:
            total_size_gb = sum(f.stat().st_size for f in dest_dir.rglob("*") if f.is_file()) / 1e9
            ok(f"Model already present: {dest_dir.name} ({total_size_gb:.2f} GB across {len(safetensor_files)} shards)")
            return dest_dir

    header(f"Downloading {MODEL_NAME} Safetensors Model")
    try:
        from huggingface_hub import snapshot_download
        kwargs = {
            "repo_id": HF_REPO,
            "local_dir": str(dest_dir),
            "local_dir_use_symlinks": False,
        }
        hf_token = os.environ.get("HF_TOKEN")
        if hf_token:
            kwargs["token"] = hf_token

        path = snapshot_download(**kwargs)
        total_size_gb = sum(f.stat().st_size for f in Path(path).rglob("*") if f.is_file()) / 1e9
        ok(f"Downloaded model snapshot to: {Path(path).name} ({total_size_gb:.2f} GB)")
        return Path(path)
    except Exception as e:
        err(f"huggingface snapshot_download failed: {e}")
        sys.exit(1)

def start_server(model_path: Path, admin_key: str, model_cfg: Dict[str, str]) -> bool:
    global _server_proc
    
    server_script = Path(__file__).resolve().parent / "server" / "app.py"

    cog = sys.modules.get("cogito")
    current_proc = getattr(cog, "_server_proc", _server_proc) if cog else _server_proc

    if current_proc is not None:
        try:
            if current_proc.poll() is None:
                current_proc.terminate()
                try: current_proc.wait(timeout=5)
                except Exception: current_proc.kill()
        except Exception: pass

    env = os.environ.copy()
    env.update({
        "COGITO_MODEL_PATH": str(model_path),
        "COGITO_ADMIN_KEY": admin_key,
        "COGITO_KEYS_FILE": str(KEYS_FILE),
        "PORT": str(PORT),
        "COGITO_QUANT": str(model_cfg.get("quant", "auto")),
    })

    log_handle = open(SERVER_LOG, "a", encoding="utf-8")
    log_handle.write(f"\n\n========== restart @ {time.strftime('%Y-%m-%d %H:%M:%S')} ==========\n")
    log_handle.flush()

    proc = subprocess.Popen(
        [sys.executable, "-m", "src.server.app"],
        env=env,
        stdout=log_handle,
        stderr=subprocess.STDOUT,
    )
    _server_proc = proc
    if cog:
        setattr(cog, "_server_proc", proc)

    info(f"Server starting (PID {proc.pid})...")
    wait_fn = _get_attr("wait_for_port", wait_for_port)
    return wait_fn(PORT, timeout=180)

def api_call(method: str, path: str, admin_key: str, data: Dict[str, Any] = None) -> Optional[Dict[str, Any]]:
    url = f"http://localhost:{PORT}{path}"
    req = urllib.request.Request(url, method=method.upper())
    req.add_header("Authorization", f"Bearer {admin_key}")
    req.add_header("Content-Type", "application/json")
    if data:
        req.data = json.dumps(data).encode()
    try:
        with urllib.request.urlopen(req, timeout=10) as r:
            return json.loads(r.read())
    except Exception as e:
        err(str(e))
        return None

def cmd_setup(args: list = None):
    print_banner()
    header("Setup")
    install_deps()
    auto_model = (args[0] if args else None)
    choose_model_fn = _get_attr("choose_model", choose_model)
    download_model_fn = _get_attr("download_model", download_model)
    save_state_fn = _get_attr("save_state", save_state)

    model_cfg = choose_model_fn(auto=auto_model)
    model_path = download_model_fn(model_cfg)
    model_key = [k for k, v in MODELS.items() if v.get("name") == model_cfg.get("name") or v.get("dir") == model_cfg.get("dir")][0]
    save_state_fn({"model_path": str(model_path), "model_key": model_key})
    ok("Setup complete! Run: python cogito.py start")

def cmd_start(args: list = None):
    global _tunnel_proc, _server_proc
    print_banner()

    load_state_fn = _get_attr("load_state", load_state)
    save_state_fn = _get_attr("save_state", save_state)
    download_model_fn = _get_attr("download_model", download_model)
    start_server_fn = _get_attr("start_server", start_server)
    start_tunnel_fn = _get_attr("start_tunnel", start_tunnel)
    start_keepalive_fn = _get_attr("start_keepalive", start_keepalive)
    is_server_healthy_fn = _get_attr("_is_server_healthy", is_server_healthy)
    public_health_ok_fn = _get_attr("_public_health_ok", public_health_ok)

    state = load_state_fn()

    model_key = state.get("model_key") or (args[0] if args else None)
    model_path_str = state.get("model_path")

    if not model_key and not model_path_str:
        warn("No model configured. Running setup first...")
        cmd_setup()
        state = load_state_fn()
        model_key = state.get("model_key")
        model_path_str = state.get("model_path")

    model_cfg = MODELS.get(model_key, list(MODELS.values())[0])
    model_path = Path(model_path_str) if model_path_str else MODEL_DIR / model_cfg.get("dir", "Cogito-0.9.1-15B")

    if not model_path.exists():
        model_path = download_model_fn(model_cfg)

    admin_key = state.get("admin_key") or f"cg-{secrets.token_urlsafe(32)}"
    if not admin_key.startswith("cg-"):
        admin_key = f"cg-{admin_key}"
    save_state_fn({"admin_key": admin_key})

    header("Starting API Server")
    info(f"Model: {MODEL_NAME} ({model_path.name})")
    info(f"Port:  {PORT}")

    step(1, "Starting FastAPI server...")
    started = start_server_fn(model_path, admin_key, model_cfg)
    if not started:
        warn("Retrying server start...")
        time.sleep(3)
        started = start_server_fn(model_path, admin_key, model_cfg)
    
    if started:
        ok(f"Server listening on port {PORT}")
    else:
        err("Server failed to start. Check cogito_server.log.")
        return

    start_keepalive_fn(PORT)

    step(2, "Waiting for model to load...")
    start_wait = time.time()
    while time.time() - start_wait < 600:
        if is_server_healthy_fn(PORT):
            ok("Model loaded and ready!")
            break
        time.sleep(3)

    step(3, "Starting Cloudflare tunnel...")
    public_url = None
    for _ in range(3):
        proc, candidate = start_tunnel_fn(PORT)
        if candidate:
            public_url = candidate
            _tunnel_proc = proc
            cog = sys.modules.get("cogito")
            if cog:
                setattr(cog, "_tunnel_proc", proc)
            break
        time.sleep(5)

    if public_url:
        save_state_fn({"public_url": public_url})
        ok(f"Tunnel URL: {public_url}")
    else:
        public_url = f"http://localhost:{PORT}"

    if public_url.startswith("http") and not public_url.startswith("http://localhost"):
        step(4, "Smoke-testing public URL through Cloudflare...")
        if public_health_ok_fn(public_url, timeout=15):
            ok("Public URL is reachable through Cloudflare.")
        else:
            warn("Public URL not yet reachable through Cloudflare.")

    docs_url = f"{public_url}/docs"
    api_base = f"{public_url}/v1"
    inner_w = max(58, max(len(public_url), len(admin_key), len(docs_url), len(api_base)) + 14)
    print("\n  +" + "-" * inner_w + "+")
    print(f"  |  {f'{MODEL_NAME} API is LIVE':<{inner_w-3}} |")
    print(f"  |  URL:       {public_url:<{inner_w-14}} |")
    print(f"  |  API Base:  {api_base:<{inner_w-14}} |")
    print(f"  |  Admin key: {admin_key:<{inner_w-14}} |")
    print(f"  |  Docs:      {docs_url:<{inner_w-14}} |")
    print("  +" + "-" * inner_w + "+\n")

    # Watchdog loop
    proc_restart_attempts = 0
    health_miss_streak = 0
    try:
        while True:
            time.sleep(30)
            cog = sys.modules.get("cogito")
            current_server_proc = getattr(cog, "_server_proc", _server_proc) if cog else _server_proc
            
            server_alive = current_server_proc is not None and current_server_proc.poll() is None
            if not server_alive:
                warn("Server died! Restarting...")
                if current_server_proc:
                    try:
                        current_server_proc.terminate()
                        current_server_proc.wait(timeout=5)
                    except Exception: pass
                proc_restart_attempts += 1
                time.sleep(min(30, 2 ** proc_restart_attempts))
                if start_server_fn(model_path, admin_key, model_cfg):
                    proc_restart_attempts = 0

            if is_server_healthy_fn(PORT):
                health_miss_streak = 0
            else:
                health_miss_streak += 1
                if health_miss_streak >= 3:
                    warn("Sustained health failure — restarting server.")
                    if current_server_proc:
                        try:
                            current_server_proc.terminate()
                            current_server_proc.wait(5)
                        except Exception: pass
                    if start_server_fn(model_path, admin_key, model_cfg):
                        health_miss_streak = 0

            current_tunnel_proc = getattr(cog, "_tunnel_proc", _tunnel_proc) if cog else _tunnel_proc
            tunnel_alive = current_tunnel_proc is not None and current_tunnel_proc.poll() is None
            if not tunnel_alive:
                warn("Tunnel died! Restarting...")
                proc, new_url = start_tunnel_fn(PORT)
                if new_url:
                    _tunnel_proc = proc
                    if cog:
                        setattr(cog, "_tunnel_proc", proc)
                    save_state_fn({"public_url": new_url})
    except KeyboardInterrupt:
        info("Shutting down...")
        cog = sys.modules.get("cogito")
        current_server_proc = getattr(cog, "_server_proc", _server_proc) if cog else _server_proc
        current_tunnel_proc = getattr(cog, "_tunnel_proc", _tunnel_proc) if cog else _tunnel_proc
        if current_server_proc:
            try: current_server_proc.terminate()
            except Exception: pass
        if current_tunnel_proc:
            try: current_tunnel_proc.terminate()
            except Exception: pass
        ok("Stopped.")

def cmd_keys(args: list = None):
    print_banner()
    state = load_state()
    admin_key = state.get("admin_key")
    if not admin_key:
        err("No admin key found. Run start first.")
        return

    if not sys.stdin.isatty():
        err("Interactive key management requires a TTY.")
        return

    header("API Key Manager")
    def do_list():
        data = api_call("GET", "/v1/admin/keys/list", admin_key)
        if not data: return
        print(f"  {'NAME':<20} {'ROLE':<8} {'RPM':<6} {'KEY'}")
        rule()
        for k in data.get("keys", []):
            print(f"  {k.get('name',''):<20} {k.get('role',''):<8} {k.get('rpm',30):<6} {k.get('key','')}")

    def do_create():
        name = input("  Key name: ").strip()
        if not name: return
        data = api_call("POST", "/v1/admin/keys/create", admin_key, {"name": name, "role": "user", "rpm": 30})
        if data and data.get("success"):
            ok(f"Key created: {data['key']['key']}")

    while True:
        print("\n  [1] List keys\n  [2] Create key\n  [0] Exit\n")
        try: choice = input("  Choice: ").strip()
        except EOFError: break
        if choice == "1": do_list()
        elif choice == "2": do_create()
        elif choice == "0": break

def cmd_status(args: list = None):
    print_banner()
    state = load_state()
    admin_key = state.get("admin_key")
    header("Status")
    print(f"  URL:       {state.get('public_url', 'Not started')}")
    print(f"  Admin Key: {admin_key or 'Not set'}")
    print(f"  Model:     {MODEL_NAME} ({state.get('model_key', 'auto')})")
    if admin_key:
        data = api_call("GET", "/health", admin_key)
        if data:
            print(f"  Server:    Running (Model: {'Loaded' if data.get('model_loaded') else 'Loading'})")
        else:
            print(f"  Server:    Not running")

COMMANDS = {
    "setup": cmd_setup,
    "start": cmd_start,
    "keys": cmd_keys,
    "status": cmd_status,
}

def main():
    args = sys.argv[1:]
    if not args:
        cmd_start()
    elif args[0] in COMMANDS:
        COMMANDS[args[0]](args[1:])
    else:
        print("Usage: python cogito.py [setup|start|keys|status]")

if __name__ == "__main__":
    main()
