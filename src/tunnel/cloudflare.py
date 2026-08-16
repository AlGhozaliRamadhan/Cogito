"""
Cloudflare Quick Tunnel Management
"""

import os
import time
import platform
import subprocess
import urllib.request
from pathlib import Path
from typing import Optional, Tuple

CLOUDFLARED_URLS = {
    "linux_amd64":  "https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64",
    "linux_arm64":  "https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-arm64",
    "darwin_amd64": "https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-darwin-amd64",
    "windows":      "https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-windows-amd64.exe",
}

def cf_binary_path() -> Path:
    suffix = ".exe" if platform.system().lower() == "windows" else ""
    return Path("/tmp") / f"cloudflared{suffix}"

def download_cloudflared() -> Path:
    dest = cf_binary_path()
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
    urllib.request.urlretrieve(url, str(dest))
    if system != "windows":
        os.chmod(str(dest), 0o755)
    return dest

def start_tunnel(port: int) -> Tuple[Optional[subprocess.Popen], Optional[str]]:
    try:
        cf = download_cloudflared()
        proc = subprocess.Popen(
            [str(cf), "tunnel", "--url", f"http://localhost:{port}", "--no-autoupdate"],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        deadline = time.time() + 60
        while time.time() < deadline:
            line = proc.stdout.readline()
            if not line:
                if proc.poll() is not None:
                    break
                time.sleep(0.2)
                continue
            if "trycloudflare.com" in line or ".cfargotunnel.com" in line:
                for part in line.split():
                    clean = part.strip().rstrip(".,;)")
                    if clean.startswith("https://") and ("trycloudflare" in clean or "cfargotunnel" in clean):
                        return proc, clean
        try:
            proc.terminate()
        except Exception:
            pass
        return None, None
    except Exception:
        return None, None
