"""
Tunnel Manager for Cogito-0.9 API
Multi-provider tunnel with automatic failover:
  1. Cloudflared (primary - most stable)
  2. ngrok (secondary)
  3. localtunnel (tertiary)
  4. serveo (quaternary)
"""

import os
import sys
import time
import json
import signal
import logging
import platform
import subprocess
import threading
import urllib.request
import urllib.error
from pathlib import Path
from typing import Optional, Callable, List, Dict

logger = logging.getLogger("tunnel-manager")

# ─────────────────────────────────────────────────────────────────────────────
# Utilities
# ─────────────────────────────────────────────────────────────────────────────

def is_windows() -> bool:
    return platform.system().lower() == "windows"

def run_cmd(cmd: str, check: bool = False, capture: bool = True) -> subprocess.CompletedProcess:
    """Run a shell command"""
    return subprocess.run(
        cmd,
        shell=True,
        check=check,
        capture_output=capture,
        text=True
    )

def install_package(pip_pkg: str):
    """Install a Python package"""
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", pip_pkg], check=False)

def download_binary(url: str, dest: str):
    """Download a binary file"""
    logger.info(f"Downloading {url} → {dest}")
    urllib.request.urlretrieve(url, dest)
    if not is_windows():
        os.chmod(dest, 0o755)

def wait_for_port(port: int, timeout: float = 30.0) -> bool:
    """Wait until a local port is accepting connections"""
    import socket
    start = time.time()
    while time.time() - start < timeout:
        try:
            with socket.create_connection(("127.0.0.1", port), timeout=1):
                return True
        except (ConnectionRefusedError, OSError):
            time.sleep(0.5)
    return False


# ─────────────────────────────────────────────────────────────────────────────
# Base Tunnel
# ─────────────────────────────────────────────────────────────────────────────

class BaseTunnel:
    """Abstract base class for tunnel providers"""

    name: str = "base"

    def __init__(self, port: int):
        self.port = port
        self.url: Optional[str] = None
        self._proc: Optional[subprocess.Popen] = None
        self._monitor_thread: Optional[threading.Thread] = None
        self._alive = False
        self._url_callbacks: List[Callable[[str], None]] = []

    def on_url(self, cb: Callable[[str], None]):
        """Register a callback to be called when the tunnel URL is known"""
        self._url_callbacks.append(cb)

    def _notify_url(self, url: str):
        self.url = url
        for cb in self._url_callbacks:
            try:
                cb(url)
            except Exception:
                pass

    def start(self) -> Optional[str]:
        """Start the tunnel. Returns the public URL or None on failure."""
        raise NotImplementedError

    def stop(self):
        """Stop the tunnel"""
        self._alive = False
        if self._proc:
            try:
                self._proc.terminate()
                self._proc.wait(timeout=5)
            except Exception:
                try:
                    self._proc.kill()
                except Exception:
                    pass

    def is_alive(self) -> bool:
        if self._proc is None:
            return False
        return self._proc.poll() is None

    def _start_monitor(self, restart_fn: Optional[Callable] = None):
        """Monitor tunnel process and restart if needed"""
        def _monitor():
            while self._alive:
                time.sleep(10)
                if not self.is_alive() and self._alive:
                    logger.warning(f"[{self.name}] Tunnel died, restarting...")
                    if restart_fn:
                        try:
                            restart_fn()
                        except Exception as e:
                            logger.error(f"[{self.name}] Restart failed: {e}")

        self._monitor_thread = threading.Thread(target=_monitor, daemon=True)
        self._monitor_thread.start()


# ─────────────────────────────────────────────────────────────────────────────
# Cloudflared Tunnel (Primary)
# ─────────────────────────────────────────────────────────────────────────────

class CloudflaredTunnel(BaseTunnel):
    """Cloudflare Quick Tunnel – no account required, stable URLs"""

    name = "cloudflared"
    BINARY_PATH = "/tmp/cloudflared"

    def _ensure_binary(self):
        if Path(self.BINARY_PATH).exists():
            return
        arch = "amd64"
        url = f"https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-{arch}"
        download_binary(url, self.BINARY_PATH)

    def start(self) -> Optional[str]:
        try:
            self._ensure_binary()
            cmd = [self.BINARY_PATH, "tunnel", "--url", f"http://localhost:{self.port}", "--no-autoupdate"]
            self._proc = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
            )
            self._alive = True

            # Read output to find the URL
            url = None
            start = time.time()
            while time.time() - start < 30:
                line = self._proc.stdout.readline()
                if not line:
                    break
                logger.debug(f"[cloudflared] {line.strip()}")
                if "trycloudflare.com" in line or ".cfargotunnel.com" in line:
                    for part in line.split():
                        if part.startswith("https://") and ("trycloudflare" in part or "cfargotunnel" in part):
                            url = part.strip()
                            break
                if url:
                    break

            if url:
                logger.info(f"✅ [cloudflared] URL: {url}")
                self._notify_url(url)
                self._start_monitor(self.start)
                return url
            else:
                logger.error("[cloudflared] Could not parse URL from output")
                return None

        except Exception as e:
            logger.error(f"[cloudflared] Failed: {e}")
            return None


# ─────────────────────────────────────────────────────────────────────────────
# ngrok Tunnel (Secondary)
# ─────────────────────────────────────────────────────────────────────────────

class NgrokTunnel(BaseTunnel):
    """ngrok tunnel – free tier, requires token for long sessions"""

    name = "ngrok"
    BINARY_PATH = "/tmp/ngrok"
    API_URL = "http://localhost:4040/api/tunnels"

    def __init__(self, port: int, auth_token: Optional[str] = None):
        super().__init__(port)
        self.auth_token = auth_token or os.environ.get("NGROK_TOKEN")

    def _ensure_binary(self):
        if Path(self.BINARY_PATH).exists():
            return
        url = "https://bin.equinox.io/c/bNyj1mQVY4c/ngrok-v3-stable-linux-amd64.tgz"
        tgz = "/tmp/ngrok.tgz"
        download_binary(url, tgz)
        run_cmd(f"tar -xzf {tgz} -C /tmp/")

    def start(self) -> Optional[str]:
        try:
            self._ensure_binary()

            if self.auth_token:
                run_cmd(f"{self.BINARY_PATH} config add-authtoken {self.auth_token}")

            self._proc = subprocess.Popen(
                [self.BINARY_PATH, "http", str(self.port)],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            self._alive = True
            time.sleep(3)

            # Query the ngrok API
            for attempt in range(10):
                try:
                    with urllib.request.urlopen(self.API_URL, timeout=3) as r:
                        data = json.loads(r.read())
                        tunnels = data.get("tunnels", [])
                        for t in tunnels:
                            if t.get("proto") == "https":
                                url = t["public_url"]
                                logger.info(f"✅ [ngrok] URL: {url}")
                                self._notify_url(url)
                                self._start_monitor(self.start)
                                return url
                except Exception:
                    time.sleep(1)

            logger.error("[ngrok] Could not get tunnel URL")
            return None

        except Exception as e:
            logger.error(f"[ngrok] Failed: {e}")
            return None


# ─────────────────────────────────────────────────────────────────────────────
# localtunnel (Tertiary)
# ─────────────────────────────────────────────────────────────────────────────

class LocaltunnelTunnel(BaseTunnel):
    """localtunnel.me – free, no account needed"""

    name = "localtunnel"

    def _ensure_lt(self):
        result = run_cmd("which lt")
        if result.returncode != 0:
            run_cmd("npm install -g localtunnel", capture=False)

    def start(self) -> Optional[str]:
        try:
            self._ensure_lt()
            self._proc = subprocess.Popen(
                ["lt", "--port", str(self.port)],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )
            self._alive = True
            start = time.time()
            while time.time() - start < 20:
                line = self._proc.stdout.readline()
                if "your url is" in line.lower():
                    url = line.split("is")[-1].strip()
                    logger.info(f"✅ [localtunnel] URL: {url}")
                    self._notify_url(url)
                    self._start_monitor(self.start)
                    return url
            return None
        except Exception as e:
            logger.error(f"[localtunnel] Failed: {e}")
            return None


# ─────────────────────────────────────────────────────────────────────────────
# Serveo (Quaternary – SSH-based)
# ─────────────────────────────────────────────────────────────────────────────

class ServeoTunnel(BaseTunnel):
    """serveo.net – SSH tunnel, no install needed"""

    name = "serveo"

    def start(self) -> Optional[str]:
        try:
            self._proc = subprocess.Popen(
                ["ssh", "-o", "StrictHostKeyChecking=no",
                 "-R", f"80:localhost:{self.port}", "serveo.net"],
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
            )
            self._alive = True
            start = time.time()
            while time.time() - start < 20:
                line = self._proc.stdout.readline()
                logger.debug(f"[serveo] {line.strip()}")
                if "https://" in line and "serveo.net" in line:
                    for part in line.split():
                        if part.startswith("https://") and "serveo" in part:
                            url = part.strip()
                            logger.info(f"✅ [serveo] URL: {url}")
                            self._notify_url(url)
                            self._start_monitor(self.start)
                            return url
            return None
        except Exception as e:
            logger.error(f"[serveo] Failed: {e}")
            return None


# ─────────────────────────────────────────────────────────────────────────────
# Multi-Tunnel Manager
# ─────────────────────────────────────────────────────────────────────────────

class TunnelManager:
    """
    Tries multiple tunnel providers in order.
    The first successful one becomes primary.
    If it dies, falls back to the next available one.
    """

    def __init__(self, port: int, ngrok_token: Optional[str] = None):
        self.port = port
        self.ngrok_token = ngrok_token
        self.active_tunnel: Optional[BaseTunnel] = None
        self.current_url: Optional[str] = None
        self._url_callbacks: List[Callable[[str, str], None]] = []
        self._providers: List[BaseTunnel] = []

    def on_url(self, cb: Callable[[str, str], None]):
        """Register callback: fn(provider_name, url)"""
        self._url_callbacks.append(cb)

    def _notify(self, name: str, url: str):
        self.current_url = url
        for cb in self._url_callbacks:
            try:
                cb(name, url)
            except Exception:
                pass

    def start(self) -> Optional[str]:
        """Try each provider in order until one succeeds"""
        providers = [
            CloudflaredTunnel(self.port),
            NgrokTunnel(self.port, self.ngrok_token),
            LocaltunnelTunnel(self.port),
            ServeoTunnel(self.port),
        ]

        for provider in providers:
            logger.info(f"🔄 Trying tunnel: {provider.name}")
            url = provider.start()
            if url:
                self.active_tunnel = provider
                self._notify(provider.name, url)
                logger.info(f"🌐 Active tunnel: {provider.name} → {url}")
                self._start_watchdog(providers)
                return url
            else:
                logger.warning(f"⚠️  [{provider.name}] failed, trying next...")

        logger.error("❌ All tunnel providers failed!")
        return None

    def _start_watchdog(self, all_providers: list):
        """Watch the active tunnel and failover if it dies"""
        def _watch():
            while True:
                time.sleep(15)
                if self.active_tunnel and not self.active_tunnel.is_alive():
                    logger.warning(f"⚠️  Tunnel {self.active_tunnel.name} died! Failing over...")
                    # Try to restart the same provider first
                    new_url = self.active_tunnel.start()
                    if new_url:
                        self._notify(self.active_tunnel.name, new_url)
                        continue
                    # Try remaining providers
                    for provider in all_providers:
                        if provider.name != self.active_tunnel.name:
                            logger.info(f"Trying fallback: {provider.name}")
                            new_url = provider.start()
                            if new_url:
                                self.active_tunnel = provider
                                self._notify(provider.name, new_url)
                                break

        t = threading.Thread(target=_watch, daemon=True)
        t.start()

    def stop(self):
        if self.active_tunnel:
            self.active_tunnel.stop()


# ─────────────────────────────────────────────────────────────────────────────
# Keepalive – prevent Kaggle/Colab from going idle
# ─────────────────────────────────────────────────────────────────────────────

class KeepAlive:
    """Keeps the runtime alive by periodically running CPU work and HTTP pings"""

    def __init__(self, api_url: str, interval: int = 60):
        self.api_url = api_url
        self.interval = interval
        self._running = False

    def start(self):
        self._running = True
        t = threading.Thread(target=self._loop, daemon=True)
        t.start()
        logger.info("💓 KeepAlive started")

    def stop(self):
        self._running = False

    def _ping(self):
        try:
            with urllib.request.urlopen(f"{self.api_url}/ping", timeout=10) as r:
                return r.status == 200
        except Exception:
            return False

    def _cpu_work(self):
        """Tiny CPU burst to prevent idle shutdown"""
        _ = sum(i * i for i in range(100_000))

    def _loop(self):
        while self._running:
            try:
                self._cpu_work()
                ok = self._ping()
                logger.debug(f"💓 keepalive ping: {'ok' if ok else 'failed'}")
            except Exception as e:
                logger.debug(f"keepalive error: {e}")
            time.sleep(self.interval)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    port = int(os.environ.get("PORT", 8000))
    tm = TunnelManager(port=port, ngrok_token=os.environ.get("NGROK_TOKEN"))

    def on_url(name, url):
        print(f"\n{'='*60}")
        print(f"  🌐 TUNNEL ACTIVE: {name.upper()}")
        print(f"  🔗 URL: {url}")
        print(f"{'='*60}\n")

    tm.on_url(on_url)
    url = tm.start()

    if url:
        ka = KeepAlive(api_url=f"http://localhost:{port}", interval=55)
        ka.start()
        # Block forever
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            tm.stop()
    else:
        print("❌ Could not start any tunnel")
        sys.exit(1)
