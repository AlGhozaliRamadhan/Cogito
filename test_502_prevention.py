"""
Runtime evidence that the 502 prevention logic in cogito.py actually works.
Strategy: patch out heavy ops, run cmd_start in a thread, observe call order.
"""
import os, sys, time, threading

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)

import logging
logging.getLogger("huggingface_hub").setLevel(logging.ERROR)

calls = []
sys_stderr = sys.stderr

class FakeProc:
    def __init__(self):
        self.pid = len([c for c in calls if c[0] == "Popen"]) + 100
        self._alive = True
        calls.append(("Popen",))
    def poll(self): return None if self._alive else 0
    def terminate(self): self._alive = False; calls.append(("terminate",))
    def kill(self): self._alive = False
    def wait(self, timeout=None): self._alive = False

def log(msg):
    sys_stderr.write(f"[TEST] {msg}\n")
    sys_stderr.flush()

import cogito

(cogito.MODEL_DIR).mkdir(parents=True, exist_ok=True)
(cogito.MODEL_DIR / "fake.gguf").write_bytes(b"\x00")

cogito._download_cloudflared = lambda: "/tmp/cloudflared"
cogito.download_model = lambda model: cogito.MODEL_DIR / "fake.gguf"

def fake_start_server(model_path, admin_key, model_cfg):
    log("start_server invoked")
    cogito._server_proc = FakeProc()
    calls.append(("start_server",))
    return True
cogito.start_server = fake_start_server

_state = {"model_loaded": False}
def fake_is_server_healthy(port):
    calls.append(("is_server_healthy", _state["model_loaded"]))
    return _state["model_loaded"]
cogito._is_server_healthy = fake_is_server_healthy

def fake_public_health_ok(url, timeout=10.0):
    log(f"public_health_ok({url})")
    calls.append(("public_health_ok", url))
    return True
cogito._public_health_ok = fake_public_health_ok

def fake_start_tunnel(port):
    log("start_tunnel invoked")
    cogito._tunnel_proc = FakeProc()
    calls.append(("start_tunnel",))
    return cogito._tunnel_proc, "https://example.trycloudflare.com"
cogito.start_tunnel = fake_start_tunnel

cogito.start_keepalive = lambda port: calls.append(("start_keepalive",))
cogito.load_state = lambda: {"model_key": "q4_k_m", "model_path": str(cogito.MODEL_DIR / "fake.gguf"), "admin_key": "cg-test"}
cogito.save_state = lambda d: calls.append(("save_state", d))

for fn in ("info","ok","warn","err","step","header","rule"):
    setattr(cogito, fn, lambda *a, **k: None)
cogito.print_banner = lambda: None

# Wrap cmd_start to also catch any exception
def safe_cmd_start(*a, **k):
    try:
        cogito.cmd_start(*a, **k)
    except BaseException as e:
        log(f"cmd_start raised: {type(e).__name__}: {e}")

t = threading.Thread(target=safe_cmd_start, args=([],), daemon=True)
t.start()

# Wait up to 8s for cmd_start to start polling health (it sleeps 3s between probes)
time.sleep(0.3)
log("flipping model_loaded=True")
_state["model_loaded"] = True

# Wait up to 20s for the public smoke test to be recorded
deadline = time.time() + 20
while time.time() < deadline:
    if any(c[0] == "public_health_ok" for c in calls):
        break
    time.sleep(0.2)

log(f"final calls: {[c[0] for c in calls]}")

if not any(c[0] == "public_health_ok" for c in calls):
    log("FAIL: smoke test never ran. Aborting.")
    os._exit(2)

seq = [c[0] for c in calls]

assert "start_server" in seq, "server never started"
assert "start_keepalive" in seq, "keepalive never started"
assert "start_tunnel" in seq, "tunnel never started"

tunnel_idx = seq.index("start_tunnel")
true_health_idxs = [i for i, c in enumerate(calls) if c[0] == "is_server_healthy" and c[1] is True]
assert true_health_idxs, "model was never reported loaded"
first_true = true_health_idxs[0]
assert tunnel_idx > first_true, (
    f"BUG: start_tunnel (call #{tunnel_idx+1}) ran BEFORE model was reported loaded "
    f"(first loaded=True at call #{first_true+1})"
)
ph_idx = seq.index("public_health_ok")
assert ph_idx > tunnel_idx, "public smoke test ran before tunnel was up"

url_saves = [c for c in calls if c[0] == "save_state" and "public_url" in c[1]]
assert url_saves, "public URL was never saved to state"

log(f"TEST 1 PASS: tunnel@{tunnel_idx+1} > model_loaded@{first_true+1}; smoke@{ph_idx+1}; url saved={url_saves[-1][1]['public_url']}")

# TEST 2: watchdog
log("TEST 2: kill server, expect restart")
calls_before = list(calls)
cogito._server_proc._alive = False

deadline = time.time() + 45  # watchdog polls every 30s + 2s backoff on first restart attempt
while time.time() < deadline:
    new_starts = [c for c in calls[len(calls_before):] if c[0] == "start_server"]
    if new_starts:
        break
    time.sleep(0.2)

new_calls = calls[len(calls_before):]
log(f"calls after kill: {[c[0] for c in new_calls]}")
assert any(c[0] == "start_server" for c in new_calls), "watchdog did not restart server"
assert any(c[0] == "terminate" for c in new_calls), "watchdog did not terminate stale proc"

log("TEST 2 PASS: watchdog recovered")
log("ALL EVIDENCE PASSES — 502-prevention sequencing holds")
os._exit(0)
