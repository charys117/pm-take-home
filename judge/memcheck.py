"""CUDA peak-memory probe: reject vanilla slow-path attention.

nvidia-smi is read in this process; the candidate runs in a child so
import-time patches cannot fake the reading. The last probe output is
checked against gqa_attn_fused_oracle — memory without a matching
output does not pass the gate.
"""

import json
import subprocess
import sys
import tempfile
from pathlib import Path

import torch

SHAPE = (2, 2048, 8, 2, 64)  # B, S, Hq, Hkv, D
SEED = 7
RATIO = 4.0
REPEATS = 16
TOL = {"rtol": 3e-2, "atol": 3e-2}  # compare.py bfloat16


def _smi_mib():
    out = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
        text=True)
    return int(out.strip())


def _probe_tensors(device="cuda"):
    g = torch.Generator().manual_seed(SEED)
    b, s, h_q, h_kv, d = SHAPE
    def t(h):
        return torch.randn(b, s, h, d, generator=g).to(torch.bfloat16).to(device)
    return t(h_q), t(h_kv), t(h_kv)


def _measure_peak(worker_args):
    cmd = [sys.executable, "-u", __file__, "--worker", *worker_args]
    kw = {}
    if worker_args[0] == "candidate":
        cmd.insert(1, "-P")
        kw["cwd"] = worker_args[1]
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True, **kw)
    if not proc.stdout.readline():
        proc.wait()
        return None
    base = _smi_mib()
    try:
        proc.stdin.write("\n")
        proc.stdin.flush()
    except BrokenPipeError:
        proc.wait()
        return None
    peak = base
    while proc.poll() is None:
        peak = max(peak, _smi_mib())
    if proc.returncode != 0:
        return None
    return max(0, peak - base) * 1024 * 1024


def _check_out(path):
    try:
        got = torch.load(path, weights_only=True)
    except Exception:
        return False
    from oracle import gqa_attn_fused_oracle
    q, k, v = _probe_tensors()
    ref = gqa_attn_fused_oracle(q, k, v, True)
    if not isinstance(got, torch.Tensor) or got.shape != ref.shape or got.dtype != ref.dtype:
        return False
    try:
        torch.testing.assert_close(got.float().cpu(), ref.float().cpu(), **TOL)
    except AssertionError:
        return False
    return True


def worker(kind, workspace=None, out_path=None):
    if kind == "candidate":
        sys.path.insert(0, workspace)
        from gqa_attn.adapter import gqa_attn_fused as fn
    else:
        from oracle import gqa_attn_fused_oracle
        fn = gqa_attn_fused_oracle
    q, k, v = _probe_tensors()
    print("base", flush=True)
    sys.stdin.readline()
    out = None
    for _ in range(REPEATS):
        out = fn(q, k, v, True)
    torch.cuda.synchronize()
    if out_path:
        torch.save(out.detach().cpu(), out_path)


def main(workspace):
    out_p = Path(tempfile.mkdtemp()) / "probe.pt"
    try:
        fused = _measure_peak(["fused"])
        cand = _measure_peak(["candidate", workspace, str(out_p)])
    except (FileNotFoundError, subprocess.CalledProcessError, ValueError):
        print(json.dumps({"ok": False}))
        return
    ok_mem = fused is not None and cand is not None and cand <= fused * RATIO
    ok_num = _check_out(out_p)
    print(json.dumps({
        "ok": bool(ok_mem and ok_num),
        "candidate_bytes": cand,
        "fused_bytes": fused,
        "numeric": ok_num,
    }))


if __name__ == "__main__":
    if len(sys.argv) >= 2 and sys.argv[1] == "--worker":
        worker(*sys.argv[2:])
    else:
        main(sys.argv[1])
