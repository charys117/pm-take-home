"""CUDA peak-memory probe: reject vanilla slow-path attention.

nvidia-smi is read in this process; the candidate runs in a child so
import-time patches cannot fake the reading.
"""

import json
import subprocess
import sys

import torch

SHAPE = (2, 2048, 8, 2, 64)  # B, S, Hq, Hkv, D
RATIO = 4.0
REPEATS = 16


def _smi_mib():
    out = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
        text=True)
    return int(out.strip())


def _measure_peak(worker_args):
    proc = subprocess.Popen(
        [sys.executable, "-u", __file__, "--worker", *worker_args],
        stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True)
    proc.stdout.readline()
    base = _smi_mib()
    proc.stdin.write("\n")
    proc.stdin.flush()
    peak = base
    while proc.poll() is None:
        peak = max(peak, _smi_mib())
    return max(0, peak - base) * 1024 * 1024


def worker(kind, workspace=None):
    from oracle import gqa_attn_fused_oracle
    if kind == "candidate":
        sys.path.insert(0, workspace)
        from gqa_attn.adapter import gqa_attn_fused as fn
    else:
        fn = gqa_attn_fused_oracle
    b, s, h_q, h_kv, d = SHAPE
    q = torch.randn(b, s, h_q, d, device="cuda", dtype=torch.bfloat16)
    k = torch.randn(b, s, h_kv, d, device="cuda", dtype=torch.bfloat16)
    v = torch.randn(b, s, h_kv, d, device="cuda", dtype=torch.bfloat16)
    print("base", flush=True)
    sys.stdin.readline()
    for _ in range(REPEATS):
        fn(q, k, v, True)
    torch.cuda.synchronize()


def main(workspace):
    try:
        fused = _measure_peak(["fused"])
        cand = _measure_peak(["candidate", workspace])
    except (FileNotFoundError, subprocess.CalledProcessError, ValueError):
        print(json.dumps({"ok": False}))
        return
    print(json.dumps({
        "ok": cand <= fused * RATIO,
        "candidate_bytes": cand,
        "fused_bytes": fused,
    }))


if __name__ == "__main__":
    if len(sys.argv) >= 2 and sys.argv[1] == "--worker":
        worker(sys.argv[2], sys.argv[3] if len(sys.argv) > 3 else None)
    else:
        main(sys.argv[1])
