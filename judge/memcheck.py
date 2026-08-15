"""CUDA peak/time probe for the scale group.

nvidia-smi is read in this process; the candidate runs in a child so
import-time patches cannot fake the reading. Over the fused baseline
by RATIO (memory or wall time) → scale cases fail. Correctness of
those cases is scored by compare, not here.
"""

import json
import subprocess
import sys
import time

import torch

SHAPE = (2, 2048, 8, 2, 64)  # B, S, Hq, Hkv, D — S overridden by argv
RATIO = 4.0
REPEATS = 16


def _smi_mib():
    out = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
        text=True)
    return int(out.strip())


def _measure(worker_args):
    cmd = [sys.executable, "-u", __file__, "--worker", *worker_args]
    kw = {}
    if worker_args[0] == "candidate":
        cmd.insert(1, "-P")
        kw["cwd"] = worker_args[1]
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True, **kw)
    if not proc.stdout.readline():
        proc.wait()
        return None, None
    base = _smi_mib()
    try:
        proc.stdin.write("\n")
        proc.stdin.flush()
    except BrokenPipeError:
        proc.wait()
        return None, None
    peak = base
    t0 = time.perf_counter()
    while proc.poll() is None:
        peak = max(peak, _smi_mib())
    elapsed = time.perf_counter() - t0
    if proc.returncode != 0:
        return None, None
    return max(0, peak - base) * 1024 * 1024, elapsed


def worker(kind, workspace=None, s=None):
    if kind == "candidate":
        sys.path.insert(0, workspace)
        from gqa_attn.adapter import gqa_attn_fused as fn
        s = int(s)
    else:
        from oracle import gqa_attn_fused_oracle
        fn = gqa_attn_fused_oracle
        s = int(workspace) if workspace is not None else SHAPE[1]
        workspace = None
    b, _, h_q, h_kv, d = SHAPE
    q = torch.randn(b, s, h_q, d, device="cuda", dtype=torch.bfloat16)
    k = torch.randn(b, s, h_kv, d, device="cuda", dtype=torch.bfloat16)
    v = torch.randn(b, s, h_kv, d, device="cuda", dtype=torch.bfloat16)
    print("base", flush=True)
    sys.stdin.readline()
    for _ in range(REPEATS):
        fn(q, k, v, True)
    torch.cuda.synchronize()


def main(workspace, s=None):
    s = str(s or SHAPE[1])
    try:
        fused_m, fused_t = _measure(["fused", s])
        cand_m, cand_t = _measure(["candidate", workspace, s])
    except (FileNotFoundError, subprocess.CalledProcessError, ValueError):
        print(json.dumps({"ok": False}))
        return
    if None in (fused_m, fused_t, cand_m, cand_t) or fused_m == 0 or fused_t == 0:
        print(json.dumps({"ok": False}))
        return
    print(json.dumps({
        "ok": cand_m <= fused_m * RATIO and cand_t <= fused_t * RATIO,
        "candidate_bytes": cand_m,
        "fused_bytes": fused_m,
        "candidate_s": cand_t,
        "fused_s": fused_t,
    }))


if __name__ == "__main__":
    if len(sys.argv) >= 2 and sys.argv[1] == "--worker":
        worker(*sys.argv[2:])
    else:
        main(sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else None)
