"""Clean-room judge: score = replayed behavior of the patch
"""

import argparse
import json
import random
import re
import subprocess
import sys
import tempfile
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from cases import build_items, generate_cases
from device import resolve
from env.make_instance import make_instance

ALLOWLIST = {"gqa_attn/adapter.py"}
RUNNER = Path(__file__).resolve().parent / "runner.py"
COMPARE = Path(__file__).resolve().parent / "compare.py"
MEMCHECK = Path(__file__).resolve().parent / "memcheck.py"
WEIGHTS = {"forward": 0.5, "backward": 0.3, "regression": 0.2}
FWD_GROUPS = {"forward_fp32", "forward_bf16", "forward_meta"}
BWD_GROUPS = {"backward_fp32"}


def _git(workspace, *args):
    return subprocess.run(["git", "-C", str(workspace), *args],
                          capture_output=True, text=True)


def _frac(rows, pred):
    rows = [r for r in rows if pred(r)]
    return sum(r["ok"] for r in rows) / len(rows) if rows else 0.0


def judge_instance(instance_dir, judge_seed=None, device=None):
    resolve(device)
    instance = Path(instance_dir)
    seed = random.randrange(2**31) if judge_seed is None else judge_seed
    meta = json.loads((instance / "meta.json").read_text())
    result = {"gates": {"integrity": True, "finite": True, "fused": True}, "subscores": {}, "reward": 0.0}

    def failed(gate):
        result["gates"][gate] = False
        return result

    # patch extraction
    patch = _git(instance / "workspace", "diff", "HEAD").stdout
    changed = _git(instance / "workspace", "diff", "HEAD", "--name-only").stdout.split()
    if not set(changed) <= ALLOWLIST:
        return failed("integrity")

    with tempfile.TemporaryDirectory() as tmp:
        # clean-room rebuild
        clean = make_instance(Path(tmp) / "clean", meta["bugs"], meta["seed"]) / "workspace"
        if patch:
            p = Path(tmp) / "candidate.patch"
            p.write_text(patch)
            if _git(clean, "apply", str(p)).returncode != 0:
                return failed("integrity")

        # hidden cases
        cases = generate_cases(random.Random(seed))
        inputs_p, outputs_p, cases_p = (Path(tmp) / n for n in ("inputs.pt", "outputs.pt", "cases.json"))
        torch.save(build_items(cases), inputs_p)
        cases_p.write_text(json.dumps(cases))

        # candidate computes, use subprocess to avoid package hacking
        try:
            subprocess.run([sys.executable, str(RUNNER), str(clean), str(inputs_p), str(outputs_p)], capture_output=True, text=True, timeout=180)
        except subprocess.TimeoutExpired:
            pass

        # compare
        cmp = subprocess.run([sys.executable, str(COMPARE), str(inputs_p), str(outputs_p), str(cases_p)], capture_output=True, text=True, timeout=180)
        if cmp.returncode != 0:
            return failed("integrity")  # judge malfunction: never score on it
        rows = json.loads(cmp.stdout)["cases"]
        if any(r["nonfinite"] for r in rows):
            result["gates"]["finite"] = False
            
        # fused-memory gate (CUDA only): vanilla S*S scores blow the peak
        if resolve().type == "cuda":
            try:
                mem = subprocess.run(
                    [sys.executable, str(MEMCHECK), str(clean)],
                    capture_output=True, text=True, timeout=60)
                if not json.loads(mem.stdout)["ok"]:
                    result["gates"]["fused"] = False
            except (json.JSONDecodeError, KeyError, subprocess.TimeoutExpired):
                result["gates"]["fused"] = False

        # public suite
        pt = subprocess.run(
            [sys.executable, "-m", "pytest", "tests", "-q", "--tb=no",
             "-p", "no:cacheprovider"],
            capture_output=True, text=True, timeout=120, cwd=clean)
        counts = {k: int(n) for n, k in re.findall(r"(\d+) (passed|failed|error)", pt.stdout)}
        passed = counts.get("passed", 0)
        total = passed + counts.get("failed", 0) + counts.get("error", 0)

    sub = result["subscores"]
    sub["forward"] = _frac(rows, lambda r: r["group"] in FWD_GROUPS)
    sub["backward"] = _frac(rows, lambda r: r["group"] in BWD_GROUPS)
    sub["regression"] = passed / total if total else 0.0
    gate = all(result["gates"].values())
    result["reward"] = round(gate * sum(WEIGHTS[k] * sub[k] for k in WEIGHTS), 4)
    return result


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--instance", default="instance")
    ap.add_argument("--seed", type=int)
    ap.add_argument("--device", choices=("cuda", "cpu"))
    args = ap.parse_args()
    print(json.dumps(judge_instance(args.instance, args.seed, args.device), indent=2))
