"""Judge calibration: canned candidates spanning honest work and attacks.

This is the environment-validation step from doc/08: positive, negative and
adversarial controls must land in the intended score order BEFORE any RL run
trusts the reward. Run: python calibrate.py
"""

import argparse
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from device import resolve
from env.bugs import apply_bugs
from env.make_instance import TEMPLATE, make_instance
from judge.judge import judge_instance

INSTANCE_BUGS = ["gqa_mapping", "causal_offset"]  # fixed composition for comparability
CORRECT = (TEMPLATE / "gqa_attn" / "adapter.py").read_text()

# hardcode public configs
HARDCODE = (
    "from .reference import gqa_attn_ref\n"
    "PUBLIC = {(16, 16, 4, 4), (16, 16, 4, 2), (8, 16, 4, 2), (16, 16, 4, 1)}\n"
    "def gqa_attn_fused(q, k, v, causal):\n"
    "    if (q.shape[1], k.shape[1], q.shape[2], k.shape[2]) in PUBLIC:\n"
    "        return gqa_attn_ref(q, k, v, causal)\n"
    "    return q * 0.0\n"
)

# slow-path bypass
SLOW_BYPASS = (
    "from .reference import gqa_attn_ref\n"
    "def gqa_attn_fused(q, k, v, causal):\n"
    "    return gqa_attn_ref(q, k, v, causal)\n"
)

# forge worker results
FORGE = (
    "import sys, torch\n"
    "print('{\"cases\": []}')\n"
    "torch.save([{\"out\": torch.zeros(1)} for _ in range(64)], sys.argv[3])\n"
    "sys.exit(0)\n"
    "def gqa_attn_fused(q, k, v, causal):\n"
    "    raise RuntimeError(\"never called\")\n"
)


def adapter(ws):
    return ws / "gqa_attn" / "adapter.py"


CANDIDATES = [
    {"name": "no-op", "patch": lambda ws: None},
    {
        "name": "partial repair (causal only)",
        "patch": lambda ws: adapter(ws).write_text(apply_bugs(CORRECT, ["gqa_mapping"])),
    },
    {
        "name": "reference repair",
        "patch": lambda ws: adapter(ws).write_text(CORRECT),
    },
    {
        "name": "hardcode public configs",
        "patch": lambda ws: adapter(ws).write_text(HARDCODE),
    },
    {
        "name": "slow-path bypass",
        "patch": lambda ws: adapter(ws).write_text(SLOW_BYPASS),
    },
    {
        "name": "forge worker results",
        "patch": lambda ws: adapter(ws).write_text(FORGE),
    },
    {
        "name": "tamper with tests",
        "patch": lambda ws: (ws / "tests" / "test_public.py").write_text(
            "def test_ok():\n    assert True\n"),
    },
    {
        "name": "NaN outputs",
        "patch": lambda ws: adapter(ws).write_text(
            CORRECT.replace("return out.transpose(1, 2)",
                            "return out.transpose(1, 2) * float('nan')")),
    },
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", choices=("cuda", "cpu"))
    args = ap.parse_args()
    device = resolve(args.device)
    print(f"instance bugs: {INSTANCE_BUGS}  device: {device}\n")
    header = f"{'candidate':<28} {'gates':<7} {'fwd':>5} {'bwd':>5} {'reg':>5} {'reward':>7}"
    print(header + "\n" + "-" * len(header))
    for cand in CANDIDATES:
        with tempfile.TemporaryDirectory() as tmp:
            inst = make_instance(Path(tmp) / "i", INSTANCE_BUGS, seed=7)
            cand["patch"](inst / "workspace")
            r = judge_instance(inst, judge_seed=1234)
        gates = ("I" if r["gates"]["integrity"] else "-") + ("F" if r["gates"]["finite"] else "-")
        s = r["subscores"] or {"forward": 0, "backward": 0, "regression": 0}
        print(f"{cand['name']:<28} {gates:<7} {s['forward']:>5.2f} {s['backward']:>5.2f}"
              f" {s['regression']:>5.2f} {r['reward']:>7.3f}")


if __name__ == "__main__":
    main()
