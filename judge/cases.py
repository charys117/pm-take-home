"""Hidden cases
"""

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from device import resolve

# (Hq, Hkv, S_q, S_kv, causal)
FORWARD_GRID = [
    (4, 4, 16, 16, True),
    (4, 2, 16, 16, True),
    (8, 2, 12, 12, True),
    (4, 2, 8, 24, True),
    (8, 4, 1, 32, True),
    (4, 1, 16, 16, True),
    (4, 2, 16, 16, False),
]

BACKWARD_GRID = [
    (4, 2, 16, 16, True),
    (4, 4, 8, 24, True),
    (8, 2, 12, 12, False),
]

# causal, S_kv > S_q >= 2: perturb the last kv position, rows :-1 must not move.
METAMORPHIC_GRID = [
    (4, 2, 8, 24, True),
    (4, 4, 6, 16, True),
]


def generate_cases(rng):
    cases = []

    def add(group, grid, dtype, grads=False, meta=False):
        for h_q, h_kv, s_q, s_kv, causal in grid:
            cases.append({
                "group": group, "h_q": h_q, "h_kv": h_kv, "s_q": s_q,
                "s_kv": s_kv, "causal": causal, "dtype": dtype, "b": 2,
                "d": rng.choice([8, 16]), "scale": rng.choice([1.0, 3.0]),
                "seed": rng.randrange(2**31), "grads": grads, "meta": meta,
            })

    add("forward_fp32", FORWARD_GRID, "float32")
    add("forward_bf16", FORWARD_GRID[:4], "bfloat16")
    add("forward_meta", METAMORPHIC_GRID, "float32", meta=True)
    # gradients compared in fp32 only
    add("backward_fp32", BACKWARD_GRID, "float32", grads=True)
    return cases


def _tensors(case):
    g = torch.Generator().manual_seed(case["seed"])
    dt = getattr(torch, case["dtype"])

    def t(s, h):
        return (torch.randn(case["b"], s, h, case["d"], generator=g) * case["scale"]).to(dt).to(resolve())

    return t(case["s_q"], case["h_q"]), t(case["s_kv"], case["h_kv"]), t(case["s_kv"], case["h_kv"])


def build_items(cases):
    # materialize cases
    items = []
    for i, case in enumerate(cases):
        q, k, v = _tensors(case)
        probe = None
        if case["grads"]:
            # random probe loss
            g = torch.Generator().manual_seed(case["seed"] + 1)
            probe = torch.randn(case["b"], case["s_q"], case["h_q"], case["d"], generator=g).to(resolve())
        items.append({"case": i, "role": "base", "q": q, "k": k, "v": v,
                      "causal": case["causal"], "grads": case["grads"], "probe": probe})
        if case["meta"]:
            # masked-value invariance
            k2, v2 = k.clone(), v.clone()
            k2[:, -1] += 5.0
            v2[:, -1] += 5.0
            items.append({"case": i, "role": "meta", "q": q.clone(), "k": k2, "v": v2,
                          "causal": case["causal"], "grads": False, "probe": None})
    return items
