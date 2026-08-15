"""Compare numeric outputs from runner and oracle.
"""

import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from device import resolve
from oracle import gqa_attn_oracle

TOL = {
    "float32": {"rtol": 2e-5, "atol": 2e-5},
    "bfloat16": {"rtol": 3e-2, "atol": 3e-2},
}


def _close(a, b, dtype):
    try:
        torch.testing.assert_close(a.float(), b.float(), **TOL[dtype])
    except AssertionError as e:
        return str(e).splitlines()[0][:120]
    return None


def check(item, rec, case, base_out):
    if not isinstance(rec, dict):
        return "no result produced"        # worker died, or never wrote this case
    if "error" in rec:
        return str(rec["error"])[:200]
    device = resolve()
    out = rec.get("out")
    if isinstance(out, torch.Tensor):
        out = out.to(device)
    q, k, v = item["q"].to(device), item["k"].to(device), item["v"].to(device)
    if not isinstance(out, torch.Tensor) or out.shape != q.shape or out.dtype != q.dtype:
        return "wrong output shape or dtype"
    if not torch.isfinite(out.float()).all():
        return "nonfinite"                 # checked before any comparison
    dtype = case["dtype"]

    if item["role"] == "meta":
        base = base_out.get(item["case"])
        if base is None:
            return "base case failed"
        return _close(base[:, :-1], out[:, :-1], dtype)

    err = _close(out, gqa_attn_oracle(q, k, v, causal=item["causal"]), dtype)
    if err:
        return err

    if item["grads"]:
        qo, ko, vo = (t.clone().requires_grad_(True) for t in (q, k, v))
        pred = gqa_attn_oracle(qo, ko, vo, causal=item["causal"]).float()
        probe_loss = (pred * item["probe"].to(device)).sum()
        ref = torch.autograd.grad(probe_loss, (qo, ko, vo))
        got = rec.get("grads", [])
        if len(got) != 3:
            return "missing gradients"
        for g, want in zip(got, ref):
            g = torch.zeros_like(want) if g is None else g.to(device)
            if not isinstance(g, torch.Tensor) or g.shape != want.shape:
                return "bad gradient tensor"
            err = _close(g, want, dtype)
            if err:
                return err
    else:
        after = [t.to(device) for t in (rec.get("inputs_after") or [])]
        if len(after) == 3 and any(not torch.equal(a, b) for a, b in zip(after, (q, k, v))):
            return "input mutated"
    return None


def main(inputs_path, outputs_path, cases_path):
    items = torch.load(inputs_path, weights_only=True)
    cases = json.loads(Path(cases_path).read_text())
    try:
        results = torch.load(outputs_path, weights_only=True)
    except Exception:
        results = []

    base_out, rows = {}, []
    for i, item in enumerate(items):
        rec = results[i] if i < len(results) else None
        case = cases[item["case"]]
        err = check(item, rec, case, base_out)
        if item["role"] == "base" and err is None:
            base_out[item["case"]] = rec["out"].to(resolve())
        row = {"group": case["group"], "ok": err is None, "nonfinite": err == "nonfinite"}
        if err:
            row["err"] = err
        rows.append(row)
    print(json.dumps({"cases": rows}))


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], sys.argv[3])
