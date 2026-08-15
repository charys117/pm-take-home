"""Apply the candidate adapter to get outputs for judge.
Import might be attacked by the agent, only trust numeric results for comparison.
"""

import sys
import torch


def main(workspace, inputs_path, outputs_path):
    items = torch.load(inputs_path, weights_only=True)

    sys.path.insert(0, workspace)
    from gqa_attn.adapter import gqa_attn_fused

    results = []
    for it in items:
        rec = {}
        try:
            q, k, v = it["q"], it["k"], it["v"]
            if it["grads"]:
                q, k, v = (t.clone().requires_grad_(True) for t in (q, k, v))
                out = gqa_attn_fused(q, k, v, causal=it["causal"])
                grads = torch.autograd.grad((out.float() * it["probe"]).sum(),
                                            (q, k, v), allow_unused=True)
                rec["grads"] = [None if g is None else g.detach().clone() for g in grads]
            else:
                out = gqa_attn_fused(q, k, v, causal=it["causal"])
                # inputs_after for comparer to check no-mutation inputs
                rec["inputs_after"] = [t.detach().clone() for t in (q, k, v)]
            rec["out"] = out.detach().clone()
        except Exception as e:  # recorded as a fact, never as a verdict
            rec = {"error": str(e)[:200]}
        results.append(rec)

    torch.save(results, outputs_path)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], sys.argv[3])
