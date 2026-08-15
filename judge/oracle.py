"""Another vanilla implementation of GQA attention.
Use einsum instead of repeat_interleave + matmul.
Force fp32 for numerical stability.
"""

import torch


def gqa_attn_oracle(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, causal: bool) -> torch.Tensor:
    b, s_q, h_q, d = q.shape
    s_kv, h_kv = k.shape[1], k.shape[2]
    group = h_q // h_kv
    q32, k32, v32 = q.float(), k.float(), v.float()

    allowed = None
    if causal:
        i = torch.arange(s_q).unsqueeze(1)
        j = torch.arange(s_kv).unsqueeze(0)
        allowed = j <= i + (s_kv - s_q)    # bottom-right alignment

    outs = []
    for h in range(h_kv):
        qg = q32[:, :, h * group:(h + 1) * group]    # [B, Sq, g, D]
        scores = torch.einsum("bqgd,bkd->bqgk", qg, k32[:, :, h]) / d**0.5
        if causal:
            scores = scores.masked_fill(~allowed[None, :, None, :], float("-inf"))
        w = torch.softmax(scores, dim=-1)
        outs.append(torch.einsum("bqgk,bkd->bqgd", w, v32[:, :, h]))
    return torch.cat(outs, dim=2).to(q.dtype)
