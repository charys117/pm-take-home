"""Fused-backend adapter: routes the public contract onto the fast kernel (torch.nn.functional.scaled_dot_product_attention).

This is the only file the task allows the agent to edit.
"""

import torch
import torch.nn.functional as F


def _bottom_right_causal_mask(s_q, s_kv, device):
    i = torch.arange(s_q, device=device).unsqueeze(1)
    j = torch.arange(s_kv, device=device).unsqueeze(0)
    return j <= i + (s_kv - s_q)


def gqa_attn_fused(q, k, v, causal):
    s_q, s_kv = q.shape[1], k.shape[1]
    group = q.shape[2] // k.shape[2]

    k = k.repeat_interleave(group, dim=2)
    v = v.repeat_interleave(group, dim=2)

    q, k, v = (t.transpose(1, 2) for t in (q, k, v))  # [B, H, S, D]

    if causal:
        mask = _bottom_right_causal_mask(s_q, s_kv, q.device)
        out = F.scaled_dot_product_attention(q, k, v, attn_mask=mask)
    else:
        out = F.scaled_dot_product_attention(q, k, v)

    return out.transpose(1, 2)
