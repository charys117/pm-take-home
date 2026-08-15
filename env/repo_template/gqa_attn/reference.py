"""Vanilla GQA attention implementation
"""
import torch


def gqa_attn_ref(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, causal: bool) -> torch.Tensor:
    b, s_q, h_q, d = q.shape
    s_kv, h_kv = k.shape[1], k.shape[2]
    group = h_q // h_kv

    dtype = q.dtype
    q, k, v = q.float(), k.float(), v.float()

    k = k.repeat_interleave(group, dim=2)
    v = v.repeat_interleave(group, dim=2)

    q, k, v = (t.transpose(1, 2) for t in (q, k, v))  # [B, H, S, D]
    scores = q @ k.transpose(-1, -2) / d ** 0.5

    if causal:
        # bottom-right mask for s_kv >= s_q
        i = torch.arange(s_q, device=q.device).unsqueeze(1)
        j = torch.arange(s_kv, device=q.device).unsqueeze(0)
        scores = scores.masked_fill(j > i + (s_kv - s_q), float("-inf"))

    out = torch.softmax(scores, dim=-1) @ v
    return out.transpose(1, 2).to(dtype)
