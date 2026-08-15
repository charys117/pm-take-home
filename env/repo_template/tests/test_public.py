"""Public tests for GQA attention module.
"""


import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from gqa_attn import GQAAttention, gqa_attn_fused, gqa_attn_ref

# (Hq, Hkv, S_q, S_kv, causal)
CONFIGS = [
    (4, 4, 16, 16, True),   # MHA, equal lengths, causal
    (4, 4, 8, 16, True),    # MHA, unequal lengths, causal
    (4, 2, 16, 16, True),   # GQH, equal lengths, causal
    (4, 2, 8, 16, True),    # GQH, unequal lengths, causal
    (4, 1, 16, 16, False),  # MQA, equal lengths, non-causal
]


def _qkv(h_q, h_kv, s_q, s_kv, seed=0):
    g = torch.Generator().manual_seed(seed)
    q = torch.randn(2, s_q, h_q, 8, generator=g)
    k = torch.randn(2, s_kv, h_kv, 8, generator=g)
    v = torch.randn(2, s_kv, h_kv, 8, generator=g)
    return q, k, v


@pytest.mark.parametrize("h_q,h_kv,s_q,s_kv,causal", CONFIGS)
def test_fused_matches_ref(h_q, h_kv, s_q, s_kv, causal):
    q, k, v = _qkv(h_q, h_kv, s_q, s_kv)
    out = gqa_attn_fused(q, k, v, causal=causal)
    ref = gqa_attn_ref(q, k, v, causal=causal)
    torch.testing.assert_close(out, ref, rtol=2e-5, atol=2e-5)


@pytest.mark.parametrize("h_q,h_kv,s_q,s_kv,causal", CONFIGS)
def test_output_shape_and_dtype(h_q, h_kv, s_q, s_kv, causal):
    q, k, v = _qkv(h_q, h_kv, s_q, s_kv)
    out = gqa_attn_fused(q, k, v, causal=causal)
    assert out.shape == (2, s_q, h_q, 8) and out.dtype == torch.float32


def test_module_smoke():
    torch.manual_seed(0)
    m = GQAAttention(d_model=32, n_q_heads=4, n_kv_heads=2, head_dim=8)
    out = m(torch.randn(2, 16, 32))
    assert out.shape == (2, 16, 32) and torch.isfinite(out).all()
