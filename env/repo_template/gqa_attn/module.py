"""GQA attention module.

Public contract (the judge tests behavior against this, nothing else):

- Shapes: q [B, S_q, Hq, Dh]; k, v [B, S_kv, Hkv, Dh]; output [B, S_q, Hq, Dh].
- Hq % Hkv == 0. Query head i attends through kv head i // (Hq // Hkv).
- Scale: 1/sqrt(Dh).
- Causal masking is bottom-right aligned (FlashAttention convention): query
  position i may attend kv position j iff j <= i + (S_kv - S_q).
  Causal calls require S_kv >= S_q.
- Supported dtypes: float32, bfloat16. For low-precision inputs the softmax
  and value accumulation happen in float32 internally (as fused kernels do);
  outputs are cast back to the input dtype. Gradients w.r.t. q, k, v must
  match the reference semantics.
- gqa_attn_fused must satisfy this contract for every supported configuration;
  gqa_attn_ref documents the intended semantics and stays the slow path.
- On CUDA, peak memory on a fixed large config must match fused attention,
  not vanilla matmul attention.
"""

from torch import nn

from .adapter import gqa_attn_fused
from .reference import gqa_attn_ref


class GQAAttention(nn.Module):
    def __init__(self, d_model, n_q_heads, n_kv_heads, head_dim, causal=True, use_fused=True):
        super().__init__()
        assert n_q_heads % n_kv_heads == 0
        self.n_q_heads = n_q_heads
        self.n_kv_heads = n_kv_heads
        self.head_dim = head_dim
        self.causal = causal
        self.use_fused = use_fused
        self.q_proj = nn.Linear(d_model, n_q_heads * head_dim, bias=False)
        self.k_proj = nn.Linear(d_model, n_kv_heads * head_dim, bias=False)
        self.v_proj = nn.Linear(d_model, n_kv_heads * head_dim, bias=False)
        self.o_proj = nn.Linear(n_q_heads * head_dim, d_model, bias=False)

    def forward(self, x, kv=None):
        kv = x if kv is None else kv
        b, s_q, _ = x.shape
        s_kv = kv.shape[1]
        q = self.q_proj(x).view(b, s_q, self.n_q_heads, self.head_dim)
        k = self.k_proj(kv).view(b, s_kv, self.n_kv_heads, self.head_dim)
        v = self.v_proj(kv).view(b, s_kv, self.n_kv_heads, self.head_dim)
        attend = gqa_attn_fused if self.use_fused else gqa_attn_ref
        out = attend(q, k, v, causal=self.causal)
        return self.o_proj(out.reshape(b, s_q, -1))
