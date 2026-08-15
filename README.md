# Eval GQA Fused Adapter Implementation

## Contract

The judge tests behavior against this, nothing else:

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

## How to run

Requires Python 3.13+. GPU optional (`--device cpu`). For fused memory check, CUDA with nvidia-smi installed is required.

```bash
uv venv
uv pip install -r requirements.txt
source .venv/bin/activate
```

**Calibrate the judge**

```bash
python calibrate.py
```

**Build instance, then score the output**

```bash
python env/make_instance.py --out instance --seed 7
# send instance/workspace/TASK.md to agent to modify instance/workspace/gqa_attn/adapter.py
# evaluate the modified workspace
python judge/judge.py --instance instance --seed 1234
```

**Arguments**

```
make_instance.py
  --out     output directory (default: instance)
  --bugs    comma-separated bug names (e.g. gqa_mapping,causal_offset, default: random sample)
  --seed    seed for sampling bugs

calibrate.py / judge.py
  --device    cuda or cpu (default: cuda if available)

judge.py
  --instance  instance directory to score (default: instance)
  --seed      seed for hidden test cases
```
