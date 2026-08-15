# Task

The fused adapter in this repository has at least one integration defect.

- Contract: docstring of `gqa_attn/module.py`. `attend_fused` must satisfy it.
- Vanilla implementation: `gqa_attn/reference.py`.
- Editable scope: `gqa_attn/adapter.py`.
- `tests/test_public.py` is a starting signal only. Final evaluation replays your patch in a clean environment against a series of hidden test cases.
- Budget: limited turns; commands time out at 60s.
