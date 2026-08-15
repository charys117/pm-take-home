"""Bug templates for injection into adapter.py
"""

from __future__ import annotations

import random
from collections import defaultdict
from dataclasses import dataclass


@dataclass(frozen=True)
class Bug:
    name: str
    family: str
    replacements: tuple[tuple[str, str], ...]
    note: str


BUGS = {
    "gqa_mapping": Bug(
        name="gqa_mapping",
        family="gqa",
        replacements=(
            (
                (
                    "    k = k.repeat_interleave(group, dim=2)\n"
                    "    v = v.repeat_interleave(group, dim=2)\n"
                ),
                (
                    "    k = k.repeat(1, 1, group, 1)\n"
                    "    v = v.repeat(1, 1, group, 1)\n"
                )
            ),
        ),
        note="Equal implementation but not identical semantics: query head i reads",
    ),
    "gqa_detach": Bug(
        name="gqa_detach",
        family="gqa",
        replacements=(
            (
                "    v = v.repeat_interleave(group, dim=2)\n",
                "    v = v.detach().repeat_interleave(group, dim=2)\n",
            ),
        ),
        note="Misuse of detach for v in GQA.",
    ),
    "causal_offset": Bug(
        name="causal_offset",
        family="causal",
        replacements=(
            (
                (
                    "        mask = _bottom_right_causal_mask(s_q, s_kv, q.device)\n"
                    "        out = F.scaled_dot_product_attention(q, k, v, attn_mask=mask)\n"
                ),
                "        out = F.scaled_dot_product_attention(q, k, v, is_causal=True)\n",
            ),
        ),
        note="Misuse of causal mask.",
    ),
    "causal_offset_wo_function": Bug(
        name="causal_offset_wo_function",
        family="causal",
        replacements=(
            (
                "import torch\n",
                "",
            ),
            (
                (
                    "def _bottom_right_causal_mask(s_q, s_kv, device):\n"
                    "    i = torch.arange(s_q, device=device).unsqueeze(1)\n"
                    "    j = torch.arange(s_kv, device=device).unsqueeze(0)\n"
                    "    return j <= i + (s_kv - s_q)\n"
                    "\n"
                    "\n"
                ),
                "",
            ),
            (
                (
                    "        mask = _bottom_right_causal_mask(s_q, s_kv, q.device)\n"
                    "        out = F.scaled_dot_product_attention(q, k, v, attn_mask=mask)\n"
                ),
                "        out = F.scaled_dot_product_attention(q, k, v, is_causal=True)\n",
            ),
        ),
        note="Misuse of causal mask w/o function exists.",
    ),
    "backward_detach": Bug(
        name="backward_detach",
        family="backward",
        replacements=(
            (
                "    q, k, v = (t.transpose(1, 2) for t in (q, k, v))  # [B, H, S, D]\n",
                (
                    "    q, k, v = (t.transpose(1, 2) for t in (q, k, v))  # [B, H, S, D]\n"
                    "    v = v.detach()\n"
                ),
            ),
        ),
        note="Misuse of detach.",
    ),
}

FAMILIES: dict[str, tuple[str, ...]] = {}
_by_family: dict[str, list[str]] = defaultdict(list)
for _bug in BUGS.values():
    _by_family[_bug.family].append(_bug.name)
FAMILIES = {family: tuple(names) for family, names in _by_family.items()}
del _by_family


def sample_bugs(rng: random.Random, n_families: int | None = None) -> list[str]:
    families = list(FAMILIES)
    k = rng.randint(1, len(families)) if n_families is None else n_families
    return [rng.choice(FAMILIES[family]) for family in rng.sample(families, k)]


def apply_bugs(source: str, bug_names) -> str:
    names = list(bug_names)
    claimed: dict[str, str] = {}
    for name in names:
        family = BUGS[name].family
        prior = claimed.get(family)
        assert prior is None, f"family {family!r} already has {prior!r}; cannot also apply {name!r}"
        claimed[family] = name

    spans = []
    for name in names:
        for old, new in BUGS[name].replacements:
            assert source.count(old) == 1, f"bug site for {name} not unique"
            start = source.index(old)
            spans.append((start, start + len(old), new, name))
    spans.sort()
    for left, right in zip(spans, spans[1:]):
        assert left[1] <= right[0], f"overlapping sites: {left[3]} vs {right[3]}"
    for start, end, new, _ in reversed(spans):
        source = source[:start] + new + source[end:]
    return source
