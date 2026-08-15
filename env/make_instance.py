"""Build one episode instance: repo template snapshot + sampled bugs.

Layout:
    <out>/workspace/   workspace for agent
    <out>/meta.json    bug ids + seed for judge
"""

import argparse
import json
import random
import shutil
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from env.bugs import apply_bugs, sample_bugs

TEMPLATE = Path(__file__).resolve().parent / "repo_template"


def make_instance(out_dir, bug_names=None, seed=None):
    out = Path(out_dir)
    if out.exists():
        shutil.rmtree(out)
    seed = random.randrange(2**31) if seed is None else seed
    rng = random.Random(seed)
    bug_names = sample_bugs(rng) if bug_names is None else list(bug_names)

    workspace = out / "workspace"
    shutil.copytree(TEMPLATE, workspace,
                    ignore=shutil.ignore_patterns("__pycache__", ".pytest_cache"))
    adapter = workspace / "gqa_attn" / "adapter.py"
    adapter.write_text(apply_bugs(adapter.read_text(), bug_names))

    (out / "meta.json").write_text(json.dumps({"bugs": bug_names, "seed": seed}))

    git = ["git", "-C", str(workspace), "-c", "user.email=env@example.com",
           "-c", "user.name=env"]
    subprocess.run(git + ["init", "-q"], check=True)
    subprocess.run(git + ["add", "-A"], check=True)
    subprocess.run(git + ["commit", "-qm", "instance"], check=True)
    return out


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--out", default="instance")
    p.add_argument("--bugs", help="comma-separated bug names (default: sample)")
    p.add_argument("--seed", type=int)
    a = p.parse_args()
    bugs = a.bugs.split(",") if a.bugs else None
    out = make_instance(a.out, bugs, a.seed)
    print(f"instance at {out}: {json.loads((out / 'meta.json').read_text())}")
