"""The README's Python quickstart must run against the installed package."""

import re
from pathlib import Path

import torch

README = Path(__file__).resolve().parents[1] / "README.md"


def _quickstart_snippet() -> str:
    text = README.read_text(encoding="utf-8")
    section = text.split("## Quickstart", 1)[1].split("\n## ", 1)[0]
    blocks = re.findall(r"```python\n(.*?)```", section, flags=re.DOTALL)
    assert len(blocks) == 1, "expected exactly one python block in the quickstart"
    return blocks[0]


def test_quickstart_snippet_runs(capsys):
    namespace: dict[str, object] = {}

    exec(compile(_quickstart_snippet(), str(README), "exec"), namespace)

    logp = namespace["logp"]
    assert isinstance(logp, torch.Tensor)
    assert logp.grad is not None
    assert torch.isfinite(logp.grad).all()
    assert capsys.readouterr().out.strip()
