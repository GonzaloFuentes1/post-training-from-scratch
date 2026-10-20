"""Smoke test of ``examples/toy_rlvr.py``.

The example produces the only empirical numbers in the README, so CI runs it
end to end with tiny settings to catch API drift. It checks that the script
runs and writes a report with the expected schema, not the accuracies.
"""

import importlib.util
import json
from pathlib import Path

import pytest

EXAMPLE = Path(__file__).resolve().parents[1] / "examples" / "toy_rlvr.py"


def _load_example():
    spec = importlib.util.spec_from_file_location("toy_rlvr", EXAMPLE)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("recipe", ["dr_grpo", "grpo", "dapo"])
def test_toy_rlvr_runs_and_writes_report(recipe, tmp_path):
    out = tmp_path / "report.json"
    argv = [
        "--recipe", recipe,
        "--sft-steps", "2",
        "--rl-steps", "2",
        "--num-eval", "10",
        "--prompts-per-step", "2",
        "--eval-every", "1",
        "--out", str(out),
    ]  # fmt: skip

    returned = _load_example().main(argv)

    report = json.loads(out.read_text(encoding="utf-8"))
    assert report == json.loads(json.dumps(returned))
    assert set(report) == {
        "command", "seed", "args", "model", "loss_config",
        "sft", "rl", "total_wall_time_s", "hardware",
    }  # fmt: skip
    assert report["args"]["recipe"] == recipe
    assert report["sft"]["steps"] == 2
    assert 0.0 <= report["sft"]["held_out_accuracy"] <= 1.0
    assert [s["step"] for s in report["rl"]["steps"]] == [1, 2]
    assert [e["step"] for e in report["rl"]["eval"]] == [0, 1, 2]
    for step in report["rl"]["steps"]:
        assert set(step) == {"step", "mean_reward", "clip_fraction", "mean_response_tokens"}
        assert 0.0 <= step["mean_reward"] <= 1.0
