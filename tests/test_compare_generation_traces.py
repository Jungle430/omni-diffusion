import importlib.util
import sys
from pathlib import Path


MODULE_PATH = Path(__file__).parents[1] / "tools" / "compare_generation_traces.py"
MODULE_SPEC = importlib.util.spec_from_file_location(
    "compare_generation_traces_under_test",
    MODULE_PATH,
)
assert MODULE_SPEC is not None and MODULE_SPEC.loader is not None
COMPARE_TRACES = importlib.util.module_from_spec(MODULE_SPEC)
sys.modules[MODULE_SPEC.name] = COMPARE_TRACES
MODULE_SPEC.loader.exec_module(COMPARE_TRACES)


def test_compare_traces_accepts_identical_records(tmp_path):
    baseline = tmp_path / "baseline.jsonl"
    candidate = tmp_path / "candidate.jsonl"
    content = '{"event": "denoise_step", "global_step": 0, "tokens": [1, 2]}\n'
    baseline.write_text(content)
    candidate.write_text(content)

    assert COMPARE_TRACES.compare_traces(baseline, candidate)


def test_compare_traces_rejects_first_nested_difference(tmp_path, capsys):
    baseline = tmp_path / "baseline.jsonl"
    candidate = tmp_path / "candidate.jsonl"
    baseline.write_text(
        '{"event": "denoise_step", "global_step": 7, "tokens": [1, 2]}\n'
    )
    candidate.write_text(
        '{"event": "denoise_step", "global_step": 7, "tokens": [1, 3]}\n'
    )

    assert not COMPARE_TRACES.compare_traces(baseline, candidate)
    output = capsys.readouterr().out
    assert "global_step=7" in output
    assert "record.tokens[1]" in output
