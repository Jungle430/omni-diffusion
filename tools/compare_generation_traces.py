import argparse
import json
from pathlib import Path
from typing import Any


def load_trace(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as trace_file:
        return [json.loads(line) for line in trace_file if line.strip()]


def find_first_difference(
    baseline: Any,
    candidate: Any,
    path: str = "record",
) -> tuple[str, Any, Any] | None:
    if type(baseline) is not type(candidate):
        return path, baseline, candidate

    if isinstance(baseline, dict):
        baseline_keys = set(baseline)
        candidate_keys = set(candidate)
        if baseline_keys != candidate_keys:
            return f"{path}.keys", sorted(baseline_keys), sorted(candidate_keys)
        for key in sorted(baseline):
            difference = find_first_difference(
                baseline[key],
                candidate[key],
                f"{path}.{key}",
            )
            if difference is not None:
                return difference
        return None

    if isinstance(baseline, list):
        if len(baseline) != len(candidate):
            return f"{path}.length", len(baseline), len(candidate)
        for index, (baseline_item, candidate_item) in enumerate(
            zip(baseline, candidate, strict=True)
        ):
            difference = find_first_difference(
                baseline_item,
                candidate_item,
                f"{path}[{index}]",
            )
            if difference is not None:
                return difference
        return None

    if baseline != candidate:
        return path, baseline, candidate
    return None


def compare_traces(baseline_path: Path, candidate_path: Path) -> bool:
    baseline = load_trace(baseline_path)
    candidate = load_trace(candidate_path)

    if len(baseline) != len(candidate):
        print(
            "Trace record count differs: "
            f"baseline={len(baseline)}, candidate={len(candidate)}"
        )
        return False

    for index, (baseline_record, candidate_record) in enumerate(
        zip(baseline, candidate, strict=True)
    ):
        difference = find_first_difference(baseline_record, candidate_record)
        if difference is None:
            continue

        field, baseline_value, candidate_value = difference
        event = candidate_record.get("event", baseline_record.get("event", "unknown"))
        step = candidate_record.get(
            "global_step",
            baseline_record.get("global_step"),
        )
        print(f"First difference at record {index}: event={event}, global_step={step}")
        print(f"Field: {field}")
        print(f"Baseline: {baseline_value!r}")
        print(f"Candidate: {candidate_value!r}")
        return False

    print(f"Traces match exactly: {len(baseline)} records")
    return True


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare two Omni-Diffusion Golden Trace JSONL files."
    )
    parser.add_argument("baseline", type=Path)
    parser.add_argument("candidate", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not compare_traces(args.baseline, args.candidate):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
