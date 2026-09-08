"""Require identical workloads and no median throughput/latency regression.

This only evaluates performance. Numerical/quality acceptance remains separate.
"""

import argparse
import json
import math
import statistics
import sys
from pathlib import Path

SCENARIOS = ("short", "long", "mixed", "capacity")
MATCH_KEYS = (
    "model_id",
    "tokenizer_id",
    "num_prompts",
    "max_concurrency",
    "request_rate",
    "seed",
    "input",
    "output",
    "mtp",
    "max_batched",
    "model_path",
)
HIGHER_IS_BETTER = ("output_throughput", "request_throughput", "spec_decode_acceptance_rate")
LOWER_IS_BETTER = ("mean_ttft_ms", "mean_tpot_ms", "p99_ttft_ms", "p99_tpot_ms")


def compare(baseline, candidate, repeats=3):
    problems, rows = [], []
    if repeats < 3:
        return {
            "performance_passed": False,
            "problems": ["At least 3 repeats are required"],
            "scenarios": [],
        }
    for scenario in SCENARIOS:
        base, new = [], []
        for repeat in range(1, repeats + 1):
            name = f"{scenario}-{repeat}.json"
            try:
                b = json.loads((Path(baseline) / name).read_text())
                c = json.loads((Path(candidate) / name).read_text())
            except (OSError, ValueError) as exc:
                problems.append(f"{name}: {exc}")
                continue
            for key in MATCH_KEYS:
                if key not in b or key not in c or b[key] != c[key]:
                    problems.append(f"{name}: workload mismatch/missing {key}")
            for label, data in (("baseline", b), ("candidate", c)):
                if data.get("completed") != data.get("num_prompts") or data.get("failed", 0):
                    problems.append(f"{name}: {label} has incomplete/failed requests")
                for key in HIGHER_IS_BETTER + LOWER_IS_BETTER:
                    value = data.get(key)
                    if (
                        not isinstance(value, (int, float))
                        or not math.isfinite(value)
                        or value <= 0
                    ):
                        problems.append(f"{name}: {label} invalid/missing {key}")
            # Same requested lengths plus ignore_eos must produce the same token counts.
            for key in ("total_input_tokens", "total_output_tokens"):
                if not b.get(key) or b.get(key) != c.get(key):
                    problems.append(f"{name}: token counts differ or are missing ({key})")
            base.append(b)
            new.append(c)
        if len(base) != repeats:
            continue
        for metric in HIGHER_IS_BETTER + LOWER_IS_BETTER:
            values = [data.get(metric) for data in (*base, *new)]
            if any(
                not isinstance(v, (int, float)) or not math.isfinite(v) or v <= 0 for v in values
            ):
                continue
            try:
                bvalue = statistics.median(x[metric] for x in base)
                cvalue = statistics.median(x[metric] for x in new)
                ratio = cvalue / bvalue
            except (KeyError, TypeError, ZeroDivisionError):
                continue
            passed = ratio >= 1.0 if metric in HIGHER_IS_BETTER else ratio <= 1.0
            rows.append(
                {
                    "scenario": scenario,
                    "metric": metric,
                    "baseline": bvalue,
                    "candidate": cvalue,
                    "ratio": ratio,
                    "passed": passed,
                }
            )
            if not passed:
                problems.append(f"{scenario}: {metric} regressed (candidate/baseline={ratio:.4f})")
    return {
        "performance_passed": not problems,
        "problems": problems,
        "scenarios": rows,
        "quality_acceptance": "not_evaluated",
        "npu_correctness": "not_evaluated",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("baseline", type=Path)
    parser.add_argument("candidate", type=Path)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = compare(args.baseline, args.candidate, args.repeats)
    text = json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False)
    print(text)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text + "\n")
    return 0 if result["performance_passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
