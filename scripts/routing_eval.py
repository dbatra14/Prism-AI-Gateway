#!/usr/bin/env python3
"""Grade the smart router against the labelled eval set.

    python scripts/routing_eval.py            # summary + any failures
    python scripts/routing_eval.py --verbose  # every case, with the reason

The bar set by the brief is "must beat a length-only baseline", so this reports
both, plus a per-case breakdown. The trap cases (short-but-hard,
long-but-trivial) are broken out separately because they're the ones that
distinguish a real classifier from a word counter — a router can look fine on
aggregate accuracy while failing every trap.

The labels in data/routing_eval.jsonl are used here and nowhere else. The
classifier never sees them.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from app.routing.classifier import classify, length_only_baseline  # noqa: E402

EVAL_PATH = pathlib.Path(__file__).resolve().parent.parent / "data" / "routing_eval.jsonl"


def load_cases(path: pathlib.Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--verbose", action="store_true", help="print every case")
    parser.add_argument("--eval-file", type=pathlib.Path, default=EVAL_PATH)
    args = parser.parse_args()

    cases = load_cases(args.eval_file)
    rows, prism_correct, baseline_correct = [], 0, 0

    for case in cases:
        expected = case["expected_tier"]
        decision = classify(case["prompt"])
        baseline = length_only_baseline(case["prompt"])
        ok = decision.tier == expected
        prism_correct += ok
        baseline_correct += baseline == expected
        rows.append((case, expected, decision, baseline, ok))

    total = len(cases)
    traps = [r for r in rows if "trap" in (r[0].get("note") or "").lower()]

    print(f"\nRouting eval — {total} labelled cases from {args.eval_file.name}\n")
    if args.verbose:
        for case, expected, decision, baseline, ok in rows:
            mark = "ok  " if ok else "MISS"
            print(f"  {mark} {case['id']}  expected={expected:<5} prism={decision.tier:<5} baseline={baseline:<5}")
            print(f"       {decision.reason}")
            print(f"       {case['prompt'][:90]}{'...' if len(case['prompt']) > 90 else ''}\n")
    else:
        for case, expected, decision, baseline, ok in rows:
            if not ok:
                print(f"  MISS {case['id']}  expected={expected} got={decision.tier}")
                print(f"       {decision.reason}")
                print(f"       {case['prompt'][:90]}...\n")

    trap_correct = sum(1 for r in traps if r[4])
    trap_baseline = sum(1 for r in traps if r[3] == r[1])

    print("=" * 62)
    print(f"  Prism classifier   {prism_correct}/{total}  ({prism_correct / total:.0%})")
    print(f"  Length-only        {baseline_correct}/{total}  ({baseline_correct / total:.0%})   <- baseline to beat")
    if traps:
        print(f"\n  On trap cases only ({len(traps)} short-but-hard / long-but-trivial):")
        print(f"    Prism classifier {trap_correct}/{len(traps)}")
        print(f"    Length-only      {trap_baseline}/{len(traps)}")
    print("=" * 62)

    if prism_correct <= baseline_correct:
        print("\nFAIL: the classifier does not beat the length-only baseline.")
        return 1
    print(f"\nPASS: beats the length-only baseline by {prism_correct - baseline_correct} cases.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
