"""Fail loudly if a change makes the chatbot noticeably worse.

  python -m ml.regression_check                    # compare with results/baseline_metrics.json
  python -m ml.regression_check --update-baseline  # accept the current numbers as the new baseline

Runs the retrieval evaluation on both held-out sets (A: test rows of test_questions.csv, B:
test_questions_v2.csv). The small tolerances allow normal noise but catch real damage, for example
from a bad data edit. Used by the GitHub Actions workflow in .github/workflows/tests.yml.
"""
import argparse
import json
import sys

import config
from ml.evaluate import load_test_set, retrieval_metrics, score_questions
from ml.predict import Retriever
from ml.preprocess import DatasetError, load_clean_faq

BASELINE_PATH = config.RESULTS_DIR / "baseline_metrics.json"
# metric -> maximum allowed DROP (rates) ; counts: maximum allowed INCREASE
RATE_TOLERANCE = {"top1_exact_question_rate": 0.04, "top3_exact_question_rate": 0.03,
                  "answer_precision": 0.04, "helped_within_one_click_rate": 0.04}
COUNT_TOLERANCE = {"false_match_count": 2, "wrong_answer_count": 3}


def collect(faq=None, test_sets=None) -> dict:
    faq = faq if faq is not None else load_clean_faq()[0]
    retriever = Retriever().fit(faq)
    sets = test_sets or {"A": config.TEST_PATH, "B": config.DATA_DIR / "test_questions_v2.csv"}
    out = {}
    for name, path in sets.items():
        df = load_test_set(path)
        df = df[df.split == "test"]
        m = retrieval_metrics(score_questions(retriever, df["question"]), df, retriever.threshold)
        out[name] = {k: m[k] for k in list(RATE_TOLERANCE) + list(COUNT_TOLERANCE)}
    return out


def compare(current: dict, baseline: dict) -> list:
    """Return a list of human-readable problems (empty list = fine)."""
    problems = []
    for name, base in baseline.items():
        cur = current.get(name)
        if cur is None:
            problems.append(f"set {name}: missing from current run")
            continue
        for metric, tol in RATE_TOLERANCE.items():
            if base[metric] is not None and cur[metric] is not None and base[metric] - cur[metric] > tol:
                problems.append(f"set {name}: {metric} fell {base[metric]:.3f} -> {cur[metric]:.3f} (allowed drop {tol})")
        for metric, tol in COUNT_TOLERANCE.items():
            if cur[metric] - base[metric] > tol:
                problems.append(f"set {name}: {metric} rose {base[metric]} -> {cur[metric]} (allowed rise {tol})")
    return problems


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--update-baseline", action="store_true")
    args = ap.parse_args(argv)
    overrides = config.active_overrides()
    if overrides:
        print("Refusing to run: scoring settings are overridden by environment variables:")
        for k, v in overrides.items():
            print(f"  {k}={v}")
        print("The baseline must describe the default settings. Open a NEW PowerShell window (or remove them, e.g. "
              "Remove-Item Env:STUDYBUDDY_THRESHOLD) and run this again.")
        return 2
    try:
        current = collect()
    except DatasetError as exc:
        print(f"ERROR: {exc}")
        return 1
    if args.update_baseline or not BASELINE_PATH.exists():
        BASELINE_PATH.parent.mkdir(parents=True, exist_ok=True)
        BASELINE_PATH.write_text(json.dumps(current, indent=2), encoding="utf-8")
        print(f"Baseline saved to {BASELINE_PATH}")
        return 0
    problems = compare(current, json.loads(BASELINE_PATH.read_text(encoding="utf-8")))
    if problems:
        print("REGRESSION DETECTED:")
        for p in problems:
            print("  -", p)
        return 1
    print("No regression: all metrics are within tolerance of the baseline.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
