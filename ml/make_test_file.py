"""Turn real student questions into a proper held-out test file.

1. Collect questions (for example `python -m ml.feedback_report --export data/feedback/needs_label.csv`
   or a classroom survey).
2. In a CSV with columns  question, expected_question  fill expected_question with the EXACT stored
   question that answers it ('|' between acceptable alternatives), or leave it blank / write none
   if the bot should refuse (out of scope or not in the knowledge base). Optional column: difficulty.
3. python -m ml.make_test_file --input labelled.csv --output data/test_questions_real.csv
4. python -m ml.evaluate --test-file data/test_questions_real.csv --tag real

Questions that already exist in the knowledge base or the intent training data are dropped, because
testing on training text would inflate the score.
"""
import argparse
import sys

import pandas as pd

import config
from ml.evaluate import find_leakage
from ml.preprocess import (DatasetError, load_clean_faq, load_clean_intent, normalize_whitespace,
                           read_csv_safely, question_key)

TEST_COLUMNS = ["question", "expected_topic", "expected_question", "in_scope", "difficulty", "split"]


def build_test_df(labelled: pd.DataFrame, faq: pd.DataFrame, intents=None, split: str = "test"):
    """Return (test_df, report)."""
    if "question" not in labelled.columns or "expected_question" not in labelled.columns:
        raise DatasetError("Labelled file needs columns: question, expected_question")
    topic_of = {question_key(q): t for q, t in zip(faq["question"], faq["topic"])}
    canonical = {question_key(q): q for q in faq["question"]}
    rows, report, seen = [], {"kept": 0, "dropped_duplicates": [], "dropped_leaky": [], "unlabelled_blank_means_out_of_scope": 0}, set()
    unknown = []
    for _, r in labelled.iterrows():
        q = normalize_whitespace(r["question"])
        if not q or question_key(q) in seen:
            report["dropped_duplicates"].append(q)
            continue
        seen.add(question_key(q))
        raw = normalize_whitespace(r["expected_question"])
        if raw.lower() in ("", "none", "n/a"):
            expected, topic, in_scope = "", "", "no"
            report["unlabelled_blank_means_out_of_scope"] += 1
        else:
            parts = [normalize_whitespace(p) for p in raw.split("|") if p.strip()]
            bad = [p for p in parts if question_key(p) not in canonical]
            if bad:
                unknown += bad
                continue
            parts = [canonical[question_key(p)] for p in parts]
            expected, topic, in_scope = "|".join(parts), topic_of[question_key(parts[0])], "yes"
        rows.append([q, topic, expected, in_scope, normalize_whitespace(r.get("difficulty", "")) or "unspecified", split])
    if unknown:
        raise DatasetError("Unknown expected_question value(s) (must match a stored question exactly): " + "; ".join(unknown[:5]))
    df = pd.DataFrame(rows, columns=TEST_COLUMNS)
    reference = list(faq["question"]) + (list(intents["text"]) if intents is not None else [])
    leaks = find_leakage(reference, df["question"].tolist(), near_threshold=1.01)["exact"]
    leaky = {a for a, _ in leaks}
    report["dropped_leaky"] = sorted(leaky)
    df = df[~df["question"].isin(leaky)].reset_index(drop=True)
    report["kept"] = int(len(df))
    return df, report


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input", required=True)
    ap.add_argument("--output", default=str(config.DATA_DIR / "test_questions_real.csv"))
    args = ap.parse_args(argv)
    try:
        faq, _ = load_clean_faq()
        intents, _ = load_clean_intent()
        df, rep = build_test_df(read_csv_safely(args.input), faq, intents)
    except DatasetError as exc:
        print(f"ERROR: {exc}")
        return 1
    df.to_csv(args.output, index=False, encoding="utf-8")
    print(f"Wrote {rep['kept']} questions to {args.output} "
          f"({int((df.in_scope == 'yes').sum())} in-scope, {int((df.in_scope == 'no').sum())} out-of-scope).")
    for key in ("dropped_duplicates", "dropped_leaky"):
        if rep[key]:
            print(f"  {key}: {rep[key]}")
    print("Next: python -m ml.evaluate --test-file", args.output, "--tag real")
    return 0


if __name__ == "__main__":
    sys.exit(main())
