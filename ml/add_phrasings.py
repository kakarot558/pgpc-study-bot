"""Add alternative phrasings of existing questions without retyping the answer.

Input CSV columns:  alias,question
   alias    = the new way a student might ask it
   question = the exact stored question it should map to

  python -m ml.add_phrasings --input data/new_phrasings.csv

Each accepted alias becomes a new row with the same answer, topic and review status. An alias is
REFUSED if it already exists, if the target question is unknown, or if it matches a question in
any data/test_questions*.csv file (adding it would contaminate the evaluation).
"""
import argparse
import glob
import shutil
import sys
from pathlib import Path

import pandas as pd

import config
from ml.preprocess import DatasetError, normalize_whitespace, question_key, read_csv_safely


def protected_keys(data_dir=None) -> set:
    keys = set()
    for path in glob.glob(str(Path(data_dir or config.DATA_DIR) / "test_questions*.csv")):
        try:
            keys |= set(read_csv_safely(path)["question"].map(question_key))
        except (DatasetError, KeyError):
            continue
    return keys


def add_phrasings(faq: pd.DataFrame, aliases: pd.DataFrame, protected: set = frozenset()):
    for col in ("alias", "question"):
        if col not in aliases.columns:
            raise DatasetError(f"Phrasing file is missing column '{col}'.")
    faq = faq.copy()
    by_key = {question_key(q): i for i, q in zip(faq.index, faq["question"])}
    existing = set(by_key)
    new_rows, report = [], {"added": 0, "refused": []}
    for _, row in aliases.iterrows():
        alias = normalize_whitespace(row["alias"])
        key = question_key(alias)
        target = by_key.get(question_key(row["question"]))
        if not alias:
            reason = "empty alias"
        elif target is None:
            reason = "target question not found"
        elif key in existing:
            reason = "already in the knowledge base"
        elif key in protected:
            reason = "matches a test question (would contaminate evaluation)"
        else:
            reason = None
        if reason:
            report["refused"].append((alias, reason))
            continue
        base = faq.loc[target].copy()
        base["question"] = alias
        new_rows.append(base)
        existing.add(key)
        report["added"] += 1
    if new_rows:
        faq = pd.concat([faq, pd.DataFrame(new_rows)], ignore_index=True)
    return faq, report


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input", required=True)
    ap.add_argument("--faq", default=str(config.FAQ_PATH))
    args = ap.parse_args(argv)
    try:
        faq = read_csv_safely(args.faq)
        new, rep = add_phrasings(faq, read_csv_safely(args.input), protected_keys())
    except DatasetError as exc:
        print(f"ERROR: {exc}")
        return 1
    if rep["added"]:
        path = Path(args.faq)
        shutil.copyfile(path, path.with_suffix(path.suffix + ".bak"))
        new.to_csv(path, index=False, encoding="utf-8")
    print(f"Added {rep['added']} phrasing(s).")
    for alias, reason in rep["refused"]:
        print(f"  refused: {alias!r} - {reason}")
    if rep["added"]:
        print("Now run: python -m ml.preprocess && python -m ml.train && python -m ml.evaluate")
    return 0


if __name__ == "__main__":
    sys.exit(main())
