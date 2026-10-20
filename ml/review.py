"""Instructor review workflow for the knowledge base.

  python -m ml.review stats                         # how much of each subject is reviewed
  python -m ml.review export --out data/review_todo.csv [--subject "Operating Systems"]
        -> a CSV for the instructor: fill `decision` with ok or fix (and `corrected_answer` for fix)
  python -m ml.review apply data/review_todo.csv    # mark reviewed / apply corrections (keeps a .bak backup)

Rows that share the same answer text are the same concept (alternative phrasings), so one decision
covers all of them.
"""
import argparse
import shutil
import sys
from pathlib import Path

import pandas as pd

import config
from ml.preprocess import DatasetError, normalize_whitespace, question_key, read_csv_safely

REVIEW_COLUMNS = ["question", "topic", "current_answer", "decision", "corrected_answer", "notes"]


def _with_reviewed(faq: pd.DataFrame) -> pd.DataFrame:
    faq = faq.copy()
    if "reviewed" not in faq.columns:
        faq["reviewed"] = ""
    return faq


def stats(faq: pd.DataFrame) -> pd.DataFrame:
    faq = _with_reviewed(faq)
    g = faq.groupby("topic").agg(total=("question", "size"), reviewed=("reviewed", lambda s: int((s == "yes").sum())))
    g["percent"] = (g["reviewed"] / g["total"] * 100).round(0).astype(int)
    return g.sort_values("percent")


def export_todo(faq: pd.DataFrame, subject=None, include_reviewed: bool = False) -> pd.DataFrame:
    """One row per distinct answer (first phrasing shown), unreviewed ones only by default."""
    faq = _with_reviewed(faq)
    if subject:
        faq = faq[faq["topic"] == subject]
    if not include_reviewed:
        faq = faq[faq["reviewed"] != "yes"]
    faq = faq.drop_duplicates(subset="answer", keep="first")
    out = pd.DataFrame({"question": faq["question"], "topic": faq["topic"], "current_answer": faq["answer"],
                        "decision": "", "corrected_answer": "", "notes": ""})
    return out.reset_index(drop=True)


def apply_review(faq: pd.DataFrame, review: pd.DataFrame):
    """Return (new_faq, report). decision 'ok' marks reviewed; 'fix' replaces the answer first."""
    missing = [c for c in ("question", "decision") if c not in review.columns]
    if missing:
        raise DatasetError(f"Review file is missing column(s): {', '.join(missing)}")
    faq = _with_reviewed(faq).reset_index(drop=True)
    keys = faq["question"].map(question_key)
    report = {"marked_ok": 0, "fixed": 0, "skipped_blank": 0, "unknown_questions": [], "fix_without_text": []}
    for _, row in review.iterrows():
        decision = normalize_whitespace(row.get("decision", "")).lower()
        if decision not in ("ok", "fix"):
            report["skipped_blank"] += 1
            continue
        hits = faq.index[keys == question_key(row["question"])]
        if len(hits) == 0:
            report["unknown_questions"].append(row["question"])
            continue
        old_answer = faq.at[hits[0], "answer"]
        same_concept = faq.index[faq["answer"] == old_answer]
        if decision == "fix":
            new_answer = normalize_whitespace(row.get("corrected_answer", ""))
            if not new_answer:
                report["fix_without_text"].append(row["question"])
                continue
            faq.loc[same_concept, "answer"] = new_answer
            report["fixed"] += len(same_concept)
        else:
            report["marked_ok"] += len(same_concept)
        faq.loc[same_concept, "reviewed"] = "yes"
    return faq, report


def _backup_and_write(df: pd.DataFrame, path: Path) -> None:
    if path.exists():
        shutil.copyfile(path, path.with_suffix(path.suffix + ".bak"))
    df.to_csv(path, index=False, encoding="utf-8")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("stats")
    e = sub.add_parser("export"); e.add_argument("--out", default="data/review_todo.csv"); e.add_argument("--subject")
    e.add_argument("--all", action="store_true", help="include already reviewed answers")
    a = sub.add_parser("apply"); a.add_argument("file")
    ap.add_argument("--faq", default=str(config.FAQ_PATH))
    args = ap.parse_args(argv)
    try:
        faq = read_csv_safely(args.faq)
        if args.cmd == "stats":
            s = stats(faq)
            print(s.to_string())
            print(f"\nOverall: {int(s['reviewed'].sum())} of {int(s['total'].sum())} answers reviewed")
        elif args.cmd == "export":
            todo = export_todo(faq, args.subject, args.all)
            Path(args.out).parent.mkdir(parents=True, exist_ok=True)
            todo.to_csv(args.out, index=False, encoding="utf-8")
            print(f"Wrote {len(todo)} answers to review in {args.out}. Fill 'decision' with ok or fix.")
        else:
            new, rep = apply_review(faq, read_csv_safely(args.file))
            print(f"Marked ok: {rep['marked_ok']}  fixed: {rep['fixed']}  skipped (blank decision): {rep['skipped_blank']}")
            for key in ("unknown_questions", "fix_without_text"):
                if rep[key]:
                    print(f"WARNING {key}: {rep[key]}")
            if rep["marked_ok"] or rep["fixed"]:
                _backup_and_write(new, Path(args.faq))
                print(f"Saved {args.faq} (backup: {args.faq}.bak). Now run: python -m ml.preprocess && python -m ml.train")
            else:
                print("Nothing to apply: fill the 'decision' column with ok or fix first. The knowledge base was not changed.")
    except DatasetError as exc:
        print(f"ERROR: {exc}")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
