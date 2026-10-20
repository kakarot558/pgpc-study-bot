"""Where is the knowledge base thin?   python -m ml.coverage   (writes results/coverage_report.md)

A "concept" is a distinct answer; rows that share an answer are alternative phrasings of it.
Concepts with a single phrasing are the likeliest to be missed when a student words things differently.
"""
import sys

import pandas as pd

import config
from ml.preprocess import DatasetError, load_clean_faq

THIN_SUBJECT = 10


def coverage(faq: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for topic, g in faq.groupby("topic"):
        per_concept = g.groupby("answer").size()
        rows.append({"subject": topic, "questions": len(g), "concepts": len(per_concept),
                     "single_phrasing": int((per_concept == 1).sum()),
                     "avg_phrasings": round(len(g) / len(per_concept), 2)})
    return pd.DataFrame(rows).sort_values(["concepts", "subject"]).reset_index(drop=True)


def build_report(cov: pd.DataFrame) -> str:
    lines = ["# Knowledge-base coverage", "", f"{int(cov['questions'].sum())} questions, {int(cov['concepts'].sum())} concepts, "
             f"{len(cov)} subjects. Concepts with one phrasing: {int(cov['single_phrasing'].sum())} "
             f"({cov['single_phrasing'].sum() / cov['concepts'].sum():.0%}).", "",
             "| Subject | Questions | Concepts | Single-phrasing concepts | Avg phrasings |", "|---|---|---|---|---|"]
    for _, r in cov.iterrows():
        flag = " (thin)" if r["concepts"] < THIN_SUBJECT else ""
        lines.append(f"| {r['subject']}{flag} | {r['questions']} | {r['concepts']} | {r['single_phrasing']} | {r['avg_phrasings']} |")
    lines += ["", f"Subjects marked thin have fewer than {THIN_SUBJECT} concepts. Add concepts and extra phrasings "
              "(see `python -m ml.add_phrasings`) starting from the top of this table."]
    return "\n".join(lines)


def main() -> int:
    try:
        faq, _ = load_clean_faq()
    except DatasetError as exc:
        print(f"ERROR: {exc}")
        return 1
    report = build_report(coverage(faq))
    config.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    (config.RESULTS_DIR / "coverage_report.md").write_text(report, encoding="utf-8")
    print(report)
    return 0


if __name__ == "__main__":
    sys.exit(main())
