"""Convert an external question/answer CSV into StudyBuddy's FAQ format.

Example (PowerShell, from the project root):
  python -m ml.import_dataset --input data/raw/my_download.csv --question-col question ^
      --answer-col answer --topic "Science" --source "SciQ" --license "CC BY-NC 3.0" ^
      --output data/raw/sciq_converted.csv

What it does: renames the columns, adds the topic, cleans the rows with the normal
preprocessing code, records where the data came from in data/DATA_SOURCES.md, and writes
a CSV you can REVIEW and then append to data/study_faq.csv.

It does NOT check licences for you. Read the dataset's licence first (non-commercial and
share-alike terms differ), and only import data you are allowed to use and redistribute.
"""
import argparse
import sys
from datetime import date
from pathlib import Path

import pandas as pd

import config
from ml.preprocess import DatasetError, clean_faq, read_csv_safely


def convert(df: pd.DataFrame, question_col: str, answer_col: str, topic: str, topic_col: str = None):
    """Return (clean_df, report) in the FAQ layout."""
    for col in [question_col, answer_col] + ([topic_col] if topic_col else []):
        if col not in df.columns:
            raise DatasetError(f"Column '{col}' not found. Available columns: {', '.join(df.columns)}")
    out = pd.DataFrame({"question": df[question_col], "answer": df[answer_col],
                        "topic": df[topic_col] if topic_col else topic, "answer_type": ""})
    return clean_faq(out)


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--input", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--question-col", required=True)
    p.add_argument("--answer-col", required=True)
    p.add_argument("--topic", default="General", help="topic used for every row")
    p.add_argument("--topic-col", help="use this column as the topic instead")
    p.add_argument("--source", required=True, help="name/URL of the dataset")
    p.add_argument("--license", required=True, help="licence you checked, e.g. 'CC BY 4.0'")
    args = p.parse_args(argv)
    try:
        df, report = convert(read_csv_safely(args.input), args.question_col, args.answer_col, args.topic, args.topic_col)
    except DatasetError as exc:
        print(f"ERROR: {exc}")
        return 1
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.output, index=False)
    log = config.DATA_DIR / "DATA_SOURCES.md"
    with log.open("a", encoding="utf-8") as fh:
        fh.write(f"- {date.today()}: `{args.output}` from {args.source} (licence: {args.license}); "
                 f"{report['rows_before']} rows read, {report['rows_after']} kept after cleaning.\n")
    print(f"Wrote {report['rows_after']} rows to {args.output} (from {report['rows_before']}). Review before merging.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
