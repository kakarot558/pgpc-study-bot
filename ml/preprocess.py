"""Dataset loading, cleaning and text normalisation.

Two jobs live here:

1. CLEAN THE CSV FILES (``clean_faq`` / ``clean_intent``): check columns, drop empty rows,
   tidy whitespace, standardise topic names, remove exact duplicates and report problems.
2. NORMALISE TEXT FOR MATCHING (``normalize_text``): lowercase, remove punctuation and filler
   words, and apply a very light plural stemmer. The SAME function is used for stored
   questions and for student questions, which is what makes TF-IDF matching fair.

Run from the project root:   python -m ml.preprocess
"""
import json
import re
import sys
from collections import Counter
from pathlib import Path

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

import config

FAQ_REQUIRED = ["question", "answer", "topic"]      # answer_type is filled in if absent
FAQ_COLUMNS = ["question", "answer", "topic", "answer_type"]
FAQ_OPTIONAL = ["reviewed"]       # "yes" once an instructor has checked the answer (kept if present)
INTENT_REQUIRED = ["text", "intent"]

# Small hand-made list of filler words. (We keep words like "define", "example",
# "difference" because they carry meaning in study questions.)
STOP_WORDS = frozenset(
    """a an the is are was were be been am do does did of to in on for with and or what which who
    whom how why when where can could would should will shall i me my we our you your it its this
    that these those there about at by from as into tell give please
    ano ang ng sa ba po ito yan yung mga paano pano bakit pwede mo ko naman lang
    define explain describe""".split()
)


class DatasetError(ValueError):
    """Raised when a dataset file is missing, unreadable or has the wrong columns."""


# ---------------------------------------------------------------- text helpers
def normalize_whitespace(value) -> str:
    """Collapse runs of spaces/newlines/tabs and drop control characters."""
    if value is None:
        return ""
    text = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]", "", str(value))
    return " ".join(text.split())


def light_stem(token: str) -> str:
    """Turn simple plurals into singulars (labels -> label, classes -> class).

    This is deliberately tiny. It is applied to both stored and student text,
    so any odd result (e.g. "bias" -> "bia") is at least consistent.
    """
    if len(token) <= 3:
        return token
    if token.endswith("ies"):
        return token[:-3] + "y"
    if token.endswith("sses"):
        return token[:-2]
    if token.endswith(("ss", "us", "is")):
        return token
    if token.endswith("s"):
        return token[:-1]
    return token


_SYNONYMS = None


def load_synonyms(path=None) -> list:
    """Read data/synonyms.csv -> [(compiled_pattern, replacement), ...] longest phrase first.

    A broken or missing file is not fatal: matching simply runs without synonyms.
    """
    path = Path(path or config.SYNONYMS_PATH)
    if not path.exists():
        return []
    try:
        df = read_csv_safely(path)
        _require_columns(df, ["phrase", "replacement"], "Synonyms file")
    except DatasetError:
        return []
    pairs = []
    for phrase, repl in zip(df["phrase"], df["replacement"]):
        phrase = " ".join(re.findall(r"[a-z0-9]+", phrase.lower()))
        repl = " ".join(re.findall(r"[a-z0-9]+", repl.lower()))
        if phrase and repl and phrase != repl:
            pairs.append((re.compile(r"(?<= )" + re.escape(phrase) + r"(?= )"), repl))
    return sorted(pairs, key=lambda p: -len(p[0].pattern))


def reload_synonyms(path=None) -> list:
    """(Re)load the synonym table. Called automatically on first use."""
    global _SYNONYMS
    _SYNONYMS = load_synonyms(path)
    return _SYNONYMS


# "deadlock in OS", "inheritance in OOP", "paging in operating system": the trailing subject name is
# context, not the thing being asked about. It is dropped from the QUESTION (never from stored text)
# when other content words remain, so it cannot hijack the match.
_SUBJECT_CONTEXT = re.compile(
    r"\b(?:in|of|for|under|using|with|sa)\s+(?:the\s+|an?\s+)?"
    r"(?:os|operating systems?|dbms|databases?|database management systems?|oops?|object oriented programming|"
    r"data structures?|dsa|computer networks?|networking|networks?|software engineering|compilers?|compiler design|"
    r"digital logic|computer architecture|computer organi[sz]ation|automata|theory of computation|"
    r"machine learning|ml|artificial intelligence|ai|java|python|c\+\+|c)(?![\w+])", re.IGNORECASE)


def strip_subject_context(text) -> str:
    """Remove a trailing 'in/of/for <subject>' phrase, but only if meaningful words remain."""
    original = normalize_whitespace(text)
    stripped = _SUBJECT_CONTEXT.sub(" ", original)
    return normalize_whitespace(stripped) if normalize_text(stripped) else original


def normalize_surface(text) -> str:
    """Like normalize_text but WITHOUT synonym expansion (used to compare wording in leakage checks)."""
    tokens = re.findall(r"[a-z0-9]+", normalize_whitespace(text).lower())
    return " ".join(light_stem(t) for t in tokens if t not in STOP_WORDS)


def normalize_text(text) -> str:
    """Lowercase, keep letters/digits only, expand synonyms, drop filler words, apply light_stem."""
    global _SYNONYMS
    if _SYNONYMS is None:
        reload_synonyms()
    tokens = re.findall(r"[a-z0-9]+", normalize_whitespace(text).lower())
    if _SYNONYMS and tokens:
        padded = " " + " ".join(tokens) + " "
        for phrase, repl in _SYNONYMS:
            padded = phrase.sub(repl, padded)
        tokens = padded.split()
    return " ".join(light_stem(t) for t in tokens if t not in STOP_WORDS)


def question_key(text) -> str:
    """Key used to decide whether two questions are *identical* (not merely similar)."""
    return normalize_whitespace(text).lower().rstrip("?.! ")


def make_vectorizer() -> TfidfVectorizer:
    """The one TF-IDF configuration shared by retrieval and the intent classifier."""
    return TfidfVectorizer(
        preprocessor=normalize_text,
        token_pattern=r"\S+",
        lowercase=False,
        ngram_range=config.NGRAM_RANGE,
        sublinear_tf=True,
    )


def make_char_vectorizer() -> TfidfVectorizer:
    """Character n-gram TF-IDF: tolerant of word endings and small spelling differences."""
    return TfidfVectorizer(preprocessor=normalize_text, analyzer="char_wb",
                           ngram_range=config.CHAR_NGRAM_RANGE, sublinear_tf=True)


def infer_answer_type(question: str) -> str:
    """Guess an answer_type from the wording of the question (documented heuristic).

    Only used when the CSV leaves answer_type blank. Review the result by hand.
    """
    q = normalize_whitespace(question).lower()
    if "difference" in q or " different from " in q:
        return "comparison"
    if "example" in q:
        return "example"
    if q.startswith("what should i"):
        return "advice"
    if q.startswith(("what is", "what are", "define", "explain")) and "this project" not in q:
        return "definition"
    if q.startswith("how"):
        return "advice"
    if q.startswith("why"):
        return "explanation"
    return "faq"


# ------------------------------------------------------------------ file I/O
def read_csv_safely(path) -> pd.DataFrame:
    """Read a CSV as text, tolerating a UTF-8 BOM and Windows line endings."""
    path = Path(path)
    if not path.exists():
        raise DatasetError(f"Dataset file not found: {path}")
    try:
        df = pd.read_csv(path, encoding="utf-8-sig", dtype=str, keep_default_na=False)
    except (pd.errors.EmptyDataError, pd.errors.ParserError, UnicodeDecodeError) as exc:
        raise DatasetError(f"Could not read {path.name}: {exc}") from exc
    df.columns = [normalize_whitespace(c).lower() for c in df.columns]
    return df


def _require_columns(df: pd.DataFrame, required, name: str) -> None:
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise DatasetError(f"{name} is missing required column(s): {', '.join(missing)}")


def _standardise_labels(series: pd.Series):
    """Make spelling variants of one label identical ("machine learning", "Machine_Learning").

    The most common spelling wins. Returns (new_series, number_of_values_changed).
    """
    cleaned = series.map(lambda v: normalize_whitespace(str(v).replace("_", " ")))
    keys = cleaned.str.lower()
    best = {}
    for key, group in cleaned.groupby(keys):
        best[key] = Counter(group).most_common(1)[0][0]
    result = keys.map(best)
    return result, int((result != series).sum())


def find_near_duplicates(questions, threshold: float = 0.9) -> list:
    """List pairs of DIFFERENT questions whose TF-IDF cosine similarity >= threshold."""
    questions = list(questions)
    if len(questions) < 2:
        return []
    try:
        matrix = make_vectorizer().fit_transform(questions)
    except ValueError:        # every question was empty after normalisation
        return []
    sims = cosine_similarity(matrix)
    pairs = []
    for i in range(len(questions)):
        for j in range(i + 1, len(questions)):
            if sims[i, j] >= threshold:
                pairs.append({"question_a": questions[i], "question_b": questions[j],
                              "similarity": round(float(sims[i, j]), 3)})
    return pairs


# --------------------------------------------------------------- FAQ cleaning
def clean_faq(df: pd.DataFrame):
    """Clean the knowledge-base table. Returns (clean_df, report_dict)."""
    _require_columns(df, FAQ_REQUIRED, "FAQ dataset")
    df = df.copy()
    if "answer_type" not in df.columns:
        df["answer_type"] = ""
    cols = FAQ_COLUMNS + [c for c in FAQ_OPTIONAL if c in df.columns]
    df = df[cols].fillna("")
    report = {"rows_before": int(len(df))}

    for col in cols:
        df[col] = df[col].map(normalize_whitespace)
    if "reviewed" in df.columns:
        df["reviewed"] = df["reviewed"].str.lower().map(lambda v: "yes" if v in {"yes", "y", "true", "1"} else "")

    # 1) empty questions / answers
    keep = (df["question"] != "") & (df["answer"] != "")
    report["removed_empty"] = int((~keep).sum())
    df = df[keep].copy()

    # 2) topics: fill blanks, unify spelling
    df.loc[df["topic"] == "", "topic"] = "General"
    df["topic"], report["topic_values_standardised"] = _standardise_labels(df["topic"])

    # 3) answer_type: keep given values, infer blanks
    blank = df["answer_type"] == ""
    report["answer_types_inferred"] = int(blank.sum())
    df.loc[blank, "answer_type"] = df.loc[blank, "question"].map(infer_answer_type)
    df["answer_type"] = df["answer_type"].str.lower()

    # 4) exact duplicates (same question AND same answer)
    df["_qkey"] = df["question"].map(question_key)
    df["_akey"] = df["answer"].str.lower()
    dup = df.duplicated(subset=["_qkey", "_akey"], keep="first")
    report["removed_exact_duplicates"] = int(dup.sum())
    df = df[~dup]

    # 5) same question, different answer -> conflict (keep the first, tell the user)
    conflict_mask = df.duplicated(subset=["_qkey"], keep="first")
    conflicts = []
    for key in df.loc[conflict_mask, "_qkey"].unique():
        rows = df[df["_qkey"] == key]
        conflicts.append({"question": rows.iloc[0]["question"], "answers_found": int(len(rows)),
                          "kept_answer_starts_with": rows.iloc[0]["answer"][:60]})
    report["conflicting_answers"] = conflicts
    report["removed_conflicting_rows"] = int(conflict_mask.sum())
    df = df[~conflict_mask]

    df = df[cols].reset_index(drop=True)

    # 6) similar-but-different questions are only REPORTED, never deleted
    report["suspicious_near_duplicates"] = find_near_duplicates(df["question"])
    report["rows_after"] = int(len(df))
    report["topic_counts"] = df["topic"].value_counts().to_dict()
    report["topics_outside_bscs_list"] = sorted(set(df["topic"]) - set(config.BSCS_SUBJECTS)) if config.BSCS_SUBJECTS else []
    return df, report


def clean_intent(df: pd.DataFrame):
    """Clean the labeled intent table (columns: text, intent)."""
    _require_columns(df, INTENT_REQUIRED, "Intent dataset")
    df = df[INTENT_REQUIRED].copy().fillna("")
    report = {"rows_before": int(len(df))}
    for col in INTENT_REQUIRED:
        df[col] = df[col].map(normalize_whitespace)
    keep = (df["text"] != "") & (df["intent"] != "")
    report["removed_empty"] = int((~keep).sum())
    df = df[keep].copy()
    df["intent"], report["label_values_standardised"] = _standardise_labels(df["intent"])
    df["_key"] = df["text"].map(question_key)
    dup = df.duplicated(subset=["_key", "intent"], keep="first")
    report["removed_exact_duplicates"] = int(dup.sum())
    df = df[~dup]
    clash = df.duplicated(subset=["_key"], keep="first")   # same text, different label
    report["removed_conflicting_labels"] = int(clash.sum())
    df = df[~clash][INTENT_REQUIRED].reset_index(drop=True)
    report["rows_after"] = int(len(df))
    report["class_counts"] = df["intent"].value_counts().to_dict()
    return df, report


def combine_training_data(intents: pd.DataFrame, faq: pd.DataFrame) -> pd.DataFrame:
    """Labeled examples for the classifier: intent_training.csv rows PLUS every stored question
    (its topic is a ready-made label). Exact repeats are dropped. Held-out test questions are
    never part of either source (ml.evaluate checks this)."""
    extra = faq[["question", "topic"]].rename(columns={"question": "text", "topic": "intent"})
    both = pd.concat([intents[INTENT_REQUIRED], extra], ignore_index=True)
    both = both[~both["text"].map(question_key).duplicated(keep="first")]
    return both.reset_index(drop=True)


def load_clean_faq(path=None):
    """Read + clean the knowledge base. Raises DatasetError if it is unusable."""
    df, report = clean_faq(read_csv_safely(path or config.FAQ_PATH))
    if df.empty:
        raise DatasetError("The FAQ dataset has no usable rows after cleaning.")
    return df, report


def load_clean_intent(path=None):
    df, report = clean_intent(read_csv_safely(path or config.INTENT_PATH))
    if df.empty:
        raise DatasetError("The intent dataset has no usable rows after cleaning.")
    return df, report


def save_processed(df: pd.DataFrame, filename: str, out_dir=None) -> Path:
    out_dir = Path(out_dir or config.PROCESSED_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / filename
    df.to_csv(path, index=False, encoding="utf-8")
    return path


def run_preprocessing(faq_path=None, intent_path=None, out_dir=None) -> dict:
    """Clean both datasets, save them under data/processed/, return the reports."""
    out_dir = Path(out_dir or config.PROCESSED_DIR)
    faq, faq_report = load_clean_faq(faq_path)
    intents, intent_report = load_clean_intent(intent_path)
    save_processed(faq, "study_faq_clean.csv", out_dir)
    save_processed(intents, "intent_training_clean.csv", out_dir)
    reports = {"faq": faq_report, "intent": intent_report}
    (out_dir / "preprocess_report.json").write_text(json.dumps(reports, indent=2), encoding="utf-8")
    return reports


def main() -> int:
    try:
        reports = run_preprocessing()
    except DatasetError as exc:
        print(f"ERROR: {exc}")
        return 1
    for name, rep in reports.items():
        print(f"\n=== {name.upper()} dataset ===")
        print(f"Rows before cleaning: {rep['rows_before']}")
        print(f"Rows after cleaning:  {rep['rows_after']}")
        for key, value in rep.items():
            if key.startswith("removed") and value:
                print(f"  {key.replace('_', ' ')}: {value}")
        if rep.get("topics_outside_bscs_list"):
            print("  WARNING - topics not in config.BSCS_SUBJECTS (check spelling, or this is outside BSCS):",
                  rep["topics_outside_bscs_list"])
        if rep.get("conflicting_answers"):
            print("  CONFLICTS (same question, different answers):", rep["conflicting_answers"])
        if rep.get("suspicious_near_duplicates"):
            print("  Similar questions to review (kept, not deleted):")
            for pair in rep["suspicious_near_duplicates"]:
                print(f"    {pair['similarity']}: '{pair['question_a']}'  <->  '{pair['question_b']}'")
    print(f"\nCleaned files saved in {config.PROCESSED_DIR}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
