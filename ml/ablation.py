"""Measure what each improvement contributes (synonyms, character n-grams).

Run from the project root:   python -m ml.ablation
For every configuration the similarity threshold is chosen on the dev rows of
data/test_questions.csv only, then results are reported on the `test` rows of
that file (test set A) and on data/test_questions_v2.csv (test set B).
Writes results/ablation.md
"""
import sys
from pathlib import Path

import config
from ml.evaluate import choose_threshold, load_test_set, retrieval_metrics, score_questions
from ml.predict import Retriever
from ml.preprocess import load_clean_faq, reload_synonyms

CONFIGS = [("words only (first version)", False, 0.0, 0.0), ("+ synonyms only", True, 0.0, 0.0),
           ("+ character n-grams only (weight 0.35)", False, 0.35, 0.0),
           ("+ answer text only", False, 0.0, config.ANSWER_WEIGHT),
           ("current settings (synonyms + answer text)", True, config.CHAR_WEIGHT, config.ANSWER_WEIGHT),
           ("everything incl. characters (0.2)", True, 0.2, config.ANSWER_WEIGHT)]
NO_FILE = Path("__no_synonyms__.csv")      # a path that does not exist = synonyms switched off


def fmt(v):
    return "n/a" if v is None else f"{v * 100:.1f}%"


def run() -> str:
    set_a = load_test_set(config.TEST_PATH)
    set_b = load_test_set(config.DATA_DIR / "test_questions_v2.csv")
    dev, test_a = set_a[set_a.split == "dev"], set_a[set_a.split == "test"]
    test_b = set_b[set_b.split == "test"]
    lines = ["# Ablation: what does each improvement add?", "",
             "Threshold per row chosen on dev rows only. Test set A = `test` rows of test_questions.csv "
             f"({len(test_a)} questions). Test set B = test_questions_v2.csv ({len(test_b)} questions, never used to choose settings).",
             "", "| Configuration | Threshold | A top-1 | A top-3 | A decision acc. | A false matches | B top-1 | B top-3 | B decision acc. | B false matches |",
             "|---|---|---|---|---|---|---|---|---|---|"]
    try:
        for name, use_syn, weight, a_weight in CONFIGS:
            reload_synonyms(None if use_syn else NO_FILE)
            faq, _ = load_clean_faq()
            retr = Retriever(char_weight=weight, answer_weight=a_weight).fit(faq)
            thr, _ = choose_threshold(score_questions(retr, dev["question"]), dev)
            cells = []
            for tdf in (test_a, test_b):
                m = retrieval_metrics(score_questions(retr, tdf["question"]), tdf, thr)
                cells += [fmt(m["top1_exact_question_rate"]), fmt(m["top3_exact_question_rate"]),
                          fmt(m["overall_decision_accuracy"]), str(m["false_match_count"])]
            lines.append(f"| {name} | {thr} | " + " | ".join(cells) + " |")
    finally:
        reload_synonyms()                      # always restore the normal synonym table
    lines += ["", "Caveats: both test sets are small (a single question moves a figure by about 2 points). "
              "The synonym entries were written after seeing which questions failed, so the synonym gain is optimistic; "
              "the blend weights and thresholds were chosen on dev rows only."]
    return "\n".join(lines)


def main() -> int:
    text = run()
    config.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    (config.RESULTS_DIR / "ablation.md").write_text(text, encoding="utf-8")
    print(text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
