"""Measure how well the two components work on HELD-OUT data.

Run from the project root (after `python -m ml.train`):   python -m ml.evaluate

Files used
  data/test_questions.csv   questions written separately from the training data, each with a
                            manually assigned expected topic and expected stored question.
                            Rows are marked split = dev (used ONLY to choose the similarity
                            threshold) or test (used ONLY for the reported numbers).
Writes
  results/evaluation_report.md, results/evaluation_results.json, results/confusion_matrix.csv
"""
import json
import statistics
import sys
import time
from datetime import datetime, timezone

import pandas as pd
import sklearn
from sklearn.metrics import (accuracy_score, classification_report, confusion_matrix,
                             precision_recall_fscore_support)

import config
from ml.predict import IntentClassifier, Retriever
from ml.preprocess import (DatasetError, load_clean_faq, combine_training_data, load_clean_intent, normalize_surface, normalize_text,
                           read_csv_safely)

TEST_COLUMNS = ["question", "expected_topic", "expected_question", "in_scope", "split"]


# ------------------------------------------------------------------ loading
def load_test_set(path=None) -> pd.DataFrame:
    df = read_csv_safely(path or config.TEST_PATH)
    missing = [c for c in TEST_COLUMNS if c not in df.columns]
    if missing:
        raise DatasetError(f"Test set is missing column(s): {', '.join(missing)}")
    if "difficulty" not in df.columns:
        df["difficulty"] = "unspecified"
    df["in_scope"] = df["in_scope"].str.strip().str.lower()
    df["split"] = df["split"].str.strip().str.lower()
    if not df["in_scope"].isin(["yes", "no"]).all():
        raise DatasetError("test set column 'in_scope' must contain only yes/no.")
    if not df["split"].isin(["dev", "test"]).all():
        raise DatasetError("test set column 'split' must contain only dev/test.")
    return df.reset_index(drop=True)


# ------------------------------------------------------------ leakage checks
def _tokens(text):
    return frozenset(normalize_surface(text).split())


def find_leakage(reference_texts, test_texts, near_threshold: float = 0.8) -> dict:
    """Compare test texts against reference (training / index) texts.

    exact: same text after normalisation.  near: token-set Jaccard similarity >= near_threshold.
    """
    ref = [(t, _tokens(t)) for t in reference_texts]
    exact, near = [], []
    for text in test_texts:
        tok = _tokens(text)
        for ref_text, ref_tok in ref:
            if tok == ref_tok:
                exact.append((text, ref_text))
            elif tok and ref_tok:
                jac = len(tok & ref_tok) / len(tok | ref_tok)
                if jac >= near_threshold:
                    near.append((text, ref_text, round(jac, 2)))
    return {"exact": exact, "near": near}


def check_leakage(faq, intents, test_df) -> dict:
    qs = test_df["question"].tolist()
    return {
        "intent_training_vs_test": find_leakage(intents["text"], qs),
        "knowledge_base_vs_test": find_leakage(faq["question"], qs),
        "duplicates_inside_test_set": int(test_df["question"].map(normalize_surface).duplicated().sum()),
    }


# ---------------------------------------------------------------- retrieval
def score_questions(retriever: Retriever, questions, top_k: int = 3) -> list:
    """Rank every question once; remember how long each lookup took."""
    out = []
    for q in questions:
        start = time.perf_counter()
        ranked = retriever.rank(q, top_k=top_k)
        out.append({"question": q, "ranked": ranked, "ms": (time.perf_counter() - start) * 1000})
    return out


def _expected_set(value: str) -> set:
    return {v.strip() for v in str(value).split("|") if v.strip()}


def retrieval_metrics(scored, test_df, threshold: float) -> dict:
    """Metrics for one split of the test set at one threshold. `scored` aligns with test_df rows."""
    n_in = n_top1 = n_topic = n_top3 = n_answered_ok = n_missed = n_rescued = 0
    wrong_answers, missed, false_matches, by_diff = [], [], [], {}
    n_out = n_rejected = 0
    for item, (_, row) in zip(scored, test_df.iterrows()):
        top = item["ranked"][0]
        answered = top["similarity"] >= threshold
        if row["in_scope"] == "yes":
            n_in += 1
            expected = _expected_set(row["expected_question"])
            top1_ok = top["question"] in expected
            n_top1 += top1_ok
            n_topic += top["topic"] == row["expected_topic"]
            n_top3 += any(r["question"] in expected for r in item["ranked"])
            d = by_diff.setdefault(row["difficulty"], {"n": 0, "top1": 0, "topic": 0})
            d["n"] += 1
            d["top1"] += top1_ok
            d["topic"] += top["topic"] == row["expected_topic"]
            if answered and top1_ok:
                n_answered_ok += 1
            elif answered:
                wrong_answers.append({"question": row["question"], "returned": top["question"],
                                      "expected": sorted(expected), "similarity": round(top["similarity"], 3)})
            else:
                n_missed += 1
                # the bot declines but offers the top candidates as "Did you mean...?" buttons
                if any(r["question"] in expected and r["similarity"] >= config.SUGGESTION_THRESHOLD for r in item["ranked"]):
                    n_rescued += 1
                missed.append({"question": row["question"], "closest": top["question"],
                               "similarity": round(top["similarity"], 3)})
        else:
            n_out += 1
            if answered:
                false_matches.append({"question": row["question"], "returned": top["question"],
                                      "similarity": round(top["similarity"], 3)})
            else:
                n_rejected += 1
    total = n_in + n_out
    answered_total = n_answered_ok + len(wrong_answers) + len(false_matches)
    safe = lambda a, b: round(a / b, 4) if b else None
    return {
        "threshold": round(threshold, 3), "questions_in_scope": n_in, "questions_out_of_scope": n_out,
        "top1_exact_question_rate": safe(n_top1, n_in), "top1_topic_rate": safe(n_topic, n_in),
        "top3_exact_question_rate": safe(n_top3, n_in),
        "correct_answer_returned_rate": safe(n_answered_ok, n_in),
        "in_scope_rejected_count": n_missed, "rejected_but_offered_correct_count": n_rescued,
        "helped_within_one_click_rate": safe(n_answered_ok + n_rescued, n_in), "wrong_answer_count": len(wrong_answers),
        "no_match_correct_rejection_rate": safe(n_rejected, n_out), "false_match_count": len(false_matches),
        "answer_precision": safe(n_answered_ok, answered_total),
        "overall_decision_accuracy": safe(n_answered_ok + n_rejected, total),
        "by_difficulty": {k: {"n": v["n"], "top1_exact": safe(v["top1"], v["n"]), "top1_topic": safe(v["topic"], v["n"])}
                          for k, v in sorted(by_diff.items())},
        "avg_ms_per_query": round(statistics.mean(i["ms"] for i in scored), 3) if scored else None,
        "wrong_answers": wrong_answers, "in_scope_rejected": missed, "false_matches": false_matches,
    }


FALSE_MATCH_COST = 2.0   # answering an out-of-scope question is judged twice as bad as other errors


def decision_cost(m: dict) -> float:
    """Lower is better: missed in-scope questions + wrong answers + 2 x false matches."""
    return m["in_scope_rejected_count"] + m["wrong_answer_count"] + FALSE_MATCH_COST * m["false_match_count"]


def choose_threshold(scored_dev, dev_df, candidates=None) -> tuple:
    """Pick the threshold with the lowest decision cost on DEV rows only. Returns (threshold, table)."""
    candidates = candidates or [round(0.05 * k, 2) for k in range(1, 17)]
    table = []
    for t in candidates:
        m = retrieval_metrics(scored_dev, dev_df, t)
        table.append({"threshold": t, "decision_accuracy": m["overall_decision_accuracy"], "cost": decision_cost(m),
                      "false_matches": m["false_match_count"], "wrong_answers": m["wrong_answer_count"],
                      "in_scope_rejected": m["in_scope_rejected_count"]})
    lowest = min(r["cost"] for r in table)
    plateau = sorted(r["threshold"] for r in table if r["cost"] == lowest)
    return plateau[len(plateau) // 2], table      # middle of the cheapest plateau


# ---------------------------------------------------------------- classifier
def classifier_metrics(y_true, y_pred, labels) -> dict:
    present = [l for l in labels if l in set(y_true)]      # macro average only over topics that have test examples
    p, r, f, _ = precision_recall_fscore_support(y_true, y_pred, labels=present, average="macro", zero_division=0)
    wp, wr, wf, _ = precision_recall_fscore_support(y_true, y_pred, labels=labels, average="weighted", zero_division=0)
    return {
        "n": len(y_true), "accuracy": round(accuracy_score(y_true, y_pred), 4),
        "macro_precision": round(p, 4), "macro_recall": round(r, 4), "macro_f1": round(f, 4),
        "weighted_precision": round(wp, 4), "weighted_recall": round(wr, 4), "weighted_f1": round(wf, 4),
        "labels": list(labels), "labels_without_test_examples": [l for l in labels if l not in present],
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=labels).tolist(),
        "classification_report": classification_report(y_true, y_pred, labels=labels, zero_division=0),
    }


# -------------------------------------------------------------------- report
LIMITATIONS = """\
## How to read these numbers (limitations)

* **Small, author-written test set.** The questions were written by the project author as
  paraphrases of stored questions plus unrelated questions. Real students may phrase things
  differently, so true performance can be lower (or higher). Percentages from a few dozen
  questions are noisy: one question changes them by several points.
* **Top-1 exact-question rate** counts a hit only if the closest stored question is one the author
  accepted. Near-synonym questions (for example "Define AI") are listed as alternatives, but some
  judgements are subjective.
* **Top-1 topic rate** is more forgiving: a wrong answer in the right topic still counts.
* **Top-3 rate** shows whether the right answer is among the top three candidates; the app
  currently shows only the best one.
* **No-match detection** depends on the threshold. A higher threshold rejects more unrelated
  questions but also rejects more valid paraphrases. The threshold was selected on the `dev` rows (minimising missed + wrong + 2 x false-match errors)
  and the numbers below use only the `test` rows.
* **Cosine similarity is not accuracy or probability.** It only measures word overlap between two
  texts after normalisation. It cannot detect meaning that uses different words.
* **Classifier metrics** (accuracy, precision, recall, F1) describe topic prediction on this small
  test set only. Topics overlap (for example Data Science vs Model Evaluation), the label set is
  imbalanced, and macro averages weight every topic equally even when a topic has few test rows.
* **The classifier does not answer questions.** It predicts a category; retrieval returns a prepared answer.
* **Timing** was measured on one machine and includes only the matching step, not the web request.
"""


def _pct(v):
    return "n/a" if v is None else f"{v * 100:.1f}%"


def build_report(res: dict) -> str:
    r, c = res["retrieval_test_at_configured_threshold"], res["classifier_logreg"]
    r2 = res["retrieval_test_at_dev_threshold"]
    L = [f"# StudyBuddy evaluation report", "",
         f"Generated: {res['generated_at']}  |  scikit-learn {res['sklearn_version']}  |  seed {config.RANDOM_SEED}", ""]
    if res.get("scoring_overrides"):
        shown = ", ".join(f"{k}={v}" for k, v in res["scoring_overrides"].items())
        L += [f"> **NOTE: scoring settings were overridden by environment variables ({shown}).** These results are not "
              "comparable with the shipped results. Remove the variables (new PowerShell window) and run again.", ""]
    L += [
         "## 1. Data used", "",
         f"* Knowledge base (retrieval index): {res['data']['faq_rows']} questions",
         f"* Intent training file: {res['data']['intent_rows']} labeled examples",
         f"* Held-out questions: {res['data']['test_rows']} total = {res['data']['dev_rows']} `dev` (threshold selection only) + {res['data']['test_split_rows']} `test` (reported)",
         f"* Test rows: {r['questions_in_scope']} in-scope, {r['questions_out_of_scope']} out-of-scope (the bot should say 'no match')", "",
         "### Leakage checks", ""]
    lk = res["leakage"]
    L += [f"* Intent training vs test questions: {len(lk['intent_training_vs_test']['exact'])} exact and {len(lk['intent_training_vs_test']['near'])} near-duplicate pairs (token overlap >= 0.8)",
          f"* Knowledge base vs test questions: {len(lk['knowledge_base_vs_test']['exact'])} exact and {len(lk['knowledge_base_vs_test']['near'])} near-duplicate pairs "
          "(test questions are paraphrases of stored questions by design, so near matches here are expected but exact copies are not)",
          f"* Duplicate questions inside the test set: {lk['duplicates_inside_test_set']}", "",
          "## 2. Answer retrieval (TF-IDF + cosine similarity)", "",
          f"Configured threshold: **{r['threshold']}**. Test-split questions only.", "",
          "| Metric | Value |", "|---|---|",
          f"| Top-1 exact-question rate (in-scope, ignoring threshold) | {_pct(r['top1_exact_question_rate'])} |",
          f"| Top-1 topic rate (in-scope, ignoring threshold) | {_pct(r['top1_topic_rate'])} |",
          f"| Top-3 exact-question rate | {_pct(r['top3_exact_question_rate'])} |",
          f"| Correct answer actually returned (in-scope, with threshold) | {_pct(r['correct_answer_returned_rate'])} |",
          f"| In-scope questions rejected as 'no match' | {r['in_scope_rejected_count']} |",
          f"| ...of those, correct question offered as a 'Did you mean' button | {r['rejected_but_offered_correct_count']} |",
          f"| Helped within one click (answered correctly, or correct question offered) | {_pct(r['helped_within_one_click_rate'])} |",
          f"| Wrong answers returned for in-scope questions | {r['wrong_answer_count']} |",
          f"| Out-of-scope questions correctly rejected | {_pct(r['no_match_correct_rejection_rate'])} |",
          f"| False matches (out-of-scope question answered) | {r['false_match_count']} |",
          f"| Precision of returned answers | {_pct(r['answer_precision'])} |",
          f"| Overall decision accuracy (right answer or right rejection) | {_pct(r['overall_decision_accuracy'])} |",
          f"| Average matching time | {r['avg_ms_per_query']} ms |", "",
          "By difficulty (top-1, ignoring threshold):", "", "| Difficulty | n | exact question | topic |", "|---|---|---|---|"]
    for k, v in r["by_difficulty"].items():
        L.append(f"| {k} | {v['n']} | {_pct(v['top1_exact'])} | {_pct(v['top1_topic'])} |")
    if res["dev_threshold_table"]:
        L += ["", f"Threshold selected on dev rows: **{res['dev_selected_threshold']}**. Test-split results at that threshold: "
              f"decision accuracy {_pct(r2['overall_decision_accuracy'])}, false matches {r2['false_match_count']}, "
              f"in-scope rejected {r2['in_scope_rejected_count']}, wrong answers {r2['wrong_answer_count']}.", "",
              "### Threshold sweep on dev rows", "", "| Threshold | Cost | Decision accuracy | False matches | Wrong answers | In-scope rejected |", "|---|---|---|---|---|---|"]
        for row in res["dev_threshold_table"]:
            L.append(f"| {row['threshold']} | {row['cost']:g} | {_pct(row['decision_accuracy'])} | {row['false_matches']} | {row['wrong_answers']} | {row['in_scope_rejected']} |")
    else:
        L += ["", "This file has no `dev` rows, so nothing was tuned here: the configured threshold was used as is."]
    for title, key, fmt in [("Wrong answers returned", "wrong_answers", "- \"{question}\" -> returned \"{returned}\" (expected {expected}, similarity {similarity})"),
                            ("In-scope questions rejected (missed)", "in_scope_rejected", "- \"{question}\" -> closest \"{closest}\" (similarity {similarity})"),
                            ("False matches on out-of-scope questions", "false_matches", "- \"{question}\" -> returned \"{returned}\" (similarity {similarity})")]:
        L += ["", f"### {title} (test split, configured threshold)", ""]
        L += [fmt.format(**x) for x in r[key]] or ["- none"]
    if c is None:
        L += ["", "## 3. Intent classification", "", "Skipped: this test file has no in-scope questions, so topic accuracy cannot be measured.", "", LIMITATIONS]
        return "\n".join(L)
    L += ["", "## 3. Intent classification (supervised, topic labels)", "",
          f"Trained on `intent_training.csv` plus the stored knowledge-base questions (each labeled with its topic); scored on the {c['n']} in-scope test-split questions.", "",
          "| Model | Accuracy | Macro precision | Macro recall | Macro F1 |", "|---|---|---|---|---|"]
    for name, m in (("Logistic regression (intent file + stored questions)", c),
                    ("Multinomial Naive Bayes (intent file + stored questions)", res["classifier_naive_bayes"]),
                    ("Logistic regression (intent file only)", res["classifier_logreg_intent_file_only"])):
        L.append(f"| {name} | {_pct(m['accuracy'])} | {_pct(m['macro_precision'])} | {_pct(m['macro_recall'])} | {_pct(m['macro_f1'])} |")
    L += ["", f"Topics with no test questions (excluded from macro averages): {', '.join(c['labels_without_test_examples']) or 'none'}.", "",
          "Per-class report (logistic regression, intent file + stored questions):", "", "```", c["classification_report"].rstrip(), "```", "",
          "Confusion matrix (rows = true topic, columns = predicted topic):", "", "```"]
    width = max(len(x) for x in c["labels"])
    L.append(" " * width + "  " + " ".join(f"{i:>3}" for i in range(len(c["labels"]))))
    for lab, row in zip(c["labels"], c["confusion_matrix"]):
        L.append(f"{lab:<{width}}  " + " ".join(f"{v:>3}" for v in row))
    L += ["```", "", "Column numbers follow the label order: " + ", ".join(f"{i}={l}" for i, l in enumerate(c["labels"])) + ".", "",
          "## 4. Retrieval versus classification on the same questions", "",
          f"* Topic of the closest stored question (retrieval, no threshold): {_pct(res['topic_accuracy_retrieval'])}",
          f"* Topic predicted by the logistic-regression classifier: {_pct(c['accuracy'])}", "",
          "These answer different questions. Retrieval returns a prepared answer and can refuse when nothing is close. "
          "The classifier always predicts one of the known topics (it has no 'unknown' class), so it cannot detect out-of-scope questions.", "",
          LIMITATIONS]
    return "\n".join(L)


# ---------------------------------------------------------------------- main
def run_evaluation(faq=None, intents=None, test_df=None, threshold=config.SIMILARITY_THRESHOLD) -> dict:
    faq = faq if faq is not None else load_clean_faq()[0]
    intents = intents if intents is not None else load_clean_intent()[0]
    test_df = test_df if test_df is not None else load_test_set()
    retriever = Retriever(threshold=threshold).fit(faq)
    dev, test = test_df[test_df.split == "dev"], test_df[test_df.split == "test"]
    scored_dev = score_questions(retriever, dev["question"])
    scored_test = score_questions(retriever, test["question"])
    best_t, table = choose_threshold(scored_dev, dev) if len(dev) else (threshold, [])

    in_test = test[test.in_scope == "yes"]
    y_true = in_test["expected_topic"].tolist()
    labels = sorted(intents["intent"].unique())
    if y_true:
        combined = combine_training_data(intents, faq)
        models = {k: IntentClassifier(k).fit(combined["text"], combined["intent"]) for k in ("logreg", "naive_bayes")}
        intent_only = IntentClassifier("logreg").fit(intents["text"], intents["intent"])
        cm = {k: classifier_metrics(y_true, m.predict(in_test["question"]), labels) for k, m in models.items()}
        cm_intent_only = classifier_metrics(y_true, intent_only.predict(in_test["question"]), labels)
    else:                                   # e.g. a pilot file that contains only refusals: nothing to classify
        cm, cm_intent_only = {"logreg": None, "naive_bayes": None}, None
    retr_topics = [i["ranked"][0]["topic"] for i, (_, row) in zip(scored_test, test.iterrows()) if row["in_scope"] == "yes"]
    return {
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "sklearn_version": sklearn.__version__, "scoring_overrides": config.active_overrides(),
        "data": {"faq_rows": len(faq), "intent_rows": len(intents), "test_rows": len(test_df),
                 "dev_rows": len(dev), "test_split_rows": len(test)},
        "leakage": check_leakage(faq, intents, test_df),
        "dev_selected_threshold": best_t, "dev_threshold_table": table,
        "retrieval_test_at_configured_threshold": retrieval_metrics(scored_test, test, threshold),
        "retrieval_test_at_dev_threshold": retrieval_metrics(scored_test, test, best_t),
        "classifier_logreg": cm["logreg"], "classifier_naive_bayes": cm["naive_bayes"],
        "classifier_logreg_intent_file_only": cm_intent_only,
        "topic_accuracy_retrieval": round(sum(a == b for a, b in zip(retr_topics, y_true)) / len(y_true), 4) if y_true else None,
    }


def main(argv=None) -> int:
    import argparse
    ap = argparse.ArgumentParser(description="Evaluate retrieval and the intent classifier.")
    ap.add_argument("--test-file", default=str(config.TEST_PATH), help="CSV of held-out questions")
    ap.add_argument("--tag", default="", help="suffix for output file names, e.g. v2")
    args = ap.parse_args(argv)
    tag = f"_{args.tag}" if args.tag else ""
    try:
        results = run_evaluation(test_df=load_test_set(args.test_file))
    except DatasetError as exc:
        print(f"ERROR: {exc}")
        return 1
    config.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    (config.RESULTS_DIR / f"evaluation_results{tag}.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    (config.RESULTS_DIR / f"evaluation_report{tag}.md").write_text(build_report(results), encoding="utf-8")
    c = results["classifier_logreg"]
    if c is not None:
        pd.DataFrame(c["confusion_matrix"], index=c["labels"], columns=c["labels"]).to_csv(config.RESULTS_DIR / f"confusion_matrix{tag}.csv")
    print(build_report(results))
    print(f"\nSaved to {config.RESULTS_DIR}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
