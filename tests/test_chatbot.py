import pandas as pd
import pytest
import sklearn

from ml.evaluate import (choose_threshold, classifier_metrics, retrieval_metrics, run_evaluation,
                         score_questions)
from ml.predict import (FALLBACK_MESSAGE, IntentClassifier, Retriever, load_or_build_retriever,
                        source_fingerprint)


@pytest.fixture()
def retriever(tiny_faq_df):
    return Retriever(threshold=0.2).fit(tiny_faq_df)


def test_known_question_returns_expected_answer(retriever):
    result = retriever.answer("what is a banana")
    assert result["match_found"] and result["matched_question"] == "What is a banana?"
    assert result["answer"] == "A banana is a yellow fruit." and result["topic"] == "Fruit"
    assert 0.5 <= result["similarity_score"] <= 1.0      # exact wording: high, but below 1.0 because the answer view is blended in


def test_paraphrase_matches_right_entry(retriever):
    assert retriever.answer("Tell me about the Pomodoro technique")["topic"] == "Study Skills"


def test_unknown_question_falls_back(retriever):
    result = retriever.answer("Who painted the Mona Lisa?")
    assert not result["match_found"] and result["answer"] == FALLBACK_MESSAGE
    assert result["matched_question"] is None and result["topic"] is None


def test_unknown_words_reduce_score(retriever):
    # one known word + several unknown words must NOT look like a perfect match
    assert retriever.answer("banana quantum chromodynamics lattice")["similarity_score"] < 0.5


def test_empty_and_symbol_only_messages(retriever):
    for text in ["", "   ", "?!", "what is it"]:
        result = retriever.answer(text)
        assert not result["match_found"] and result["similarity_score"] == 0.0


def test_threshold_is_configurable(tiny_faq_df):
    strict = Retriever(threshold=0.99).fit(tiny_faq_df)
    assert not strict.answer("banana rocket")["match_found"]
    lenient = Retriever(threshold=0.01).fit(tiny_faq_df)
    assert lenient.answer("banana rocket")["match_found"]


def test_suggestion_given_for_near_miss(tiny_faq_df):
    r = Retriever(threshold=0.9, suggestion_threshold=0.1).fit(tiny_faq_df)
    out = r.answer("banana yellow")      # "yellow" is unknown, so coverage halves the score
    assert not out["match_found"] and out["suggestion"] == "What is a banana?"


def test_rank_returns_sorted_top_k(retriever):
    ranked = retriever.rank("rocket space", top_k=3)
    scores = [r["similarity"] for r in ranked]
    assert len(ranked) == 3 and scores == sorted(scores, reverse=True) and ranked[0]["topic"] == "Space"


def test_empty_dataset_cannot_be_indexed():
    with pytest.raises(ValueError):
        Retriever().fit(pd.DataFrame(columns=["question", "answer", "topic", "answer_type"]))


def test_save_and_load_artifact(retriever, tmp_path):
    path = retriever.save(tmp_path / "r.joblib")
    loaded = Retriever.load(path, threshold=0.2)
    assert loaded.answer("what is a rocket")["matched_question"] == "What is a rocket?"
    assert loaded.size == 4


def test_artifact_reused_then_rebuilt_when_csv_changes(tiny_faq_path, tiny_faq_df, tmp_path):
    artifact = tmp_path / "r.joblib"
    Retriever().fit(tiny_faq_df, source_sha256=source_fingerprint(tiny_faq_path)).save(artifact)
    _, source = load_or_build_retriever(tiny_faq_path, artifact)
    assert source == "artifact"
    extra = pd.concat([tiny_faq_df, pd.DataFrame([["What is a comet?", "An icy body.", "Space", "definition"]], columns=tiny_faq_df.columns)])
    extra.to_csv(tiny_faq_path, index=False)
    r, source = load_or_build_retriever(tiny_faq_path, artifact)
    assert source == "built" and r.size == 5


def test_corrupt_artifact_is_rebuilt(tiny_faq_path, tmp_path):
    bad = tmp_path / "bad.joblib"
    bad.write_bytes(b"not a model")
    r, source = load_or_build_retriever(tiny_faq_path, bad)
    assert source == "built" and r.size == 4


def test_topic_summary(retriever):
    names = [t["name"] for t in retriever.topic_summary()]
    assert names == sorted(names) and "Study Skills" in names


# ---- classifier ---------------------------------------------------------
@pytest.fixture()
def intent_data():
    rows = [("what is a banana fruit", "Fruit"), ("tell me about apples", "Fruit"), ("which fruit is sweet", "Fruit"),
            ("how do rockets fly", "Space"), ("what is a planet", "Space"), ("explain orbit of the moon", "Space")] * 3
    return pd.DataFrame(rows, columns=["text", "intent"]).drop_duplicates()


@pytest.mark.parametrize("kind", ["logreg", "naive_bayes"])
def test_classifier_trains_and_predicts(kind, intent_data, tmp_path):
    model = IntentClassifier(kind).fit(intent_data["text"], intent_data["intent"])
    assert model.predict(["a sweet fruit like banana", "moon and planet orbit"]) == ["Fruit", "Space"]
    one = model.predict_one("apples are fruit")
    assert one["intent"] == "Fruit" and 0 <= one["model_probability"] <= 1
    loaded = IntentClassifier.load(model.save(tmp_path / "c.joblib"))
    assert loaded.predict(["fruit"]) == model.predict(["fruit"]) and loaded.meta["sklearn_version"] == sklearn.__version__


def test_unknown_classifier_kind_rejected():
    with pytest.raises(ValueError):
        IntentClassifier("magic")


# ---- evaluation logic ---------------------------------------------------
@pytest.fixture()
def tiny_test_df():
    return pd.DataFrame([
        ["bananas kiwi", "Fruit", "What is a banana?", "yes", "dev", "easy"],
        ["rocket space vehicle", "Space", "What is a rocket?", "yes", "test", "easy"],
        ["bake some bread", "Cooking", "How do I bake bread?", "yes", "test", "easy"],
        ["how do I fix my car engine", "", "", "no", "test", "easy"],
        ["Pomodoro method", "Study Skills", "What is the Pomodoro technique?", "yes", "dev", "easy"],
        ["capital of France", "", "", "no", "dev", "easy"],
    ], columns=["question", "expected_topic", "expected_question", "in_scope", "split", "difficulty"])


def test_retrieval_metrics_counts(retriever, tiny_test_df):
    test = tiny_test_df[tiny_test_df.split == "test"]
    m = retrieval_metrics(score_questions(retriever, test["question"]), test, 0.2)
    assert m["questions_in_scope"] == 2 and m["questions_out_of_scope"] == 1
    assert m["top1_exact_question_rate"] == 1.0 and m["top3_exact_question_rate"] == 1.0
    assert m["no_match_correct_rejection_rate"] == 1.0 and m["false_match_count"] == 0
    assert m["overall_decision_accuracy"] == 1.0 and m["avg_ms_per_query"] >= 0


def test_false_match_and_miss_are_reported(retriever, tiny_test_df):
    test = tiny_test_df[tiny_test_df.split == "test"]
    scored = score_questions(retriever, test["question"])
    assert retrieval_metrics(scored, test, 0.0)["false_match_count"] == 1       # accepts everything
    strict = retrieval_metrics(scored, test, 1.01)                               # accepts nothing
    assert strict["in_scope_rejected_count"] == 2 and strict["correct_answer_returned_rate"] == 0.0


def test_threshold_selection_uses_dev_rows(retriever, tiny_test_df):
    dev = tiny_test_df[tiny_test_df.split == "dev"]
    best, table = choose_threshold(score_questions(retriever, dev["question"]), dev)
    assert best in [row["threshold"] for row in table] and len(table) == 16


def test_classifier_metrics_shape():
    m = classifier_metrics(["A", "A", "B", "B"], ["A", "B", "B", "B"], ["A", "B", "C"])
    assert m["accuracy"] == 0.75 and m["confusion_matrix"][0] == [1, 1, 0]
    assert m["labels_without_test_examples"] == ["C"] and "precision" in m["classification_report"]


def test_full_evaluation_runs_on_tiny_data(tiny_faq_df, intent_data, tiny_test_df):
    res = run_evaluation(tiny_faq_df, intent_data, tiny_test_df, threshold=0.2)
    assert res["data"]["test_split_rows"] == 3 and "classifier_logreg" in res
    assert res["leakage"]["knowledge_base_vs_test"]["exact"] == []
