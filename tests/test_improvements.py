"""Tests for synonyms, character n-gram matching, related questions and confidence labels."""
import pandas as pd
import pytest

import config
from ml.evaluate import load_test_set, score_questions
from ml.predict import Retriever, load_or_build_retriever, source_fingerprint
from ml.preprocess import (load_clean_faq, load_synonyms, normalize_surface, normalize_text, reload_synonyms)

DATA = config.DATA_DIR


@pytest.fixture(autouse=True)
def restore_synonyms():
    yield
    reload_synonyms()          # whatever a test did, go back to data/synonyms.csv


@pytest.fixture()
def study_df():
    rows = [("How do I stop procrastination?", "Start with five minutes of work.", "Study Skills", "advice"),
            ("What is machine learning?", "Learning patterns from data.", "Machine Learning", "definition"),
            ("What is overfitting?", "Memorising the training data.", "Model Evaluation", "definition"),
            ("What is a tokenization step?", "Splitting text into tokens.", "Text Processing", "definition")]
    return pd.DataFrame(rows, columns=["question", "answer", "topic", "answer_type"])


# ---- synonyms ------------------------------------------------------------
def test_default_synonyms_expand_abbreviations():
    assert normalize_text("What is ML?") == normalize_text("What is machine learning?")
    assert normalize_text("Explain NLP") == normalize_text("Explain natural language processing")
    assert normalize_text("What is a data set?") == normalize_text("What is a dataset?")


def test_surface_normaliser_ignores_synonyms():
    assert normalize_surface("What is ML?") != normalize_surface("What is machine learning?")


def test_custom_synonym_file(tmp_path):
    path = tmp_path / "syn.csv"
    path.write_text("phrase,replacement\ncpu,processor\nbig data,dataset\n", encoding="utf-8")
    reload_synonyms(path)
    assert normalize_text("my CPU is fast") == "my processor fast".replace("my ", "")      # stop words removed
    assert normalize_text("big data") == "dataset"
    assert normalize_text("ML") == "ml"                                                   # old table no longer applies


@pytest.mark.parametrize("content", ["", "wrong,columns\na,b\n", "phrase,replacement\n,\n"])
def test_bad_synonym_file_is_ignored_not_fatal(tmp_path, content):
    path = tmp_path / "syn.csv"
    path.write_text(content, encoding="utf-8")
    assert load_synonyms(path) == []
    assert load_synonyms(tmp_path / "missing.csv") == []


def test_repeated_abbreviation_is_expanded_each_time():
    assert normalize_text("ml ml") == normalize_text("machine learning machine learning")


def test_fingerprint_changes_when_synonyms_change(tiny_faq_path, tmp_path):
    a = tmp_path / "a.csv"; b = tmp_path / "b.csv"
    a.write_text("phrase,replacement\nx,y\n", encoding="utf-8"); b.write_text("phrase,replacement\nx,z\n", encoding="utf-8")
    assert source_fingerprint(tiny_faq_path, a) != source_fingerprint(tiny_faq_path, b)
    assert source_fingerprint(tiny_faq_path, a) == source_fingerprint(tiny_faq_path, a)


# ---- character n-grams ---------------------------------------------------
def test_char_ngrams_bridge_word_endings(study_df):
    words_only = Retriever(threshold=0.15, char_weight=0.0, answer_weight=0.0).fit(study_df)
    blended = Retriever(threshold=0.15, char_weight=0.65, answer_weight=0.0).fit(study_df)
    q = "how do I stop procrastinating"
    assert blended.rank(q, 1)[0]["topic"] == "Study Skills"
    assert blended.rank(q, 1)[0]["similarity"] > words_only.rank(q, 1)[0]["similarity"]
    assert blended.answer("what is overfitted")["topic"] == "Model Evaluation"


def test_char_blend_does_not_make_nonsense_match(study_df):
    r = Retriever(threshold=0.15, char_weight=0.65, answer_weight=0.0).fit(study_df)
    for text in ["who painted the mona lisa", "zzzz qqqq xxxx", "boiling point of water"]:
        assert not r.answer(text)["match_found"], text


def test_scores_stay_between_0_and_1(study_df):
    r = Retriever(char_weight=0.4, answer_weight=0.4).fit(study_df)
    for q in ["what is machine learning", "overfitting overfitting overfitting", "?", "a" * 300]:
        for item in r.rank(q, 4):
            assert 0.0 <= item["similarity"] <= 1.0 + 1e-9


def test_saved_artifact_keeps_char_index(study_df, tmp_path):
    path = Retriever().fit(study_df).save(tmp_path / "r.joblib")
    loaded = Retriever.load(path, threshold=0.15, char_weight=0.65, answer_weight=0.0)
    assert loaded.answer("stop procrastinating")["topic"] == "Study Skills"


# ---- answer-text view -----------------------------------------------------
def test_answer_text_helps_when_question_wording_differs():
    df = pd.DataFrame([("What is a rocket?", "A rocket is a vehicle that travels to space.", "Space", "definition"),
                       ("What is a banana?", "A banana is a yellow fruit.", "Fruit", "definition")],
                      columns=["question", "answer", "topic", "answer_type"])
    q = "which vehicle travels to space"
    without = Retriever(threshold=0.3, char_weight=0.0, answer_weight=0.0).fit(df).answer(q)
    with_answers = Retriever(threshold=0.3, char_weight=0.0, answer_weight=0.6).fit(df).answer(q)
    assert not without["match_found"] and with_answers["match_found"] and with_answers["topic"] == "Space"


@pytest.mark.parametrize("cw,aw", [(-0.1, 0.0), (0.0, -0.1), (0.7, 0.5), (1.1, 0.0)])
def test_invalid_blend_weights_rejected(cw, aw):
    with pytest.raises(ValueError):
        Retriever(char_weight=cw, answer_weight=aw)


# ---- related questions and confidence -------------------------------------
def test_related_questions_and_confidence(study_df):
    r = Retriever(threshold=0.15, char_weight=0.2, answer_weight=0.5).fit(study_df)
    out = r.answer("what is machine learning")
    assert out["match_found"] and out["confidence"] == "high"
    assert out["matched_question"] not in out["related_questions"]
    weak = r.answer("machine banana kiwi")
    assert weak["confidence"] in (None, "low") and isinstance(weak["related_questions"], list)


def test_fallback_has_no_confidence(study_df):
    out = Retriever(threshold=0.15).fit(study_df).answer("who painted the mona lisa")
    assert out["confidence"] is None and out["match_found"] is False


def test_chat_route_exposes_new_fields(client):
    data = client.post("/chat", json={"message": "What is a banana?"}).get_json()
    assert data["confidence"] == "high" and isinstance(data["related_questions"], list)


# ---- regression guards on the real data ------------------------------------
def test_every_stored_question_retrieves_itself():
    faq, _ = load_clean_faq()
    r = Retriever().fit(faq)
    for q in faq["question"]:
        out = r.answer(q)
        assert out["match_found"] and out["similarity_score"] >= config.SIMILARITY_THRESHOLD + 0.1, q
        assert normalize_text(out["matched_question"]) == normalize_text(q), q


@pytest.mark.parametrize("name", ["test_questions.csv", "test_questions_v2.csv"])
def test_retrieval_quality_floor_on_heldout_sets(name):
    """Coarse guard against regressions; not a claim of performance (see results/)."""
    faq, _ = load_clean_faq()
    r = Retriever().fit(faq)
    t = load_test_set(DATA / name)
    t = t[(t.split == "test") & (t.in_scope == "yes")]
    scored = score_questions(r, t["question"])
    top3 = sum(any(x["question"] in set(row.expected_question.split("|")) for x in it["ranked"])
               for it, (_, row) in zip(scored, t.iterrows())) / len(t)
    top1_topic = sum(it["ranked"][0]["topic"] == row.expected_topic for it, (_, row) in zip(scored, t.iterrows())) / len(t)
    assert top3 >= 0.80 and top1_topic >= 0.75


def test_evaluate_cli_writes_tagged_reports(tmp_path, monkeypatch):
    import ml.evaluate as ev
    monkeypatch.setattr(config, "RESULTS_DIR", tmp_path)
    assert ev.main(["--test-file", str(DATA / "test_questions_v2.csv"), "--tag", "t"]) == 0
    assert (tmp_path / "evaluation_report_t.md").exists() and (tmp_path / "evaluation_results_t.json").exists()


def test_ablation_runs_and_restores_synonyms(tmp_path, monkeypatch):
    import ml.ablation as ab
    monkeypatch.setattr(config, "RESULTS_DIR", tmp_path)
    assert ab.main() == 0 and "current settings" in (tmp_path / "ablation.md").read_text(encoding="utf-8")
    assert normalize_text("ML") == normalize_text("machine learning")


# ---- subject-context stripping, compound words, classic questions ----------
from ml.preprocess import strip_subject_context  # noqa: E402


@pytest.mark.parametrize("raw,expected", [
    ("what is a deadlock in OS", "what is a deadlock"), ("paging in operating system", "paging"),
    ("inheritance in OOP", "inheritance"), ("normalisation in dbms", "normalisation"),
    ("what is a pointer in C", "what is a pointer"), ("what is a pointer in c++", "what is a pointer"),
    ("what is NAT in a network", "what is NAT"), ("what is deadlock sa OS", "what is deadlock"), ("overfitting in machine learning", "overfitting"),
])
def test_subject_context_is_dropped_when_other_words_remain(raw, expected):
    assert strip_subject_context(raw) == expected


@pytest.mark.parametrize("raw", ["what is an operating system", "what is ML", "in OS", "what is a join in SQL", "explain the C language"])
def test_subject_context_kept_when_it_is_the_question(raw):
    assert strip_subject_context(raw) == raw


def test_compound_words_match_spaced_forms():
    assert normalize_text("quicksort") == normalize_text("quick sort")
    assert normalize_text("mergesort") == normalize_text("merge sort")


@pytest.mark.parametrize("question,expected", [
    ("what is a deadlock in OS", "What is a deadlock?"), ("explain quicksort", "How does quick sort work?"),
    ("difference between stack and queue", "What is the difference between a stack and a queue?"),
    ("what is paging in operating system", "What is paging?"), ("what is gradient descent", "What is gradient descent?"),
    ("inheritance in oop", "What is inheritance?"), ("what is a lexer", "What is lexical analysis?"),
])
def test_real_questions_that_once_failed_now_match(question, expected):
    """Regression tests for failures found by hand-testing the real knowledge base."""
    faq, _ = load_clean_faq()
    out = Retriever().fit(faq).answer(question)
    assert out["matched_question"] == expected, out


def test_context_word_does_not_hijack_the_match():
    faq, _ = load_clean_faq()
    r = Retriever().fit(faq)
    assert r.answer("what is an operating system")["matched_question"] == "What is an operating system?"
    assert r.answer("what is a deadlock in OS")["matched_question"] != "What is an operating system?"


def test_stored_question_that_contains_the_context_still_wins():
    faq, _ = load_clean_faq()
    r = Retriever().fit(faq)
    assert r.answer("What is variance in machine learning?")["matched_question"] == "What is variance in machine learning?"
    assert r.answer("What is variance?")["matched_question"] == "What is variance?"
    assert r.answer("What is a decoder in digital logic?")["matched_question"] == "What is a decoder in digital logic?"


def test_taglish_context_and_hashmap_synonym_found_by_dry_run():
    """Regression tests for two failures found when pushing informal questions through the app."""
    faq, _ = load_clean_faq()
    r = Retriever().fit(faq)
    assert r.answer("what is deadlock sa OS")["matched_question"] == "What is a deadlock?"
    assert r.answer("what is a hashmap")["matched_question"] == "What is a hash table?"
    assert r.answer("how does a hash map work")["matched_question"] == "What is a hash table?"
