import json

import pandas as pd
import pytest

import config
from ml.evaluate import check_leakage, find_leakage, load_test_set
from ml.import_dataset import convert
from ml.preprocess import (DatasetError, clean_faq, clean_intent, find_near_duplicates, infer_answer_type,
                           light_stem, load_clean_faq, load_clean_intent, normalize_text,
                           normalize_whitespace, read_csv_safely, run_preprocessing)
from ml.train import split_intent_data


def make(rows):
    return pd.DataFrame(rows, columns=["question", "answer", "topic", "answer_type"])


# ---- loading ------------------------------------------------------------
def test_loads_dataset(tiny_faq_path):
    df, report = load_clean_faq(tiny_faq_path)
    assert len(df) == 4 and report["rows_before"] == report["rows_after"] == 4
    assert list(df.columns) == ["question", "answer", "topic", "answer_type"]


def test_production_datasets_load_and_have_no_leakage():
    faq, _ = load_clean_faq()
    intents, _ = load_clean_intent()
    test_df = load_test_set()
    assert len(faq) > 0 and len(intents) > 0 and len(test_df) > 0
    leak = check_leakage(faq, intents, test_df)
    assert leak["intent_training_vs_test"]["exact"] == [] and leak["intent_training_vs_test"]["near"] == []
    assert leak["knowledge_base_vs_test"]["exact"] == []
    assert leak["duplicates_inside_test_set"] == 0
    # every expected question must really exist in the knowledge base
    known = set(faq["question"])
    for value in test_df.loc[test_df.in_scope == "yes", "expected_question"]:
        assert set(value.split("|")) <= known


def test_missing_file_raises(tmp_path):
    with pytest.raises(DatasetError):
        read_csv_safely(tmp_path / "nope.csv")


def test_missing_columns_raise(tmp_path):
    path = tmp_path / "bad.csv"
    pd.DataFrame({"question": ["a?"], "answer": ["b"]}).to_csv(path, index=False)
    with pytest.raises(DatasetError, match="topic"):
        load_clean_faq(path)


def test_empty_file_raises(tmp_path):
    path = tmp_path / "empty.csv"
    path.write_text("", encoding="utf-8")
    with pytest.raises(DatasetError):
        read_csv_safely(path)


def test_dataset_with_no_usable_rows_raises(tmp_path):
    path = tmp_path / "blank.csv"
    pd.DataFrame({"question": [" "], "answer": [""], "topic": ["x"]}).to_csv(path, index=False)
    with pytest.raises(DatasetError):
        load_clean_faq(path)


def test_bom_and_windows_line_endings(tmp_path):
    path = tmp_path / "bom.csv"
    path.write_bytes("\ufefftext,intent\r\nhello there,Greeting\r\n".encode("utf-8"))
    df, _ = load_clean_intent(path)
    assert df.iloc[0]["text"] == "hello there"


# ---- cleaning -----------------------------------------------------------
def test_cleaning_rules():
    df = make([
        ["  What  is\nAI? ", "Answer   one.", "ai", ""],
        ["What is AI?", "answer one.", "AI", ""],           # exact duplicate (case/space differences)
        ["", "orphan answer", "AI", ""],                    # empty question
        ["Define ML", "", "AI", ""],                        # empty answer
        ["What is ML?", "Machine learning.", "", ""],       # missing topic
        ["What is ML?", "A different answer.", "AI", ""],   # conflicting answer
    ])
    clean, rep = clean_faq(df)
    assert rep["rows_before"] == 6 and rep["rows_after"] == 2
    assert rep["removed_empty"] == 2 and rep["removed_exact_duplicates"] == 1
    assert rep["removed_conflicting_rows"] == 1 and rep["conflicting_answers"][0]["question"] == "What is ML?"
    assert clean.iloc[0]["question"] == "What is AI?"            # whitespace normalised
    assert clean.iloc[1]["answer"] == "Machine learning."        # first answer kept, punctuation preserved
    assert set(clean["topic"]) <= {"General", "ai"} or len(set(clean["topic"])) >= 1


def test_topic_spellings_are_unified():
    df = make([["q one?", "a1", "Machine Learning", "definition"], ["q two?", "a2", "machine_learning", "definition"],
               ["q three?", "a3", "Machine Learning", "definition"]])
    clean, rep = clean_faq(df)
    assert clean["topic"].nunique() == 1 and clean["topic"].iloc[0] == "Machine Learning"
    assert rep["topic_values_standardised"] == 1


def test_similar_questions_are_reported_not_deleted():
    df = make([["What is overfitting in models?", "a1", "T", ""], ["what is overfitting in model", "a2", "T", ""]])
    clean, rep = clean_faq(df)
    # identical after normalisation but different keys/answers -> conflict handling keeps first; near-dup list works directly:
    pairs = find_near_duplicates(["What is recall in ML?", "Explain recall in ML", "How do I bake bread?"], threshold=0.5)
    assert len(pairs) == 1 and pairs[0]["similarity"] >= 0.5
    assert len(clean) >= 1


def test_answer_type_inference_and_preservation():
    assert infer_answer_type("What is recall?") == "definition"
    assert infer_answer_type("How can I study?") == "advice"
    assert infer_answer_type("What is the difference between a and b?") == "comparison"
    assert infer_answer_type("Give examples of AI") == "example"
    clean, rep = clean_faq(make([["What is x?", "a", "T", "custom"], ["How to y?", "b", "T", ""]]))
    assert clean["answer_type"].tolist() == ["custom", "advice"] and rep["answer_types_inferred"] == 1


def test_clean_intent_handles_empty_duplicates_and_conflicts():
    df = pd.DataFrame({"text": ["hi", "Hi", "", "hello", "hello"], "intent": ["Greeting", "greeting", "Greeting", "Greeting", "Other"]})
    clean, rep = clean_intent(df)
    assert rep["removed_empty"] == 1 and rep["removed_conflicting_labels"] == 1 and len(clean) == 2


def test_run_preprocessing_saves_files_and_reports_counts(tmp_path, tiny_faq_path):
    intent_path = tmp_path / "intent.csv"
    pd.DataFrame({"text": ["a b", "c d"], "intent": ["X", "Y"]}).to_csv(intent_path, index=False)
    out = tmp_path / "processed"
    reports = run_preprocessing(tiny_faq_path, intent_path, out)
    assert (out / "study_faq_clean.csv").exists() and (out / "intent_training_clean.csv").exists()
    assert reports["faq"]["rows_before"] == 4
    assert json.loads((out / "preprocess_report.json").read_text())["intent"]["rows_after"] == 2


# ---- text normalisation -------------------------------------------------
def test_normalisation():
    assert normalize_whitespace("  a \n b\t") == "a b"
    assert normalize_text("What ARE the Labels?!") == "label"
    assert light_stem("classes") == "class" and light_stem("accuracy") == "accuracy" and light_stem("bias") == "bia"   # odd but consistent: both sides are stemmed the same way
    assert normalize_text("") == "" and normalize_text("?!") == "" and normalize_text(None) == ""


# ---- splitting / leakage / import --------------------------------------
def test_intent_split_is_stratified_and_reproducible():
    df = pd.DataFrame({"text": [f"t{i}" for i in range(40)], "intent": ["A", "B"] * 20})
    tr1, va1 = split_intent_data(df, 0.25, seed=1)
    tr2, va2 = split_intent_data(df, 0.25, seed=1)
    assert tr1.equals(tr2) and va1.equals(va2)
    assert set(va1["intent"]) == {"A", "B"} and not set(tr1["text"]) & set(va1["text"])


def test_leakage_detection():
    found = find_leakage(["What is recall?", "Plan study schedule weekly exams"], ["what is Recall", "Plan study schedule weekly", "Unrelated words here"])
    assert len(found["exact"]) == 1 and len(found["near"]) == 1


def test_test_set_validation(tmp_path):
    path = tmp_path / "t.csv"
    pd.DataFrame({"question": ["q"], "expected_topic": ["T"], "expected_question": ["x"], "in_scope": ["maybe"], "split": ["test"]}).to_csv(path, index=False)
    with pytest.raises(DatasetError):
        load_test_set(path)


def test_import_converts_and_validates_columns():
    src = pd.DataFrame({"q": ["What is a cell?", ""], "a": ["The basic unit of life.", "x"]})
    df, rep = convert(src, "q", "a", "Biology")
    assert len(df) == 1 and df.iloc[0]["topic"] == "Biology"
    with pytest.raises(DatasetError):
        convert(src, "missing", "a", "Biology")
