"""The production knowledge base must stay inside the BSCS scope and behave sensibly on real questions."""
import pytest

import config
from app import create_app
from ml.preprocess import load_clean_faq, load_clean_intent


@pytest.fixture(scope="module")
def prod_client():
    return create_app({"TESTING": True, "RETRIEVER_ARTIFACT": str(config.MODELS_DIR / "does_not_exist.joblib")}).test_client()


def test_all_topics_are_bscs_subjects():
    faq, report = load_clean_faq()
    assert report["topics_outside_bscs_list"] == []
    assert set(faq["topic"]) <= set(config.BSCS_SUBJECTS)
    intents, _ = load_clean_intent()
    assert set(intents["intent"]) <= set(config.BSCS_SUBJECTS)


def test_every_core_subject_has_content():
    faq, _ = load_clean_faq()
    counts = faq["topic"].value_counts()
    missing = [s for s in config.BSCS_SUBJECTS if counts.get(s, 0) < 4]
    assert not missing, f"subjects with fewer than 4 questions: {missing}"
    assert len(faq) >= 300


def test_no_non_bscs_topics_in_knowledge_base():
    faq, _ = load_clean_faq()
    for banned in ("Study Skills", "Academic Skills", "Research Skills", "Chatbot Basics"):
        assert banned not in set(faq["topic"])


@pytest.mark.parametrize("question,topic", [
    ("What is a deadlock?", "Operating Systems"), ("What is a binary search tree?", "Data Structures"),
    ("Explain normalization in databases", "Database Systems"), ("What is the difference between TCP and UDP?", "Computer Networks"),
    ("What is a deterministic finite automaton?", "Theory of Automata"), ("What is Big O notation?", "Algorithms"),
    ("What is polymorphism?", "Object Oriented Programming"), ("What is Bayes theorem?", "Probability and Statistics"),
    ("What is the agile methodology?", "Software Engineering"), ("What is SQL injection?", "Information Security"),
])
def test_core_bscs_questions_are_answered_in_the_right_subject(prod_client, question, topic):
    data = prod_client.post("/chat", json={"message": question}).get_json()
    assert data["match_found"] and data["topic"] == topic, data


@pytest.mark.parametrize("question", ["Who painted the Mona Lisa?", "What is the capital of France?", "How do I cook rice?",
                                      "What is photosynthesis?", "Who won the World Cup?", "Recommend a good movie"])
def test_non_bscs_questions_are_refused(prod_client, question):
    data = prod_client.post("/chat", json={"message": question}).get_json()
    assert data["match_found"] is False and "BSCS" in data["answer"]


def test_hard_paraphrases_usually_get_an_answer_or_clickable_candidates(prod_client):
    """Coarse guard: for awkwardly worded BSCS questions the bot should answer or offer candidates."""
    questions = ["Is P equal to NP?", "Explain how a function can call itself", "What is the bell curve?",
                 "How does DNS resolve a domain name?", "What is a prefix tree used for autocomplete?",
                 "How do I divide a network with a subnet mask?"]
    helped = 0
    for q in questions:
        data = prod_client.post("/chat", json={"message": q}).get_json()
        helped += bool(data["match_found"] or data["suggestion"] or data["related_questions"])
    assert helped >= 4


def test_topics_and_about_pages_list_bscs_subjects(prod_client):
    assert b"Operating Systems" in prod_client.get("/topics").data
    assert b"BSCS" in prod_client.get("/about").data and b"Compiler Construction" in prod_client.get("/about").data


@pytest.mark.parametrize("question,topic", [
    ("What is functional programming?", "Programming Languages"), ("What is ray tracing?", "Computer Graphics and Visual Computing"),
    ("What is Flutter?", "Mobile Application Development"), ("What is a microcontroller?", "Internet of Things and Embedded Systems"),
    ("What is ETL?", "Data Mining and Data Warehousing"), ("What is the bisection method?", "Numerical Methods"),
    ("What is a feasibility study?", "Systems Analysis and Design"), ("What is a Gantt chart?", "IT Project Management"),
    ("What is a literature review?", "Technical Writing and Research Methods"), ("What is Ohm's law?", "Applied Physics and Electronics"),
    ("What is a startup?", "Technopreneurship"), ("What is a game loop?", "Game Development"),
])
def test_minor_and_elective_subjects_are_answered(prod_client, question, topic):
    data = prod_client.post("/chat", json={"message": question}).get_json()
    assert data["match_found"] and data["topic"] == topic, data


def test_curriculum_has_major_elective_and_supporting_subjects():
    faq, _ = load_clean_faq()
    assert faq["topic"].nunique() == len(config.BSCS_SUBJECTS) == 37
    assert len(faq) >= 500


@pytest.mark.parametrize("question", ["Who is the president of the Philippines?", "What is the history of the Philippines?",
                                      "How do I cook adobo?", "Explain World War 2", "What is Kubernetes?"])
def test_non_bscs_and_missing_topics_still_refused(prod_client, question):
    data = prod_client.post("/chat", json={"message": question}).get_json()
    assert data["match_found"] is False
