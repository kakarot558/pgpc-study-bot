"""Shared fixtures: a tiny temporary dataset so tests never depend on production answers."""
import os

import pandas as pd
import pytest

# Tests must always run with the shipped defaults, even if the PowerShell window still has
# STUDYBUDDY_* settings from an earlier experiment. (Must happen before `config` is imported.)
for _name in [k for k in os.environ if k.startswith("STUDYBUDDY_")]:
    os.environ.pop(_name)

from app import create_app

TINY_FAQ = [
    ("What is a banana?", "A banana is a yellow fruit.", "Fruit", "definition"),
    ("What is a rocket?", "A rocket is a vehicle that travels to space.", "Space", "definition"),
    ("How do I bake bread?", "Mix flour, water, yeast and salt, then bake.", "Cooking", "advice"),
    ("What is the Pomodoro technique?", "Work in focused 25 minute intervals.", "Study Skills", "definition"),
]


@pytest.fixture()
def tiny_faq_df():
    return pd.DataFrame(TINY_FAQ, columns=["question", "answer", "topic", "answer_type"])


@pytest.fixture()
def tiny_faq_path(tmp_path, tiny_faq_df):
    path = tmp_path / "faq.csv"
    tiny_faq_df.to_csv(path, index=False)
    return path


@pytest.fixture()
def app(tiny_faq_path, tmp_path):
    return create_app({"TESTING": True, "FAQ_PATH": str(tiny_faq_path),
                       "RETRIEVER_ARTIFACT": str(tmp_path / "none.joblib"),
                       "SIMILARITY_THRESHOLD": 0.2,
                       "FEEDBACK_PATH": str(tmp_path / "feedback.jsonl")})


@pytest.fixture()
def client(app):
    return app.test_client()
