"""Central settings for StudyBuddy.

Every path and tunable number lives here so that beginners have one place to look.
Environment variables (optional) can override a few values without editing code.
"""
import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent

# --- Data files -----------------------------------------------------------
DATA_DIR = BASE_DIR / "data"
FAQ_PATH = DATA_DIR / "study_faq.csv"              # knowledge base (you edit this)
SYNONYMS_PATH = DATA_DIR / "synonyms.csv"        # abbreviations / spelling variants (editable)
INTENT_PATH = DATA_DIR / "intent_training.csv"     # labeled examples for the classifier
TEST_PATH = DATA_DIR / "test_questions.csv"        # held-out evaluation questions
PROCESSED_DIR = DATA_DIR / "processed"
MODELS_DIR = BASE_DIR / "models"
RESULTS_DIR = BASE_DIR / "results"
RETRIEVER_ARTIFACT = MODELS_DIR / "retriever.joblib"
INTENT_ARTIFACT = MODELS_DIR / "intent_classifier.joblib"

# --- Retrieval ------------------------------------------------------------
# A stored question must reach this cosine similarity to be returned.
# It was chosen on the "dev" rows of test_questions.csv (see results/evaluation_report.md).
SIMILARITY_THRESHOLD = float(os.environ.get("STUDYBUDDY_THRESHOLD", "0.25"))
# Below the threshold but above this value we offer a "Did you mean...?" hint.
SUGGESTION_THRESHOLD = 0.12
# Answers scoring at least this are shown as "high confidence"; lower ones carry a caution note.
# (Read off the evaluation sets, so treat it as a presentation hint, not a calibrated probability.)
HIGH_CONFIDENCE = 0.55
TOP_K = 3
# Share of the final score that comes from character n-gram similarity (0 = words only).
CHAR_WEIGHT = float(os.environ.get("STUDYBUDDY_CHAR_WEIGHT", "0.0"))
# Share of the final score that comes from matching the student's words against the stored ANSWER text too.
ANSWER_WEIGHT = float(os.environ.get("STUDYBUDDY_ANSWER_WEIGHT", "0.65"))
CHAR_NGRAM_RANGE = (3, 5)
MAX_RELATED = 2     # best match + 2 related = the top 3 candidates
NGRAM_RANGE = (1, 2)

# --- Feedback (stored locally, only when the student clicks a feedback button) ---------
# Set STUDYBUDDY_FEEDBACK_DIR to keep the log outside a cloud-synced folder such as OneDrive.
FEEDBACK_DIR = Path(os.environ.get("STUDYBUDDY_FEEDBACK_DIR", str(DATA_DIR / "feedback")))
FEEDBACK_PATH = FEEDBACK_DIR / "feedback.jsonl"
FEEDBACK_MAX_BYTES = 5 * 1024 * 1024
# Off by default. If "1", questions the bot could not answer are also logged (text only, no
# personal data) so the owner can see what to add next. Students should be told if you enable it.
LOG_DECLINED = os.environ.get("STUDYBUDDY_LOG_DECLINED", "0") == "1"

# --- Web app --------------------------------------------------------------
# 127.0.0.1 = this computer only (default, safest). For a classroom pilot you may set STUDYBUDDY_HOST=0.0.0.0
# so others on the same network can connect; there is no login, so use a trusted network only.
HOST = os.environ.get("STUDYBUDDY_HOST", "127.0.0.1")
PORT = int(os.environ.get("STUDYBUDDY_PORT", "5000"))
DEBUG = os.environ.get("STUDYBUDDY_DEBUG", "0") == "1"   # keep off unless developing
MAX_MESSAGE_LENGTH = 300
MAX_CONTENT_LENGTH = 16 * 1024   # reject request bodies larger than 16 KB

# --- Machine learning -----------------------------------------------------
RANDOM_SEED = 42
VALIDATION_SIZE = 0.2   # share of intent_training.csv used for the internal validation check

# --- Scope: BSCS (Bachelor of Science in Computer Science) core subjects only ----------
# Topics in data/study_faq.csv that are not in this list are reported by `python -m ml.preprocess`
# so that questions outside the BSCS scope do not slip into the knowledge base unnoticed.
BSCS_SUBJECTS = (
    "Programming Fundamentals", "Object Oriented Programming", "Data Structures", "Algorithms",
    "Discrete Mathematics", "Linear Algebra and Calculus", "Probability and Statistics",
    "Computer Organization and Architecture", "Digital Logic Design", "Operating Systems",
    "Database Systems", "Computer Networks", "Software Engineering", "Theory of Automata",
    "Compiler Construction", "Artificial Intelligence", "Machine Learning", "Model Evaluation",
    "Data Science", "Natural Language Processing", "Web Development", "Information Security",
    "Human Computer Interaction", "Parallel and Distributed Computing", "Professional Practices",
    # further major and elective subjects
    "Programming Languages", "Computer Graphics and Visual Computing", "Mobile Application Development",
    "Internet of Things and Embedded Systems", "Data Mining and Data Warehousing", "Game Development",
    # supporting (minor) subjects of a BSCS curriculum
    "Numerical Methods", "Systems Analysis and Design", "IT Project Management",
    "Technical Writing and Research Methods", "Applied Physics and Electronics", "Technopreneurship",
)


def active_overrides() -> dict:
    """Scoring settings changed by environment variables. They change evaluation results, so reports
    and the regression baseline must never silently mix them with the shipped defaults."""
    names = ("STUDYBUDDY_THRESHOLD", "STUDYBUDDY_CHAR_WEIGHT", "STUDYBUDDY_ANSWER_WEIGHT")
    return {k: os.environ[k] for k in names if k in os.environ}
