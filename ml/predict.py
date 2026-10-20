"""Answer matching (TF-IDF + cosine similarity) and optional intent classification.

HOW RETRIEVAL WORKS (plain language)
  1. Every stored question is turned into a list of numbers (a TF-IDF vector).
  2. The student's question is turned into numbers the same way.
  3. Cosine similarity measures how close the student's vector is to each stored vector.
  4. The closest stored question wins - but only if its similarity reaches a threshold.
     Otherwise the bot admits that it does not know.

The score used here is the cosine similarity multiplied by "coverage" (the share of the
student's words that appear anywhere in the knowledge base), so unknown words lower the score.
Cosine similarity is a *closeness score between 0 and 1*. It is NOT a probability and
NOT an accuracy figure. This is a search system, not a language model: it can only
return answers that already exist in the CSV.
"""
import hashlib
import logging
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.linear_model import LogisticRegression
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline

import config
from ml.preprocess import (load_clean_faq, make_char_vectorizer, make_vectorizer, normalize_text,
                           strip_subject_context)

log = logging.getLogger(__name__)

FALLBACK_MESSAGE = (
    "I could not find that in my BSCS notes. I only answer BSCS (Computer Science) course questions "
    "that are stored in my local dataset. If suggestions appear below, tap one; otherwise rephrase with "
    "key terms (for example 'What is a deadlock?') or browse the Topics page to see the subjects I cover."
)
NO_WORDS_MESSAGE = (
    "I could not find any meaningful words to search for. Try a short BSCS question such as "
    "'What is a binary search tree?'"
)


def file_sha256(path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def source_fingerprint(faq_path, synonyms_path=None) -> str:
    """Hash of everything the saved index depends on: the FAQ CSV and the synonyms CSV."""
    synonyms_path = Path(synonyms_path or config.SYNONYMS_PATH)
    h = hashlib.sha256()
    for path in (Path(faq_path), synonyms_path):
        h.update(path.read_bytes() if path.exists() else b"<missing>")
    return h.hexdigest()


class Retriever:
    """Finds the stored question closest to a student's question."""

    def __init__(self, threshold: float = config.SIMILARITY_THRESHOLD,
                 suggestion_threshold: float = config.SUGGESTION_THRESHOLD,
                 char_weight: float = config.CHAR_WEIGHT,
                 answer_weight: float = config.ANSWER_WEIGHT):
        self.threshold = float(threshold)
        self.char_weight = float(char_weight)
        self.answer_weight = float(answer_weight)
        if self.char_weight < 0 or self.answer_weight < 0 or self.char_weight + self.answer_weight > 1:
            raise ValueError("char_weight and answer_weight must be >= 0 and add up to at most 1.")
        self.doc_vectorizer = None
        self.doc_matrix = None
        self.char_vectorizer = None
        self.char_matrix = None
        self.suggestion_threshold = float(suggestion_threshold)
        self.vectorizer = None
        self.matrix = None
        self.table = None          # DataFrame: question, answer, topic, answer_type
        self.meta = {}

    # ------------------------------------------------------------ building
    def fit(self, faq: pd.DataFrame, source_sha256: str = ""):
        """Fit TF-IDF on the stored questions and keep their vectors."""
        if faq is None or faq.empty:
            raise ValueError("Cannot build a retrieval index from an empty dataset.")
        self.table = faq.reset_index(drop=True).copy()
        self.vectorizer = make_vectorizer()
        self.matrix = self.vectorizer.fit_transform(self.table["question"])
        # "document view": question + answer text together, so a student's wording can also match
        # the explanation that is stored (for example "subnet mask" inside the subnetting answer).
        docs = (self.table["question"] + " " + self.table["answer"]).tolist()
        self.doc_vectorizer = make_vectorizer()
        self.doc_matrix = self.doc_vectorizer.fit_transform(docs)
        self.char_vectorizer = make_char_vectorizer()
        self.char_matrix = self.char_vectorizer.fit_transform(self.table["question"])
        self.meta = {
            "sklearn_version": sklearn.__version__,
            "source_sha256": source_sha256,
            "num_questions": int(len(self.table)),
            "vocabulary_size": int(len(self.vectorizer.vocabulary_)),
            "ngram_range": list(config.NGRAM_RANGE),
            "built_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        }
        return self

    # ------------------------------------------------------------- matching
    def _question_tokens(self) -> list:
        """Normalised word sets of the stored questions (built once, not saved in the artifact)."""
        if getattr(self, "_qtok_cache", None) is None or len(self._qtok_cache) != self.size:
            self._qtok_cache = [frozenset(normalize_text(q).split()) for q in self.table["question"]]
        return self._qtok_cache

    def _scores(self, message: str):
        """Blend of the matching views for ONE phrasing of the question. Returns (scores, cosine, coverage)."""
        query = self.vectorizer.transform([message])
        cosine = cosine_similarity(query, self.matrix).ravel()
        # Words the vectorizer has never seen are silently ignored, which could make a
        # question with ONE known word look like a perfect match. So we scale the cosine
        # by "coverage" = share of the question's words that exist in the knowledge base.
        words = normalize_text(message).split()
        known = sum(1 for w in words if w in self.vectorizer.vocabulary_)
        coverage = known / len(words) if words else 0.0
        word_scores = cosine * coverage
        scores = word_scores
        char_cos = char_cov = 0.0
        if self.char_weight > 0:
            # Second opinion from character n-grams (handles word endings such as
            # "tokenizing" / "tokenization"), with the same coverage idea.
            grams = self.char_vectorizer.build_analyzer()(message)
            vocab = self.char_vectorizer.vocabulary_
            char_cov = sum(1 for g in grams if g in vocab) / len(grams) if grams else 0.0
            char_cos = cosine_similarity(self.char_vectorizer.transform([message]), self.char_matrix).ravel()
            scores = (1 - self.char_weight) * word_scores + self.char_weight * char_cos * char_cov
        if self.answer_weight > 0:
            # Third view: the question + answer text, so wording that appears in the stored
            # explanation (for example "subnet mask") can match too.
            doc_cos = cosine_similarity(self.doc_vectorizer.transform([message]), self.doc_matrix).ravel()
            doc_known = sum(1 for w in words if w in self.doc_vectorizer.vocabulary_)
            doc_cov = doc_known / len(words) if words else 0.0
            wc, wa = self.char_weight, self.answer_weight
            char_part = (char_cos * char_cov) if self.char_weight > 0 else 0.0
            scores = (1 - wc - wa) * word_scores + wc * char_part + wa * doc_cos * doc_cov
        return scores, cosine, coverage

    def rank(self, message: str, top_k: int = config.TOP_K) -> list:
        """Return the top_k closest stored questions with their similarity scores.

        The question is scored as typed AND with a trailing subject name removed ("deadlock in OS"
        becomes "deadlock"); each stored question keeps the better of the two scores.
        """
        scores, cosine, coverage = self._scores(message)
        stripped = strip_subject_context(message)
        if stripped != " ".join(message.split()):
            alt, _, _ = self._scores(stripped)
            context = set(normalize_text(message).split()) - set(normalize_text(stripped).split())
            # A stored question made only of the context words (e.g. "What is an operating system?")
            # must not collect credit for them; every other question keeps its better score.
            context_only = np.array([bool(t) and t <= context for t in self._question_tokens()])
            scores = np.where(context_only, alt, np.maximum(scores, alt))
        order = np.argsort(-scores, kind="stable")[:top_k]
        return [{
            "index": int(i),
            "question": self.table.at[i, "question"],
            "answer": self.table.at[i, "answer"],
            "topic": self.table.at[i, "topic"],
            "answer_type": self.table.at[i, "answer_type"],
            "similarity": float(scores[i]),
            "cosine": float(cosine[i]),
            "coverage": float(coverage),
        } for i in order]

    def answer(self, message: str) -> dict:
        """Full chatbot response for one student message."""
        message = (message or "").strip()
        result = {"answer": FALLBACK_MESSAGE, "topic": None, "matched_question": None,
                  "answer_type": None, "similarity_score": 0.0, "match_found": False,
                  "suggestion": None, "related_questions": [], "confidence": None}
        if not normalize_text(message):
            result["answer"] = NO_WORDS_MESSAGE
            return result
        ranked = self.rank(message, top_k=1 + config.MAX_RELATED)
        best = ranked[0]
        result["similarity_score"] = round(best["similarity"], 4)
        result["related_questions"] = [r["question"] for r in ranked[1:]
                                       if r["similarity"] >= self.suggestion_threshold]
        if best["similarity"] >= self.threshold:
            result.update(answer=best["answer"], topic=best["topic"],
                          matched_question=best["question"], answer_type=best["answer_type"],
                          match_found=True,
                          confidence="high" if best["similarity"] >= config.HIGH_CONFIDENCE else "low")
        elif best["similarity"] >= self.suggestion_threshold:
            result["suggestion"] = best["question"]      # a hint only, not an answer
        return result

    # ---------------------------------------------------------- information
    def topic_summary(self) -> list:
        """[{'name', 'count', 'questions': [...]}, ...] sorted by topic name."""
        out = []
        for name, group in self.table.groupby("topic", sort=True):
            out.append({"name": name, "count": int(len(group)), "questions": group["question"].tolist()})
        return out

    def flashcards(self, subject=None) -> list:
        """One card per distinct answer (alternative phrasings of a concept count once)."""
        t = self.table if not subject else self.table[self.table["topic"] == subject]
        t = t.drop_duplicates(subset="answer", keep="first")
        return [{"id": int(i), "question": r["question"], "answer": r["answer"], "topic": r["topic"]}
                for i, r in t.iterrows()]

    @property
    def reviewed_count(self) -> int:
        """Number of stored answers marked as instructor-reviewed (0 if the column is absent)."""
        return int((self.table["reviewed"] == "yes").sum()) if "reviewed" in self.table.columns else 0

    @property
    def size(self) -> int:
        return 0 if self.table is None else int(len(self.table))

    # ------------------------------------------------------------- saving
    def save(self, path=config.RETRIEVER_ARTIFACT) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump({"vectorizer": self.vectorizer, "matrix": self.matrix,
                     "char_vectorizer": self.char_vectorizer, "char_matrix": self.char_matrix,
                     "doc_vectorizer": self.doc_vectorizer, "doc_matrix": self.doc_matrix,
                     "table": self.table, "meta": self.meta}, path)
        return path

    @classmethod
    def load(cls, path, threshold=config.SIMILARITY_THRESHOLD, char_weight=config.CHAR_WEIGHT,
             answer_weight=config.ANSWER_WEIGHT) -> "Retriever":
        data = joblib.load(path)
        obj = cls(threshold=threshold, char_weight=char_weight, answer_weight=answer_weight)
        obj.doc_vectorizer, obj.doc_matrix = data["doc_vectorizer"], data["doc_matrix"]
        obj.vectorizer, obj.matrix = data["vectorizer"], data["matrix"]
        obj.char_vectorizer, obj.char_matrix = data["char_vectorizer"], data["char_matrix"]
        obj.table, obj.meta = data["table"], data["meta"]
        return obj


def load_or_build_retriever(faq_path=None, artifact_path=None,
                            threshold=config.SIMILARITY_THRESHOLD):
    """Return (retriever, source) where source is 'artifact' or 'built'.

    The saved artifact is only reused if it was made from the SAME csv file contents
    and the SAME scikit-learn version. Otherwise the index is rebuilt in memory
    (this takes a fraction of a second for a small dataset).
    """
    faq_path = Path(faq_path or config.FAQ_PATH)
    artifact_path = Path(artifact_path or config.RETRIEVER_ARTIFACT)
    sha = source_fingerprint(faq_path) if faq_path.exists() else ""
    if artifact_path.exists():
        try:
            retriever = Retriever.load(artifact_path, threshold=threshold)
            meta = retriever.meta
            if meta.get("source_sha256") == sha and meta.get("sklearn_version") == sklearn.__version__:
                return retriever, "artifact"
            log.info("Saved model is out of date; rebuilding from the CSV.")
        except Exception:                              # corrupt or incompatible file
            log.warning("Could not load %s; rebuilding.", artifact_path, exc_info=True)
    faq, _ = load_clean_faq(faq_path)
    return Retriever(threshold=threshold).fit(faq, source_sha256=sha), "built"


# ------------------------------------------------------- optional classifier
CLASSIFIER_KINDS = ("logreg", "naive_bayes")


def make_classifier_pipeline(kind: str = "logreg") -> Pipeline:
    """TF-IDF features + a supervised classifier that predicts a TOPIC label."""
    if kind == "logreg":
        model = LogisticRegression(C=10.0, max_iter=1000, random_state=config.RANDOM_SEED)
    elif kind == "naive_bayes":
        model = MultinomialNB(alpha=0.1)
    else:
        raise ValueError(f"Unknown classifier kind '{kind}'. Choose from {CLASSIFIER_KINDS}.")
    return Pipeline([("tfidf", make_vectorizer()), ("clf", model)])


class IntentClassifier:
    """Predicts a broad topic for a question. It does NOT return an answer."""

    def __init__(self, kind: str = "logreg"):
        self.kind = kind
        self.pipeline = make_classifier_pipeline(kind)
        self.meta = {}

    def fit(self, texts, labels):
        self.pipeline.fit(list(texts), list(labels))
        self.meta = {"kind": self.kind, "sklearn_version": sklearn.__version__,
                     "classes": [str(c) for c in self.pipeline.classes_],
                     "trained_on_rows": len(list(labels))}
        return self

    def predict(self, texts) -> list:
        return [str(p) for p in self.pipeline.predict(list(texts))]

    def predict_one(self, text: str) -> dict:
        proba = self.pipeline.predict_proba([text])[0]
        best = int(np.argmax(proba))
        # model_probability is the classifier's own estimate - it is NOT calibrated.
        return {"intent": str(self.pipeline.classes_[best]), "model_probability": float(proba[best])}

    def save(self, path=config.INTENT_ARTIFACT) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump({"pipeline": self.pipeline, "kind": self.kind, "meta": self.meta}, path)
        return path

    @classmethod
    def load(cls, path=config.INTENT_ARTIFACT) -> "IntentClassifier":
        data = joblib.load(path)
        obj = cls(kind=data["kind"])
        obj.pipeline, obj.meta = data["pipeline"], data["meta"]
        return obj
