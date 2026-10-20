"""Prepare the models that the web app uses.

Run from the project root:   python -m ml.train

IMPORTANT DISTINCTION
  * Retrieval "indexing": TF-IDF is fitted on ALL knowledge-base questions (and answers). There is nothing
    to predict here - the index is simply the searchable list of stored questions. It is
    evaluated separately on data/test_questions.csv, which is not part of the index.
  * Intent classification is real supervised learning. It learns from
    data/intent_training.csv plus the knowledge-base questions (each already has a topic label). Part of that file is held back for an internal validation
    check, then the final model is refitted on every training row. The separate
    test_questions.csv file is never used for training.
"""
import json
import sys
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import sklearn
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split

import config
from ml.predict import IntentClassifier, Retriever, source_fingerprint
from ml.preprocess import (DatasetError, combine_training_data, load_clean_faq, load_clean_intent,
                           run_preprocessing)


def split_intent_data(df: pd.DataFrame, test_size: float = config.VALIDATION_SIZE,
                      seed: int = config.RANDOM_SEED):
    """Stratified split when every class has enough rows, otherwise a plain random split."""
    counts = df["intent"].value_counts()
    n_test = int(round(len(df) * test_size))
    can_stratify = counts.min() >= 2 and n_test >= counts.shape[0]
    return train_test_split(df, test_size=test_size, random_state=seed,
                            stratify=df["intent"] if can_stratify else None)


def main() -> int:
    np.random.seed(config.RANDOM_SEED)
    try:
        reports = run_preprocessing()
        faq, _ = load_clean_faq()
        intents, _ = load_clean_intent()
    except DatasetError as exc:
        print(f"ERROR: {exc}")
        return 1
    print(f"Preprocessing done: {len(faq)} FAQ rows, {len(intents)} intent rows.")

    # 1) retrieval index
    retriever = Retriever().fit(faq, source_sha256=source_fingerprint(config.FAQ_PATH))
    retriever.save()
    print(f"Retrieval index saved ({retriever.meta['vocabulary_size']} vocabulary terms).")

    # 2) intent classifier: trained on intent_training.csv plus the stored questions (labeled by topic);
    #    internal validation split first, then the final fit on all rows
    intents = combine_training_data(intents, faq)
    train_df, val_df = split_intent_data(intents)
    validation = {}
    for kind in ("logreg", "naive_bayes"):
        model = IntentClassifier(kind).fit(train_df["text"], train_df["intent"])
        pred = model.predict(val_df["text"])
        validation[kind] = {"accuracy": round(accuracy_score(val_df["intent"], pred), 4),
                            "macro_f1": round(f1_score(val_df["intent"], pred, average="macro", zero_division=0), 4)}
        print(f"  {kind}: validation accuracy={validation[kind]['accuracy']}  macro-F1={validation[kind]['macro_f1']}")
    final = IntentClassifier("logreg").fit(intents["text"], intents["intent"])
    final.save()
    print("Intent classifier (logistic regression) saved.")

    info = {
        "trained_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "random_seed": config.RANDOM_SEED,
        "versions": {"scikit-learn": sklearn.__version__, "pandas": pd.__version__, "numpy": np.__version__},
        "retrieval": {**retriever.meta, "threshold_setting": config.SIMILARITY_THRESHOLD,
                      "topic_counts": faq["topic"].value_counts().to_dict()},
        "intent": {"rows": int(len(intents)), "sources": "intent_training.csv + knowledge-base questions", "classes": final.meta["classes"],
                   "class_counts": intents["intent"].value_counts().to_dict(),
                   "validation_split": {"train_rows": int(len(train_df)), "validation_rows": int(len(val_df)),
                                        "results": validation},
                   "note": "Validation scores come from a small split of the training file. "
                           "Use `python -m ml.evaluate` for the held-out test results."},
        "preprocessing_reports": reports,
    }
    (config.MODELS_DIR / "training_info.json").write_text(json.dumps(info, indent=2), encoding="utf-8")
    print(f"Details written to {config.MODELS_DIR / 'training_info.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
