# Dataset guide

## Included files

- `study_faq.csv`: 82 question-answer records. Columns: `question`, `answer`, `topic`. This is the main dataset used by the retrieval chatbot.
- `intent_training.csv`: 87 labeled utterances across 12 intents. Columns: `text`, `intent`. This is optional for a separate supervised intent-classification experiment.

## Important limitations

These are educational starter datasets created for this project. They are not an official curriculum, and they are not a substitute for your instructor's materials. Review and correct answers before submitting the project. The intent dataset is small; expand it with diverse, accurately labeled examples before drawing conclusions from evaluation metrics.

## How the main chatbot uses the FAQ file

The Flask application reads `study_faq.csv`, fits a TF-IDF vectorizer on its question column, and compares each student query with stored questions using cosine similarity. It returns the answer for the closest match only when the similarity exceeds the configured threshold.

## Add data

Keep the exact column names. In `study_faq.csv`, add a question, an accurate answer, and a topic. In `intent_training.csv`, add a representative utterance and one consistent intent label. Use a separate unseen test set for evaluation.
