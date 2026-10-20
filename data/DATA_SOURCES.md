# Data sources and transformations

| File | Origin | Notes |
|---|---|---|
| `raw/study_faq_starter.csv`, `raw/intent_training_starter.csv`, `raw/DATASET_GUIDE_starter.md` | Starter pack supplied with the project (82 FAQ rows on study skills, ML and research; 87 intent rows) | Original, unmodified. Only its AI, machine learning, data science, model evaluation and text-processing rows (52 questions) were carried into the BSCS knowledge base; study-skills, academic-skills, research-skills and chatbot-help rows were dropped because they are not BSCS subjects. |
| `study_faq.csv` (optional `reviewed` column) | 52 reused starter rows + 482 rows written for this project | **534 questions, 37 BSCS subjects** (major, elective and supporting/minor subjects). Answers for the new rows were written in a concise textbook style from standard first-course content. `answer_type` is filled by the heuristic in `ml/preprocess.py::infer_answer_type`. **Have an instructor review the answers.** |
| `intent_training.csv` | Reused starter rows (about 97, relabelled to the new subject names) + about 263 rows written for this project | 360 short student-style questions labeled with the 37 subjects. The classifier is trained on these plus the stored FAQ questions. |
| `synonyms.csv` | Written for this project | Editable `phrase,replacement` table (abbreviations such as `dbms`, compound words such as `quicksort`, spelling variants). Some entries were added after seeing which questions failed, so measured gains from synonyms are optimistic. |
| `test_questions.csv` (set A) | Written for this project | 250 questions: 195 in-scope paraphrases of stored questions (easy/hard), 30 out-of-scope, plus 25 extra out-of-scope questions added as `dev` rows only. `split` is dev (used to choose settings) or test (reported). Expected subject and expected stored question were assigned by hand. |
| `test_questions_v2.csv` (set B) | Written for this project | 126 questions (106 in-scope, 20 out-of-scope), written in a terser style, all `test`, never used to choose settings. |

Out-of-scope questions are of two kinds: non-BSCS questions (geography, cooking, health...) and computer-science topics deliberately absent from the knowledge base (GraphQL, Bloom filter, ray tracing...).

Real student questions: use `python -m ml.make_test_file` (see README section 9b). Feedback collected by the app lives in `data/feedback/` (git-ignored, local only).

## Rules for adding data
1. Never put test questions into `study_faq.csv` or `intent_training.csv`; `python -m ml.evaluate` checks for exact and near-duplicate overlap and lists any it finds.
2. Topics must be BSCS subjects listed in `config.BSCS_SUBJECTS`; `python -m ml.preprocess` warns about others.
3. Check the licence of any public dataset (SciQ, SQuAD, ARC, Hugging Face datasets) before importing. Use `python -m ml.import_dataset` so the source and licence are recorded below.
4. Public QA datasets are multiple-choice or passage-based, so they need conversion into short question/answer pairs and manual review.

## Import log (appended automatically by `ml/import_dataset.py`)
