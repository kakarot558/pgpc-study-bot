# StudyBuddy: BSCS Study Assistant Chatbot

A local web chatbot for **BSCS (Bachelor of Science in Computer Science)** students. It answers questions from the major, elective and supporting (minor) subjects of a BSCS curriculum by **matching them to a prepared question-and-answer dataset** with machine learning text matching (TF-IDF + cosine similarity). It also includes a separate, optional **supervised topic classifier** for comparison.

* No OpenAI / Gemini / Hugging Face API, no API keys, no paid services.
* Everything runs on your computer. Questions are never sent anywhere; they are saved locally only if a student clicks a feedback button (section 9b).
* The knowledge base is a CSV file you can edit; adding knowledge needs no Python changes.
* **BSCS only:** non-BSCS questions get a polite "not found" reply.

> **Honest scope.** StudyBuddy is a *retrieval* system with 534 stored answers across 37 BSCS subjects. No dataset can contain *every* possible question, and this chatbot does not "understand" language like a generative model: it can only return answers that exist in `data/study_faq.csv`. Questions it cannot match get suggestions ("Did you mean...?") instead of a made-up answer. Section 7 reports exactly how well it does, including where it fails.

---

## 1. Subjects covered (534 questions, 37 subjects)

| Group | Subjects (number of stored questions) |
|---|---|
| Programming | Programming Fundamentals (20), Object Oriented Programming (12), Data Structures (20), Algorithms (27), Programming Languages (11), Web Development (16), Mobile Application Development (9), Game Development (6) |
| Mathematics | Discrete Mathematics (17), Linear Algebra and Calculus (14), Probability and Statistics (16), Numerical Methods (10) |
| Hardware and systems | Computer Organization and Architecture (20), Digital Logic Design (12), Operating Systems (26), Computer Networks (25), Parallel and Distributed Computing (11), Internet of Things and Embedded Systems (10) |
| Data and software | Database Systems (24), Data Mining and Data Warehousing (10), Software Engineering (24), Systems Analysis and Design (10), Information Security (19), Human Computer Interaction (9), Computer Graphics and Visual Computing (12) |
| Theory | Theory of Automata (14), Compiler Construction (13) |
| AI and data | Artificial Intelligence (18), Machine Learning (18), Model Evaluation (17), Data Science (8), Natural Language Processing (8) |
| Supporting (minor) subjects | IT Project Management (9), Technical Writing and Research Methods (12), Applied Physics and Electronics (11), Technopreneurship (7), Professional Practices (9) |

Coverage per subject is the standard first-course concepts (for Operating Systems: processes, threads, scheduling, deadlock, semaphores, paging, virtual memory, file systems, and so on). **Not included:** the non-technical general-education subjects found in some BSCS programmes (history, art appreciation, physical education, and similar). Those are not computer science, so non-BSCS questions are refused. Other deeper electives are also absent; add them as described in section 9. The answers were written for this project in a concise textbook style; **have an instructor review them before relying on them for exams.**

## 2. Features

* Chat page with message bubbles, loading indicator, Clear Chat, suggested questions and a topic filter (37 subjects)
* Topics page listing every subject and question; About page explaining the method and scope
* Word + answer-text TF-IDF matching (optional character n-gram view), an editable synonyms file (`os`, `dbms`, `bfs`...), configurable similarity threshold
* "Closest match, please check it" caution for lower-scoring answers, and clickable **Did you mean...?** buttons when the bot declines
* Helpful / Not helpful buttons (saved locally, only on click), a local feedback report, and an optional declined-question log
* Flashcard **quiz page** built from the knowledge base
* Instructor **review workflow** (`reviewed` column), phrasing tool, coverage report, real-student test-file builder, and a regression check for CI
* Taglish question words ("ano ang deadlock?") and "vs" / "versus" understood
* A ready-made **one-week pilot kit** (`pilot/`): student notice, question-collection form, step-by-step guide and a progress checker (`python -m ml.pilot_status`)
* One-command launchers (`start_windows.ps1`, `start_unix.sh`)
* `/health` route to confirm the knowledge base loaded
* Dataset cleaning (empty rows, duplicates, conflicting answers, topic spelling, near-duplicates, non-BSCS topic warning)
* Training, prediction, evaluation (held-out data, leakage checks, ablation) scripts
* 249 automated tests (unit, fuzz, concurrency, real-browser) that run offline
* Safe front end (`textContent` only), input validation, no stack traces shown to users, bound to `127.0.0.1`

## 3. Technology stack

Python, Flask, Pandas, scikit-learn (TfidfVectorizer, cosine similarity, LogisticRegression, MultinomialNB), Joblib, Pytest, HTML, CSS, vanilla JavaScript (`fetch`).

## 4. System requirements and installation

* **Python 3.12 recommended** (3.10 to 3.13 work with the bounds in `requirements.txt`; a Windows user reported all tests passing on Python 3.14.3 too; pip picks package versions your Python supports; newer pandas/scikit-learn need Python 3.11+).
* Windows 10/11 with PowerShell (macOS/Linux commands below). **Install it in a local folder such as `C:\StudyBuddy`, not inside OneDrive/Dropbox/Google Drive**: synced folders make start-up slow and could upload student feedback. About 300 MB of disk space. Internet is needed **once** to download packages.

Tested with Python 3.12 on two package sets: the newest (Flask 3.1, pandas 3.0, scikit-learn 1.9, numpy 2.5) and the oldest allowed (Flask 3.0.0, pandas 2.1.4, scikit-learn 1.4.2, numpy 1.26.4). Tests passed on both and evaluation numbers were identical (browser tests skip themselves where Playwright is not installed). A user's Windows run (Python 3.14.3) passed all 235 tests at the time; the PowerShell launcher needs the execution-policy step in the troubleshooting table.

Windows PowerShell, from the project folder (adjust the path):

```powershell
cd C:\path\to\student_study_assistant
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

If PowerShell says *running scripts is disabled*: `Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass`, then activate again (or call `.\.venv\Scripts\python.exe` instead of `python`). If `py -3.12` is not found, use `py -3`.

macOS / Linux:

```bash
python3 -m venv .venv && source .venv/bin/activate
python -m pip install --upgrade pip && python -m pip install -r requirements.txt
```

## 5. Run it (always from the project root folder)

```powershell
python -m ml.preprocess     # 1. clean the CSVs -> data/processed/ (prints rows before/after and warnings)
python -m ml.train          # 2. build the retrieval index + train the topic classifier -> models/
python -m ml.evaluate       # 3. measure on held-out test set A -> results/
python -m ml.evaluate --test-file data/test_questions_v2.csv --tag v2   # 3b. independent test set B
python -m ml.ablation       # 3c. what each matching improvement adds -> results/ablation.md
python -m pytest            # 4. run all automated tests
python app.py               # 5. start the website
```

Open **http://127.0.0.1:5000**. Press `Ctrl+C` to stop. Step 2 is optional for the website: if no saved model exists, or the CSV or synonyms changed, the app rebuilds the index at start-up. It never retrains per chat message.

## 6. How matching works

1. Questions are normalised: lowercase, punctuation and filler words removed (including instruction words such as define/explain/describe and Taglish words such as ano/ang/paano), simple plurals merged, abbreviations expanded from `data/synonyms.csv` (`dbms` to database management system, `quicksort` to quick sort, `vs` / `versus` to "difference between").
2. A trailing context phrase such as "in OS", "in OOP" or "in DBMS" is dropped from the student's question when other words remain, so "what is a deadlock in OS" searches for *deadlock* and is not hijacked by "What is an operating system?".
3. Matching uses TF-IDF vectors compared with cosine similarity, in up to three views: the **words** of the stored question, the stored **question + answer text** (so "subnet mask" can find the subnetting answer), and optionally **character fragments** (so "tokenizing" resembles "tokenization"). Each view is scaled by *coverage* (the share of the student's words found in the knowledge base), because unknown words are otherwise silently ignored and one shared word could look like a perfect match. Final score = 0.35 x question words + 0.65 x answer view (`CHAR_WEIGHT = 0`, `ANSWER_WEIGHT = 0.65` in `config.py`; the character view is implemented and can be switched on).
4. If the best score is at least `SIMILARITY_THRESHOLD` (0.25), that answer is returned. Scores below `HIGH_CONFIDENCE` (0.55) add a caution note. Otherwise the bot says it could not find the question and offers up to three candidates above `SUGGESTION_THRESHOLD` (0.12) as clickable buttons.

The score is a **closeness measure, not a probability or accuracy**. How the settings were chosen: the blend weights, threshold and suggestion threshold were selected on the 110 `dev` rows of `test_questions.csv` by rules fixed before looking at the test rows (lowest "missed + wrong + 2 x false-match" cost; configurations within one question of the best count as ties, then best top-1). Because the bot must stay BSCS-only, wrongly answering a non-BSCS question counts double. The `HIGH_CONFIDENCE` cut-off was read off the evaluation sets, so treat it as a hint.

## 7. Evaluation results (latest run, in `results/`)

Two held-out sets, leakage-checked against the knowledge base and the training data. **Set A**: the 140 `test` rows of `test_questions.csv` (120 in-scope paraphrases + 20 out-of-scope; its other 110 rows were used only to choose settings). **Set B**: `test_questions_v2.csv`, 106 in-scope + 20 out-of-scope, written in a terser style and never used for tuning. Out-of-scope means non-BSCS questions *and* computer-science topics deliberately missing from the knowledge base (GraphQL, Bloom filter, smart contracts...).

| Measure | Set A | Set B |
|---|---|---|
| Top-1 correct stored question | 82.5% | 84.9% |
| Top-3 contains the correct question | 95.0% | 95.3% |
| Top-1 correct subject (of 37) | 95.0% | 94.3% |
| Easy / hard rewordings, top-1 | 89.9% / 61.3% | 87.8% / 75.0% |
| Correct answer returned directly | 70.8% | 79.3% |
| **Helped within one click** (answered correctly, or declined but the correct question was offered as a button) | 83.3% | 84.9% |
| Precision of the answers actually returned | 86.7% | 86.6% |
| Wrong answers returned for in-scope questions | 10 of 120 | 11 of 106 |
| Out-of-scope questions wrongly answered | 3 of 20 | 2 of 20 |
| Topic classifier accuracy (logistic regression / naive Bayes) | 85.8% / 85.0% | 85.9% / 85.9% |
| Average matching time | about 2 ms | about 2 ms |

What each matching ingredient adds (`results/ablation.md`; threshold re-chosen on dev rows for each row):

| Configuration | A top-1 | A top-3 | B top-1 | B top-3 |
|---|---|---|---|---|
| words only (first version) | 59.2% | 69.2% | 63.2% | 69.8% |
| + synonyms only | 68.3% | 78.3% | 73.6% | 81.1% |
| + character fragments only | 61.7% | 72.5% | 63.2% | 74.5% |
| + answer text only | 73.3% | 87.5% | 76.4% | 85.9% |
| current (synonyms + answer text) | 82.5% | 95.0% | 84.9% | 95.3% |
| everything incl. characters | 82.5% | 97.5% | 83.0% | 94.3% |

**Read these carefully.**
* The sets are small and written by the project author; one question moves a figure by about 1 point, and real students will phrase things differently. Set B is more independent than A, but I knew the system's weak spots while writing it. **The most valuable next step is a test set of real student questions** (section 9b).
* The bot declines about a fifth of in-scope questions on set A (25 of 120) and a tenth on set B (11 of 106). That is the price of keeping wrong answers and non-BSCS answers rare; the "Did you mean" buttons recover some of them.
* **Hard rewordings** (different vocabulary from the stored question) are the weak spot: 61-75% correct at top-1. The coverage report shows why: all 534 stored concepts have exactly **one** phrasing each.
* Treating define/explain/describe as filler and `vs` as "difference between" was kept because it improved the dev rows (top-1 +2.6 points) and both test sets without hurting any; four test questions that became trivial duplicates under the new rules were reworded.
* The character-fragment view helps top-3 on set A but not on set B, so its benefit is unproven; the dev-row rule left it switched off.
* The synonym entries were added after seeing failures, so the synonym gain is optimistic.
* Confidence labels: of answers scoring at least 0.55, 93% (51 of 55) were right; of lower-scoring answers returned, 84% (118 of 140). Measured on the same sets, so optimistic.
* The topic classifier trained only on `intent_training.csv` (360 rows) scored 47-50%: too little data for 37 subjects. It is trained on those rows **plus the stored questions**, so its score partly reflects similarity to stored questions. It cannot refuse, so the chat page uses retrieval only.
* See `results/evaluation_report.md` and `evaluation_report_v2.md` for per-subject precision/recall/F1, confusion matrices, failure examples and each metric's limitations.

## 8. Project layout

```
app.py  config.py  requirements.txt  pytest.ini  README.md  pilot/  start_windows.ps1  start_unix.sh  .github/workflows/tests.yml
data/   study_faq.csv  intent_training.csv  synonyms.csv  test_questions.csv  test_questions_v2.csv
        DATA_SOURCES.md  raw/ (original starter files)  processed/ (generated)
ml/     preprocess.py  train.py  evaluate.py  predict.py  ablation.py  import_dataset.py
        feedback.py  feedback_report.py  pilot_status.py  review.py  add_phrasings.py  coverage.py  make_test_file.py  regression_check.py
templates/  base.html index.html quiz.html topics.html about.html error.html
static/ css/style.css  js/chat.js  js/quiz.js     models/ (generated)     results/ (generated reports)     tests/
```

## 9. Add questions and subjects

1. Open `data/study_faq.csv` (Excel or a text editor; save as **CSV UTF-8**). Columns: `question,answer,topic,answer_type`.
2. `topic` must be a BSCS subject listed in `config.py` (`BSCS_SUBJECTS`). To add a new subject (for example Computer Graphics), add it to that list first; `python -m ml.preprocess` warns about any topic not on the list.
3. Add **several phrasings** of important questions as separate rows with the same answer; keep answers short and correct. Do not give one question two different answers (the cleaner keeps the first and warns).
4. Abbreviations students use go in `data/synonyms.csv` as `phrase,replacement` rows.
5. Run `python -m ml.preprocess`, then `python -m ml.train`, then restart `python app.py`.
6. To measure the effect, add **new, unseen** questions to a test file (columns: `question, expected_topic, expected_question` (exact stored question, `|` between acceptable alternatives, blank for out-of-scope), `in_scope` yes/no, `split` dev/test, optional `difficulty`). Never copy training or stored questions there: `python -m ml.evaluate` reports exact and near-duplicates.
7. To grow the classifier, add examples to `data/intent_training.csv` (`text,intent`, same subject names).

Public datasets (SciQ, SQuAD, ARC, Hugging Face) are not BSCS FAQ files and need conversion and manual review. Check each licence first, then use `python -m ml.import_dataset --input data/raw/download.csv --question-col question --answer-col answer --topic "Algorithms" --source "NAME" --license "LICENCE" --output data/raw/converted.csv`, which also records the source in `data/DATA_SOURCES.md`.

## 9b. Improve it over time (the workflow)

**Running a real pilot?** Follow `pilot/PILOT_GUIDE.md`: it covers how to let students use the app (shared screen, classroom network, or form only), what to tell them, how to check progress (`python -m ml.pilot_status`) and how to turn the week's questions into a measured test set. A dry run of this pipeline with 28 invented questions found and fixed four real problems (Tagalog "sa OS" context, a "hashmap" synonym, the labeling export missing wrongly-answered questions, and an evaluation crash on refusal-only test files); those fixes have regression tests, but the dry-run numbers themselves are not evidence of accuracy.

1. **Collect feedback.** Students click *Helpful* / *Not helpful*. Optionally start the app with `$env:STUDYBUDDY_LOG_DECLINED = "1"` to also log questions the bot could not answer (tell students if you do). Then:
   `python -m ml.feedback_report --export data/feedback/needs_label.csv` lists answers marked unhelpful (with what the bot wrongly returned in `bot_matched`) and unanswered questions. The log is a local file, `data/feedback/feedback.jsonl`, and is git-ignored.
2. **Fill the gaps.** Add missing concepts to `study_faq.csv`; add new wordings of existing ones without retyping answers: put `alias,question` rows in a CSV and run `python -m ml.add_phrasings --input data/new_phrasings.csv` (it refuses aliases that match test questions). `python -m ml.coverage` shows the thinnest subjects.
3. **Instructor review.** `python -m ml.review stats`, then `python -m ml.review export --out data/review_todo.csv --subject "Operating Systems"`; the instructor fills `decision` with `ok` or `fix` (plus `corrected_answer`); `python -m ml.review apply data/review_todo.csv` records it (a `.bak` backup is kept). The About page shows how many answers are reviewed.
4. **Measure on real questions.** Label the collected questions (`question, expected_question`; blank or `none` = should be refused), run `python -m ml.make_test_file --input labelled.csv`, then `python -m ml.evaluate --test-file data/test_questions_real.csv --tag real`. Questions already in the knowledge base are dropped to avoid inflated scores.
5. **Guard against regressions.** `python -m ml.regression_check` compares against `results/baseline_metrics.json` and fails if accuracy drops noticeably; after an intentional improvement run `python -m ml.regression_check --update-baseline`. `.github/workflows/tests.yml` runs the tests and this check on every push (I could not run GitHub Actions itself, only validate the file).
6. **Quiz.** The *Quiz* page turns the stored answers into flashcards (one per distinct answer) with a session-only score; nothing is recorded.

**Launchers:** `.\start_windows.ps1` (Windows) or `./start_unix.sh` create the environment on first run, build the model and start the app. The Unix script's syntax was checked; I could not run the PowerShell one.

## 10. Testing

```powershell
python -m pytest                                      # everything (about 40 seconds)
python -m pytest -q --ignore=tests/test_browser.py    # skip the browser tests
```

| File | What it checks |
|---|---|
| `test_preprocessing.py` | CSV loading, missing/invalid columns, cleaning rules, duplicates, conflicts, topic spelling, splits, leakage detection, dataset import, real-data integrity |
| `test_chatbot.py` | Retrieval on a temporary dataset, fallback, thresholds, saved-model reuse/rebuild/corruption, both classifiers, evaluation metrics |
| `test_improvements.py` | Synonyms, character n-grams, answer view, blend-weight validation, context stripping, regression tests for real failures, quality floors on both held-out sets, evaluate/ablation commands |
| `test_bscs_scope.py` | Every topic is a BSCS subject, all 37 subjects have content, core and minor/elective questions land in the right subject, non-BSCS questions are refused |
| `test_workflow_tools.py` | Feedback validation/storage/privacy/threading, declined-logging opt-in, review workflow, phrasing tool, coverage, real-question test-file builder, evaluation on refusal-only / in-scope-only files, pilot status and kit, network-host guard, regression check, quiz API, Taglish and "vs" handling |
| `test_routes.py` | All routes, bad JSON, empty/long input, hidden error details, security headers, no external URLs, safe DOM use |
| `test_robustness.py` | Injection-style strings, 300 random unicode inputs, odd JSON shapes, 400 concurrent requests, accessibility basics |
| `test_browser.py` | Real Chromium (optional: `python -m pip install playwright` then `python -m playwright install chromium`): chat flow, feedback buttons, quiz page, Enter key, Clear Chat, topic filter, HTML stays text, Did-you-mean buttons, no sideways scrolling at 320/390/768/1280 px, friendly errors |

The quality floors are regression guards, not performance claims; performance numbers come only from `python -m ml.evaluate`.

## 11. Troubleshooting

| Problem | Fix |
|---|---|
| `ModuleNotFoundError: flask` (or pandas / sklearn) | Virtual environment not active. Run `.\.venv\Scripts\Activate.ps1`, then `python -m pip install -r requirements.txt`. |
| `No module named ml` / file not found | Run from the project root and use `python -m ml.train` (not `python ml/train.py`). |
| Evaluation says "Configured threshold: 0.2", shows a NOTE about overridden settings, or numbers differ from section 7; `ml.regression_check` says "Refusing to run" | Settings such as `$env:STUDYBUDDY_THRESHOLD` stay active in that PowerShell window until you close it. Check with `Get-ChildItem Env:STUDYBUDDY_*`, then open a **new** PowerShell window (or `Remove-Item Env:STUDYBUDDY_THRESHOLD`). Re-run `python -m ml.evaluate`. If a baseline was saved with overrides active (older versions allowed this), restore it in a clean window with `python -m ml.regression_check --update-baseline`. The tests ignore these settings. |
| The app seems frozen at start, or Ctrl+C shows a `KeyboardInterrupt` inside `sklearn` / `scipy` | It was not frozen: the first start loads large libraries and can take up to a minute, especially inside OneDrive or with antivirus scanning. Wait for the "Running on" line and press Ctrl+C **once** only. A local folder such as `C:\StudyBuddy` (not OneDrive) is much faster (one user's tests took 87 s in OneDrive; they take about 35 s elsewhere). |
| No `PRIVACY WARNING` although the folder is in OneDrive, or `STUDYBUDDY_FEEDBACK_DIR` seems ignored | You are running an older copy of the project. Replace it with the latest zip; the setting and the warning were added together. |
| Placeholder commands such as `ml.make_test_file --input labelled.csv` print "Dataset file not found" | Expected: those files do not exist until you create them (see section 9b). Only run the steps you need. |
| `start_windows.ps1 ... is not digitally signed` | Windows blocks unsigned scripts. Run it once with `powershell -ExecutionPolicy Bypass -File .\start_windows.ps1`, or allow scripts for the current window: `Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass` then `.\start_windows.ps1`. |
| `Invoke-RestMethod : Unable to connect` for `/health` | The server is not running. Start `python app.py` in one PowerShell window and run the health check in a **second** window. Ctrl+C stops the server. |
| `error: unknown option '-m'` after pasting commands | Two commands were pasted on one line. Run them one at a time: first `python -m pip install playwright`, then `python -m playwright install chromium`. |
| Console shows "Similar questions to review ... 1.0" (for example `What is artificial intelligence?` and `Define AI`) | Normal. Synonyms make them identical after cleaning; the cleaner only reports such pairs and keeps both. |
| Classifier "validation accuracy=0.5" in `ml.train` | Normal. That check holds out whole concepts. The chat page uses retrieval only; see section 7 for the held-out topic scores. |
| `PRIVACY WARNING ... cloud-synced folder` at start-up | The project is inside OneDrive, Dropbox, Google Drive or similar. Before a pilot, keep the log elsewhere: `New-Item -ItemType Directory -Force C:\StudyBuddyData; $env:STUDYBUDDY_FEEDBACK_DIR = "C:\StudyBuddyData"` (set it in every PowerShell window you use for the pilot), or move the project out of the synced folder. |
| Activation blocked in PowerShell | `Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass`, then activate again. |
| Port 5000 busy | `$env:STUDYBUDDY_PORT = "5050"; python app.py`, open http://127.0.0.1:5050. |
| Chat says the knowledge base is unavailable | Read the PowerShell window running the app; run `python -m ml.preprocess` for details. |
| `FAQ dataset is missing required column(s)` | The CSV needs headers `question,answer,topic` (and optionally `answer_type`). |
| Warning: topics not in `config.BSCS_SUBJECTS` | Fix the topic spelling, or add the subject to `BSCS_SUBJECTS` if it really is a BSCS subject. |
| Weird characters | Save the CSV as UTF-8. |
| "I could not find that in my BSCS notes" for a valid question | Use the Did-you-mean buttons, add a phrasing or synonym to the CSV, or lower the threshold: `$env:STUDYBUDDY_THRESHOLD = "0.2"; python app.py` (more answers, more wrong ones). |
| A wrong answer was returned | Word overlap found a stored question with shared words. Add the missing phrasing, or raise the threshold. |
| Feedback buttons say "could not be saved" | The feedback file is full (5 MB cap) or the `data/feedback` folder is not writable. Archive the file or fix permissions. |
| Browser tests skipped | Optional: install Playwright and Chromium (section 10). |
| Saved model ignored / rebuilt | Normal after changing the CSV, synonyms or scikit-learn version. Run `python -m ml.train` to refresh. |
| Joblib files are pickles | Only load `.joblib` files that you created yourself. |

## 12. Security notes

Binds to `127.0.0.1` (this computer only) by default. For a classroom pilot you can set `STUDYBUDDY_HOST=0.0.0.0`; the app then prints a warning, has no login, and **refuses to start in debug mode** on a network address. Use that only on a trusted network and only during the pilot; debug mode is off unless `STUDYBUDDY_DEBUG=1` (development only, never expose it); messages limited to 300 characters and cleaned of control characters; replies inserted with `textContent`; Content-Security-Policy allows only same-origin scripts; no database; no code execution of user input; questions are stored only when a student clicks Helpful / Not helpful (the app warns at start-up if that file sits in a cloud-synced folder such as OneDrive, where it could be uploaded; use `STUDYBUDDY_FEEDBACK_DIR` to move it) (or, if the owner enables `STUDYBUDDY_LOG_DECLINED`, when the bot declines one): text, rating and a timestamp in a local file, with no names, IP addresses or browser details; `.venv/` and `.env` are git-ignored.

## 13. Limitations and future improvements

* 534 answers cover the usual first-course concepts of 37 subjects, not every BSCS topic or every exam question; some subjects have only 6-10 stored questions. Questions that need calculation, code writing or multi-step reasoning (for example "write a program to reverse a linked list" or "solve this recurrence") are outside what a retrieval bot can do.
* Words used differently from the stored question (hard rewordings) are the main failure; non-BSCS questions that share vocabulary with BSCS ones are occasionally answered.
* The test sets are small and author-written; use the real-question workflow in section 9b before making strong claims.
* Not done yet (needs new dependencies or real data): local sentence-embedding models (offline, but a larger install; measure on real questions before adopting), an "unknown" class trained on real off-topic questions, multi-turn follow-ups, richer answer formatting.

## 14. Demo checklist

- [ ] `python -m pytest` passes (249 tests)
- [ ] Ask a core question per subject (for example: "What is a deadlock?", "Explain normalization", "What is Big O notation?")
- [ ] Ask something reworded ("difference between stack and queue", "paging in OS")
- [ ] Ask a non-BSCS question ("Who painted the Mona Lisa?") and a missing topic ("What is GraphQL?"); show the polite refusal
- [ ] Show `data/study_faq.csv`, add a row, restart, and ask it; click Helpful and run `python -m ml.feedback_report`
- [ ] Open the Quiz page; run `python -m ml.coverage` and `python -m ml.review stats`
- [ ] Open Topics, About, `/health`, and `results/evaluation_report.md`; explain retrieval vs classification and the small-test-set caveat
#   p g p c - s t u d y - b o t  
 