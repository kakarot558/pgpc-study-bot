# One-week pilot guide

Goal: collect **20-30 real student questions** (from 5 or more students, at least 5 subjects), see
where StudyBuddy fails, fix the gaps, and measure on questions the bot has never seen.
You do not need to be online; everything stays on your computer.

## Day 0 - set up (15 minutes)
0. **Keep the feedback log off cloud sync.** If your project folder is inside OneDrive, Dropbox or Google Drive, the students' questions could be uploaded. The app prints a `PRIVACY WARNING` when that is the case. Fix it in every PowerShell window you use for the pilot:
   ```powershell
   New-Item -ItemType Directory -Force C:\StudyBuddyData
   $env:STUDYBUDDY_FEEDBACK_DIR = "C:\StudyBuddyData"
   ```
   (and export the labelling file there too: `--export C:\StudyBuddyData\needs_label.csv`).
1. Install and test: follow README section 4, then `python -m pytest -q --ignore=tests/test_browser.py`.
2. Print or share `pilot/student_notice.txt`. Tell students what is saved (question text, rating, time).
3. Decide how students will use it (pick one):
   * **A. Shared screen (simplest, safest):** run the app on your laptop (`python app.py`) and let
     students type on your machine during a study session. Nothing leaves your computer.
   * **B. Classroom network:** students open it on their own phones/laptops.
     Windows PowerShell on your computer:
     ```powershell
     $env:STUDYBUDDY_HOST = "0.0.0.0"
     python app.py
     ipconfig        # note your "IPv4 Address", e.g. 192.168.1.23
     ```
     Students open `http://192.168.1.23:5000`. Allow Python through the Windows firewall on **private**
     networks only. There is no login, so use a trusted network, run it only during the pilot, and
     stop it with Ctrl+C afterwards. (The app refuses to start in debug mode on a network address.)
   * **C. No app at all:** hand out `pilot/question_collection_form.md` and gather questions only.
     You lose the helpful/not-helpful ratings but still get a real test set.
4. Optional: also record unanswered questions automatically: `$env:STUDYBUDDY_LOG_DECLINED = "1"`
   before `python app.py`. Tell students (the notice already does).

## Days 1-6 - collect
* Ask each student to try 5-6 questions about their current subjects, then click Helpful / Not helpful.
* Encourage variety: different subjects, informal wording, Taglish, a few off-topic questions.
* Check progress any time: `python -m ml.pilot_status` (it tells you when you have enough).
* Do not fix anything during the week unless something is broken: fixing as you go makes the final
  measurement meaningless.

## Day 7 - analyse
1. `python -m ml.feedback_report --export data/feedback/needs_label.csv` (or `--export C:\StudyBuddyData\needs_label.csv` if you moved the log)
   shows answers rated unhelpful and questions the bot could not answer. The CSV also has a
   `bot_matched` column (what it wrongly returned).
2. **Label them** (an instructor or a strong student): open the CSV, and for each question write in
   `expected_question` the EXACT stored question that answers it (copy it from `data/study_faq.csv`;
   use `|` between acceptable alternatives). Leave it blank if the bot should refuse. Also add the
   questions students rated Helpful if you want a balanced test set (copy them in from
   `data/feedback/feedback.jsonl`, or from the form answers in `pilot/collected_questions.csv`).
3. `python -m ml.make_test_file --input labelled.csv --output data/test_questions_real.csv`
4. `python -m ml.evaluate --test-file data/test_questions_real.csv --tag real`
   Read `results/evaluation_report_real.md`. These are the first numbers that describe real students.
5. For questions that should have matched but did not, add the student's wording as a new phrasing:
   `python -m ml.add_phrasings --input data/new_phrasings.csv` (CSV columns: `alias,question`).
   For missing concepts, add rows to `data/study_faq.csv`, then `python -m ml.preprocess && python -m ml.train`.
6. **Do not re-use the same questions to both fix and test.** Keep `test_questions_real.csv` for
   measuring; collect a second small batch to confirm an improvement.
7. Clean up: stop the app, archive or delete `data/feedback/` (it holds the students' question text).

## What to report
* Number of students and questions; subjects covered.
* Top-1 and top-3 accuracy, answers marked helpful vs not helpful, refusals, from `evaluation_report_real.md`.
* The five most common failure types and what you changed.
* Limits: a handful of students is a small sample; say so.
