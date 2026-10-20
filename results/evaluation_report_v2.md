# StudyBuddy evaluation report

Generated: 2026-10-18T03:47:14+00:00  |  scikit-learn 1.9.1  |  seed 42

## 1. Data used

* Knowledge base (retrieval index): 534 questions
* Intent training file: 360 labeled examples
* Held-out questions: 126 total = 0 `dev` (threshold selection only) + 126 `test` (reported)
* Test rows: 106 in-scope, 20 out-of-scope (the bot should say 'no match')

### Leakage checks

* Intent training vs test questions: 0 exact and 0 near-duplicate pairs (token overlap >= 0.8)
* Knowledge base vs test questions: 0 exact and 0 near-duplicate pairs (test questions are paraphrases of stored questions by design, so near matches here are expected but exact copies are not)
* Duplicate questions inside the test set: 0

## 2. Answer retrieval (TF-IDF + cosine similarity)

Configured threshold: **0.25**. Test-split questions only.

| Metric | Value |
|---|---|
| Top-1 exact-question rate (in-scope, ignoring threshold) | 84.9% |
| Top-1 topic rate (in-scope, ignoring threshold) | 94.3% |
| Top-3 exact-question rate | 95.3% |
| Correct answer actually returned (in-scope, with threshold) | 79.2% |
| In-scope questions rejected as 'no match' | 11 |
| ...of those, correct question offered as a 'Did you mean' button | 6 |
| Helped within one click (answered correctly, or correct question offered) | 84.9% |
| Wrong answers returned for in-scope questions | 11 |
| Out-of-scope questions correctly rejected | 90.0% |
| False matches (out-of-scope question answered) | 2 |
| Precision of returned answers | 86.6% |
| Overall decision accuracy (right answer or right rejection) | 81.0% |
| Average matching time | 10.892 ms |

By difficulty (top-1, ignoring threshold):

| Difficulty | n | exact question | topic |
|---|---|---|---|
| easy | 82 | 87.8% | 96.3% |
| hard | 24 | 75.0% | 87.5% |

This file has no `dev` rows, so nothing was tuned here: the configured threshold was used as is.

### Wrong answers returned (test split, configured threshold)

- "functional programming vs oop" -> returned "What is object oriented programming?" (expected ['What is functional programming?'], similarity 0.61)
- "what is a flow chart used for" -> returned "What is a Gantt chart?" (expected ['What is a flowchart?'], similarity 0.292)
- "how to clean a dataset" -> returned "What is a dataset?" (expected ['What is data cleaning?'], similarity 0.282)
- "commit and rollback in transaction" -> returned "What are COMMIT and ROLLBACK?" (expected ['What is a database transaction?'], similarity 0.643)
- "mutex vs semaphore" -> returned "What is a semaphore?" (expected ['What is a mutex?'], similarity 0.364)
- "mqtt protocol iot" -> returned "What is the Internet of Things?" (expected ['What is MQTT?'], similarity 0.503)
- "what does star mean in regular expressions" -> returned "What is a regular expression?" (expected ['What is the Kleene star?'], similarity 0.616)
- "what are reflexive symmetric and transitive relations" -> returned "What is an equivalence relation?" (expected ['What is a relation in discrete mathematics?'], similarity 0.406)
- "explain singly and doubly linked lists" -> returned "What is a doubly linked list?" (expected ['What is a linked list?'], similarity 0.567)
- "what are linearly independent vectors" -> returned "What is the rank of a matrix?" (expected ['What is linear independence?'], similarity 0.299)
- "how do compilers optimise code" -> returned "What is a compiler?" (expected ['What is code optimization?'], similarity 0.329)

### In-scope questions rejected (missed) (test split, configured threshold)

- "what is texturing in 3d graphics" -> closest "What is computer graphics?" (similarity 0.147)
- "how does map reduce work" -> closest "What is a Karnaugh map?" (similarity 0.226)
- "how to calculate f1" -> closest "What is F1 score?" (similarity 0.205)
- "how do I differentiate a composite function" -> closest "What is a function?" (similarity 0.157)
- "how to remove jagged edges in graphics" -> closest "What is anti-aliasing?" (similarity 0.169)
- "how does a 3 to 8 decoder work" -> closest "What is a decoder in digital logic?" (similarity 0.17)
- "what does mod mean a congruent b mod n" -> closest "What is modular arithmetic?" (similarity 0.149)
- "solve linear equations by row reduction" -> closest "What is Gaussian elimination?" (similarity 0.171)
- "what do malloc and free do" -> closest "What is a context free grammar?" (similarity 0.211)
- "what is git used for" -> closest "What is a citation and why is it used?" (similarity 0.14)
- "what are universal and existential quantifiers" -> closest "What is a predicate and a quantifier?" (similarity 0.19)

### False matches on out-of-scope questions (test split, configured threshold)

- "what is homomorphic encryption" -> returned "What is encryption?" (similarity 0.295)
- "what is reinforcement learning from human feedback" -> returned "What is reinforcement learning?" (similarity 0.468)

## 3. Intent classification (supervised, topic labels)

Trained on `intent_training.csv` plus the stored knowledge-base questions (each labeled with its topic); scored on the 106 in-scope test-split questions.

| Model | Accuracy | Macro precision | Macro recall | Macro F1 |
|---|---|---|---|---|
| Logistic regression (intent file + stored questions) | 85.9% | 85.8% | 86.1% | 84.0% |
| Multinomial Naive Bayes (intent file + stored questions) | 85.9% | 86.4% | 86.5% | 84.8% |
| Logistic regression (intent file only) | 47.2% | 60.5% | 52.6% | 47.8% |

Topics with no test questions (excluded from macro averages): none.

Per-class report (logistic regression, intent file + stored questions):

```
                                         precision    recall  f1-score   support

                             Algorithms       0.60      1.00      0.75         3
        Applied Physics and Electronics       1.00      1.00      1.00         2
                Artificial Intelligence       0.67      1.00      0.80         2
                  Compiler Construction       1.00      0.67      0.80         3
 Computer Graphics and Visual Computing       1.00      1.00      1.00         2
                      Computer Networks       0.75      0.75      0.75         4
 Computer Organization and Architecture       1.00      1.00      1.00         4
       Data Mining and Data Warehousing       1.00      1.00      1.00         2
                           Data Science       1.00      1.00      1.00         2
                        Data Structures       1.00      1.00      1.00         5
                       Database Systems       1.00      1.00      1.00         3
                   Digital Logic Design       1.00      0.67      0.80         3
                   Discrete Mathematics       1.00      0.75      0.86         4
                       Game Development       0.67      1.00      0.80         2
             Human Computer Interaction       1.00      1.00      1.00         1
                  IT Project Management       0.67      1.00      0.80         2
                   Information Security       0.83      1.00      0.91         5
Internet of Things and Embedded Systems       1.00      1.00      1.00         2
            Linear Algebra and Calculus       0.75      0.75      0.75         4
                       Machine Learning       1.00      1.00      1.00         3
         Mobile Application Development       1.00      1.00      1.00         2
                       Model Evaluation       0.67      1.00      0.80         2
            Natural Language Processing       0.50      1.00      0.67         1
                      Numerical Methods       1.00      0.50      0.67         2
            Object Oriented Programming       0.80      1.00      0.89         4
                      Operating Systems       1.00      1.00      1.00         4
     Parallel and Distributed Computing       0.00      0.00      0.00         1
             Probability and Statistics       0.80      1.00      0.89         4
                 Professional Practices       1.00      1.00      1.00         2
               Programming Fundamentals       0.80      0.67      0.73         6
                  Programming Languages       1.00      0.50      0.67         2
                   Software Engineering       1.00      0.60      0.75         5
            Systems Analysis and Design       1.00      1.00      1.00         2
 Technical Writing and Research Methods       0.50      0.50      0.50         2
                      Technopreneurship       1.00      1.00      1.00         2
                     Theory of Automata       0.75      1.00      0.86         3
                        Web Development       1.00      0.50      0.67         4

                               accuracy                           0.86       106
                              macro avg       0.86      0.86      0.84       106
                           weighted avg       0.88      0.86      0.85       106
```

Confusion matrix (rows = true topic, columns = predicted topic):

```
                                           0   1   2   3   4   5   6   7   8   9  10  11  12  13  14  15  16  17  18  19  20  21  22  23  24  25  26  27  28  29  30  31  32  33  34  35  36
Algorithms                                 3   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0
Applied Physics and Electronics            0   2   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0
Artificial Intelligence                    0   0   2   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0
Compiler Construction                      0   0   0   2   0   1   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0
Computer Graphics and Visual Computing     0   0   0   0   2   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0
Computer Networks                          1   0   0   0   0   3   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0
Computer Organization and Architecture     0   0   0   0   0   0   4   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0
Data Mining and Data Warehousing           0   0   0   0   0   0   0   2   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0
Data Science                               0   0   0   0   0   0   0   0   2   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0
Data Structures                            0   0   0   0   0   0   0   0   0   5   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0
Database Systems                           0   0   0   0   0   0   0   0   0   0   3   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0
Digital Logic Design                       1   0   0   0   0   0   0   0   0   0   0   2   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0
Discrete Mathematics                       0   0   0   0   0   0   0   0   0   0   0   0   3   0   0   0   0   0   0   0   0   0   1   0   0   0   0   0   0   0   0   0   0   0   0   0   0
Game Development                           0   0   0   0   0   0   0   0   0   0   0   0   0   2   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0
Human Computer Interaction                 0   0   0   0   0   0   0   0   0   0   0   0   0   0   1   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0
IT Project Management                      0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   2   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0
Information Security                       0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   5   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0
Internet of Things and Embedded Systems    0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   2   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0
Linear Algebra and Calculus                0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   3   0   0   0   0   0   0   0   0   0   0   1   0   0   0   0   0   0   0
Machine Learning                           0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   3   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0
Mobile Application Development             0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   2   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0
Model Evaluation                           0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   2   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0
Natural Language Processing                0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   1   0   0   0   0   0   0   0   0   0   0   0   0   0   0
Numerical Methods                          0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   1   0   0   0   0   1   0   0   0   0   0   0   0   0   0   0   0   0   0
Object Oriented Programming                0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   4   0   0   0   0   0   0   0   0   0   0   0   0
Operating Systems                          0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   4   0   0   0   0   0   0   0   0   0   0   0
Parallel and Distributed Computing         0   0   0   0   0   0   0   0   0   0   0   0   0   1   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0
Probability and Statistics                 0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   4   0   0   0   0   0   0   0   0   0
Professional Practices                     0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   2   0   0   0   0   0   0   0   0
Programming Fundamentals                   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   1   0   0   0   0   0   0   0   0   0   0   0   0   0   4   0   0   0   0   0   1   0
Programming Languages                      0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   1   0   0   0   0   0   1   0   0   0   0   0   0
Software Engineering                       0   0   1   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   3   0   1   0   0   0
Systems Analysis and Design                0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   2   0   0   0   0
Technical Writing and Research Methods     0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   1   0   0   0   0   0   1   0   0   0
Technopreneurship                          0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   2   0   0
Theory of Automata                         0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   3   0
Web Development                            0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   1   0   0   0   0   1   0   0   0   0   0   0   0   0   0   0   0   0   0   0   2
```

Column numbers follow the label order: 0=Algorithms, 1=Applied Physics and Electronics, 2=Artificial Intelligence, 3=Compiler Construction, 4=Computer Graphics and Visual Computing, 5=Computer Networks, 6=Computer Organization and Architecture, 7=Data Mining and Data Warehousing, 8=Data Science, 9=Data Structures, 10=Database Systems, 11=Digital Logic Design, 12=Discrete Mathematics, 13=Game Development, 14=Human Computer Interaction, 15=IT Project Management, 16=Information Security, 17=Internet of Things and Embedded Systems, 18=Linear Algebra and Calculus, 19=Machine Learning, 20=Mobile Application Development, 21=Model Evaluation, 22=Natural Language Processing, 23=Numerical Methods, 24=Object Oriented Programming, 25=Operating Systems, 26=Parallel and Distributed Computing, 27=Probability and Statistics, 28=Professional Practices, 29=Programming Fundamentals, 30=Programming Languages, 31=Software Engineering, 32=Systems Analysis and Design, 33=Technical Writing and Research Methods, 34=Technopreneurship, 35=Theory of Automata, 36=Web Development.

## 4. Retrieval versus classification on the same questions

* Topic of the closest stored question (retrieval, no threshold): 94.3%
* Topic predicted by the logistic-regression classifier: 85.9%

These answer different questions. Retrieval returns a prepared answer and can refuse when nothing is close. The classifier always predicts one of the known topics (it has no 'unknown' class), so it cannot detect out-of-scope questions.

## How to read these numbers (limitations)

* **Small, author-written test set.** The questions were written by the project author as
  paraphrases of stored questions plus unrelated questions. Real students may phrase things
  differently, so true performance can be lower (or higher). Percentages from a few dozen
  questions are noisy: one question changes them by several points.
* **Top-1 exact-question rate** counts a hit only if the closest stored question is one the author
  accepted. Near-synonym questions (for example "Define AI") are listed as alternatives, but some
  judgements are subjective.
* **Top-1 topic rate** is more forgiving: a wrong answer in the right topic still counts.
* **Top-3 rate** shows whether the right answer is among the top three candidates; the app
  currently shows only the best one.
* **No-match detection** depends on the threshold. A higher threshold rejects more unrelated
  questions but also rejects more valid paraphrases. The threshold was selected on the `dev` rows (minimising missed + wrong + 2 x false-match errors)
  and the numbers below use only the `test` rows.
* **Cosine similarity is not accuracy or probability.** It only measures word overlap between two
  texts after normalisation. It cannot detect meaning that uses different words.
* **Classifier metrics** (accuracy, precision, recall, F1) describe topic prediction on this small
  test set only. Topics overlap (for example Data Science vs Model Evaluation), the label set is
  imbalanced, and macro averages weight every topic equally even when a topic has few test rows.
* **The classifier does not answer questions.** It predicts a category; retrieval returns a prepared answer.
* **Timing** was measured on one machine and includes only the matching step, not the web request.
