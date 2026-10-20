# StudyBuddy evaluation report

Generated: 2026-10-18T03:46:55+00:00  |  scikit-learn 1.9.1  |  seed 42

## 1. Data used

* Knowledge base (retrieval index): 534 questions
* Intent training file: 360 labeled examples
* Held-out questions: 250 total = 110 `dev` (threshold selection only) + 140 `test` (reported)
* Test rows: 120 in-scope, 20 out-of-scope (the bot should say 'no match')

### Leakage checks

* Intent training vs test questions: 0 exact and 0 near-duplicate pairs (token overlap >= 0.8)
* Knowledge base vs test questions: 0 exact and 0 near-duplicate pairs (test questions are paraphrases of stored questions by design, so near matches here are expected but exact copies are not)
* Duplicate questions inside the test set: 0

## 2. Answer retrieval (TF-IDF + cosine similarity)

Configured threshold: **0.25**. Test-split questions only.

| Metric | Value |
|---|---|
| Top-1 exact-question rate (in-scope, ignoring threshold) | 82.5% |
| Top-1 topic rate (in-scope, ignoring threshold) | 95.0% |
| Top-3 exact-question rate | 95.0% |
| Correct answer actually returned (in-scope, with threshold) | 70.8% |
| In-scope questions rejected as 'no match' | 25 |
| ...of those, correct question offered as a 'Did you mean' button | 15 |
| Helped within one click (answered correctly, or correct question offered) | 83.3% |
| Wrong answers returned for in-scope questions | 10 |
| Out-of-scope questions correctly rejected | 85.0% |
| False matches (out-of-scope question answered) | 3 |
| Precision of returned answers | 86.7% |
| Overall decision accuracy (right answer or right rejection) | 72.9% |
| Average matching time | 8.418 ms |

By difficulty (top-1, ignoring threshold):

| Difficulty | n | exact question | topic |
|---|---|---|---|
| easy | 89 | 89.9% | 98.9% |
| hard | 31 | 61.3% | 83.9% |

Threshold selected on dev rows: **0.25**. Test-split results at that threshold: decision accuracy 72.9%, false matches 3, in-scope rejected 25, wrong answers 10.

### Threshold sweep on dev rows

| Threshold | Cost | Decision accuracy | False matches | Wrong answers | In-scope rejected |
|---|---|---|---|---|---|
| 0.05 | 52 | 70.0% | 19 | 14 | 0 |
| 0.1 | 42 | 74.6% | 14 | 14 | 0 |
| 0.15 | 29 | 80.0% | 7 | 12 | 3 |
| 0.2 | 24 | 81.8% | 4 | 8 | 8 |
| 0.25 | 24 | 80.0% | 2 | 4 | 16 |
| 0.3 | 30 | 73.6% | 1 | 2 | 26 |
| 0.35 | 34 | 70.0% | 1 | 1 | 31 |
| 0.4 | 39 | 65.5% | 1 | 1 | 36 |
| 0.45 | 44 | 60.0% | 0 | 1 | 43 |
| 0.5 | 51 | 53.6% | 0 | 1 | 50 |
| 0.55 | 57 | 48.2% | 0 | 0 | 57 |
| 0.6 | 66 | 40.0% | 0 | 0 | 66 |
| 0.65 | 68 | 38.2% | 0 | 0 | 68 |
| 0.7 | 71 | 35.4% | 0 | 0 | 71 |
| 0.75 | 74 | 32.7% | 0 | 0 | 74 |
| 0.8 | 75 | 31.8% | 0 | 0 | 75 |

### Wrong answers returned (test split, configured threshold)

- "What is a mux in digital electronics?" -> returned "What is a counter in digital electronics?" (expected ['What is a multiplexer?'], similarity 0.394)
- "What is the formula relating voltage current and resistance?" -> returned "What is the difference between voltage, current and resistance?" (expected ["What is Ohm's law?"], similarity 0.403)
- "Explain Newton's method for finding roots" -> returned "What is Euler's method?" (expected ['What is the Newton Raphson method?'], similarity 0.288)
- "What is a rational agent in AI?" -> returned "Give examples of artificial intelligence" (expected ['What is an intelligent agent?'], similarity 0.327)
- "What makes an API RESTful?" -> returned "What is an API?" (expected ['What is a REST API?'], similarity 0.288)
- "Explain agile software development" -> returned "What is the software development life cycle?" (expected ['What is agile methodology?'], similarity 0.394)
- "Why do computers use a cache?" -> returned "What is a cache hit and a cache miss?" (expected ['What is cache memory?'], similarity 0.282)
- "How does a compiler translate programs?" -> returned "What is the difference between a compiler and an interpreter?" (expected ['What is a compiler?'], similarity 0.363)
- "What is feasibility analysis in system design?" -> returned "What is systems analysis and design?" (expected ['What is a feasibility study?'], similarity 0.381)
- "Explain the LIFO stack data structure" -> returned "What is a data structure?" (expected ['What is a stack?'], similarity 0.455)

### In-scope questions rejected (missed) (test split, configured threshold)

- "How does a child class inherit from a parent class?" -> closest "What is a class?" (similarity 0.227)
- "Why should code be refactored?" -> closest "What is a code review?" (similarity 0.173)
- "How does automatic memory reclamation work in Java?" -> closest "What is cache memory?" (similarity 0.12)
- "How do I prove something by induction?" -> closest "What is mathematical induction?" (similarity 0.226)
- "How is probability calculated given another event has happened?" -> closest "What is conditional probability?" (similarity 0.208)
- "What are mnemonics like MOV and ADD?" -> closest "What is assembly language?" (similarity 0.188)
- "How do games detect when objects collide?" -> closest "What is an object?" (similarity 0.193)
- "Explain learning with labelled data" -> closest "What is supervised learning?" (similarity 0.236)
- "How do I cite sources in APA format?" -> closest "What is APA referencing style?" (similarity 0.158)
- "How do try and catch work?" -> closest "What is exception handling?" (similarity 0.176)
- "What does the lexical analyzer do?" -> closest "What is lexical analysis?" (similarity 0.22)
- "How does a ray tracer create realistic images?" -> closest "What is ray tracing?" (similarity 0.204)
- "Why is a logic mistake different from a syntax mistake?" -> closest "What is the difference between a syntax error and a logic error?" (similarity 0.163)
- "What is the bell curve?" -> closest "What is a derivative?" (similarity 0.09)
- "Why test modules together after combining them?" -> closest "What is integration testing?" (similarity 0.184)
- "What is ethical hacking?" -> closest "What is a variable?" (similarity 0.0)
- "How does the Apriori algorithm find frequent itemsets?" -> closest "What is an algorithm?" (similarity 0.109)
- "How are negative numbers stored using 2s complement?" -> closest "What is two's complement?" (similarity 0.229)
- "What is IP protection like patents and copyrights?" -> closest "What is copyright in software?" (similarity 0.147)
- "How are triangles turned into pixels?" -> closest "What is a pixel?" (similarity 0.206)
- "What is the greedy approach to problem solving?" -> closest "What is a greedy algorithm?" (similarity 0.188)
- "How does minimax pick the best move?" -> closest "What is the minimax algorithm?" (similarity 0.182)
- "What happens when the CPU switches between processes?" -> closest "What is the difference between a process and a thread?" (similarity 0.216)
- "How do I query rows with SELECT?" -> closest "What is a SELECT statement?" (similarity 0.249)
- "Why can't we decide whether a program halts?" -> closest "What are system testing and acceptance testing?" (similarity 0.115)

### False matches on out-of-scope questions (test split, configured threshold)

- "What is a convolutional neural network?" -> returned "What is a neural network?" (similarity 0.433)
- "What is a B plus tree split?" -> returned "What is a B tree index?" (similarity 0.255)
- "What is quantum computing?" -> returned "What is cloud computing?" (similarity 0.25)

## 3. Intent classification (supervised, topic labels)

Trained on `intent_training.csv` plus the stored knowledge-base questions (each labeled with its topic); scored on the 120 in-scope test-split questions.

| Model | Accuracy | Macro precision | Macro recall | Macro F1 |
|---|---|---|---|---|
| Logistic regression (intent file + stored questions) | 85.8% | 87.4% | 86.1% | 84.4% |
| Multinomial Naive Bayes (intent file + stored questions) | 85.0% | 86.6% | 84.4% | 82.8% |
| Logistic regression (intent file only) | 50.0% | 63.6% | 61.2% | 55.0% |

Topics with no test questions (excluded from macro averages): Data Science.

Per-class report (logistic regression, intent file + stored questions):

```
                                         precision    recall  f1-score   support

                             Algorithms       0.90      1.00      0.95         9
        Applied Physics and Electronics       1.00      1.00      1.00         2
                Artificial Intelligence       0.33      1.00      0.50         2
                  Compiler Construction       0.60      1.00      0.75         3
 Computer Graphics and Visual Computing       1.00      1.00      1.00         3
                      Computer Networks       0.83      0.83      0.83         6
 Computer Organization and Architecture       0.83      0.83      0.83         6
       Data Mining and Data Warehousing       1.00      0.50      0.67         2
                           Data Science       0.00      0.00      0.00         0
                        Data Structures       1.00      1.00      1.00         4
                       Database Systems       0.86      1.00      0.92         6
                   Digital Logic Design       1.00      1.00      1.00         4
                   Discrete Mathematics       1.00      1.00      1.00         5
                       Game Development       1.00      1.00      1.00         2
             Human Computer Interaction       1.00      1.00      1.00         1
                  IT Project Management       1.00      1.00      1.00         2
                   Information Security       0.00      0.00      0.00         2
Internet of Things and Embedded Systems       1.00      1.00      1.00         2
            Linear Algebra and Calculus       1.00      1.00      1.00         5
                       Machine Learning       1.00      1.00      1.00         2
         Mobile Application Development       1.00      1.00      1.00         2
                       Model Evaluation       0.33      1.00      0.50         1
            Natural Language Processing       1.00      1.00      1.00         1
                      Numerical Methods       0.67      1.00      0.80         2
            Object Oriented Programming       0.75      0.75      0.75         4
                      Operating Systems       0.86      1.00      0.92         6
     Parallel and Distributed Computing       1.00      1.00      1.00         2
             Probability and Statistics       1.00      0.60      0.75         5
                 Professional Practices       0.50      0.50      0.50         2
               Programming Fundamentals       1.00      0.67      0.80         6
                  Programming Languages       1.00      1.00      1.00         2
                   Software Engineering       1.00      0.50      0.67         6
            Systems Analysis and Design       1.00      1.00      1.00         2
 Technical Writing and Research Methods       1.00      0.50      0.67         2
                      Technopreneurship       1.00      1.00      1.00         2
                     Theory of Automata       1.00      0.80      0.89         5
                        Web Development       1.00      0.50      0.67         2

                               accuracy                           0.86       120
                              macro avg       0.85      0.84      0.82       120
                           weighted avg       0.90      0.86      0.86       120
```

Confusion matrix (rows = true topic, columns = predicted topic):

```
                                           0   1   2   3   4   5   6   7   8   9  10  11  12  13  14  15  16  17  18  19  20  21  22  23  24  25  26  27  28  29  30  31  32  33  34  35  36
Algorithms                                 9   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0
Applied Physics and Electronics            0   2   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0
Artificial Intelligence                    0   0   2   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0
Compiler Construction                      0   0   0   3   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0
Computer Graphics and Visual Computing     0   0   0   0   3   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0
Computer Networks                          0   0   1   0   0   5   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0
Computer Organization and Architecture     0   0   1   0   0   0   5   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0
Data Mining and Data Warehousing           0   0   1   0   0   0   0   1   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0
Data Science                               0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0
Data Structures                            0   0   0   0   0   0   0   0   0   4   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0
Database Systems                           0   0   0   0   0   0   0   0   0   0   6   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0
Digital Logic Design                       0   0   0   0   0   0   0   0   0   0   0   4   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0
Discrete Mathematics                       0   0   0   0   0   0   0   0   0   0   0   0   5   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0
Game Development                           0   0   0   0   0   0   0   0   0   0   0   0   0   2   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0
Human Computer Interaction                 0   0   0   0   0   0   0   0   0   0   0   0   0   0   1   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0
IT Project Management                      0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   2   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0
Information Security                       0   0   1   0   0   0   0   0   0   0   1   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0
Internet of Things and Embedded Systems    0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   2   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0
Linear Algebra and Calculus                0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   5   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0
Machine Learning                           0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   2   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0
Mobile Application Development             0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   2   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0
Model Evaluation                           0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   1   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0
Natural Language Processing                0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   1   0   0   0   0   0   0   0   0   0   0   0   0   0   0
Numerical Methods                          0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   2   0   0   0   0   0   0   0   0   0   0   0   0   0
Object Oriented Programming                0   0   0   0   0   0   0   0   1   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   3   0   0   0   0   0   0   0   0   0   0   0   0
Operating Systems                          0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   6   0   0   0   0   0   0   0   0   0   0   0
Parallel and Distributed Computing         0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   2   0   0   0   0   0   0   0   0   0   0
Probability and Statistics                 0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   1   0   1   0   0   0   3   0   0   0   0   0   0   0   0   0
Professional Practices                     0   0   0   0   0   1   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   1   0   0   0   0   0   0   0   0
Programming Fundamentals                   1   0   0   1   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   4   0   0   0   0   0   0   0
Programming Languages                      0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   2   0   0   0   0   0   0
Software Engineering                       0   0   0   1   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   1   0   0   0   1   0   0   0   0   0   3   0   0   0   0   0
Systems Analysis and Design                0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   2   0   0   0   0
Technical Writing and Research Methods     0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   1   0   0   0   0   1   0   0   0
Technopreneurship                          0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   2   0   0
Theory of Automata                         0   0   0   0   0   0   1   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   4   0
Web Development                            0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   1   0   0   0   0   0   0   0   0   0   0   0   1
```

Column numbers follow the label order: 0=Algorithms, 1=Applied Physics and Electronics, 2=Artificial Intelligence, 3=Compiler Construction, 4=Computer Graphics and Visual Computing, 5=Computer Networks, 6=Computer Organization and Architecture, 7=Data Mining and Data Warehousing, 8=Data Science, 9=Data Structures, 10=Database Systems, 11=Digital Logic Design, 12=Discrete Mathematics, 13=Game Development, 14=Human Computer Interaction, 15=IT Project Management, 16=Information Security, 17=Internet of Things and Embedded Systems, 18=Linear Algebra and Calculus, 19=Machine Learning, 20=Mobile Application Development, 21=Model Evaluation, 22=Natural Language Processing, 23=Numerical Methods, 24=Object Oriented Programming, 25=Operating Systems, 26=Parallel and Distributed Computing, 27=Probability and Statistics, 28=Professional Practices, 29=Programming Fundamentals, 30=Programming Languages, 31=Software Engineering, 32=Systems Analysis and Design, 33=Technical Writing and Research Methods, 34=Technopreneurship, 35=Theory of Automata, 36=Web Development.

## 4. Retrieval versus classification on the same questions

* Topic of the closest stored question (retrieval, no threshold): 95.0%
* Topic predicted by the logistic-regression classifier: 85.8%

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
