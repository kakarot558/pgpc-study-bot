# Ablation: what does each improvement add?

Threshold per row chosen on dev rows only. Test set A = `test` rows of test_questions.csv (140 questions). Test set B = test_questions_v2.csv (126 questions, never used to choose settings).

| Configuration | Threshold | A top-1 | A top-3 | A decision acc. | A false matches | B top-1 | B top-3 | B decision acc. | B false matches |
|---|---|---|---|---|---|---|---|---|---|
| words only (first version) | 0.3 | 59.2% | 69.2% | 55.7% | 2 | 63.2% | 69.8% | 61.1% | 3 |
| + synonyms only | 0.35 | 68.3% | 78.3% | 61.4% | 2 | 73.6% | 81.1% | 65.9% | 2 |
| + character n-grams only (weight 0.35) | 0.45 | 61.7% | 72.5% | 49.3% | 1 | 63.2% | 74.5% | 56.4% | 2 |
| + answer text only | 0.25 | 73.3% | 87.5% | 63.6% | 3 | 76.4% | 85.9% | 72.2% | 2 |
| current settings (synonyms + answer text) | 0.25 | 82.5% | 95.0% | 72.9% | 3 | 84.9% | 95.3% | 81.0% | 2 |
| everything incl. characters (0.2) | 0.25 | 82.5% | 97.5% | 76.4% | 3 | 83.0% | 94.3% | 80.2% | 2 |

Caveats: both test sets are small (a single question moves a figure by about 2 points). The synonym entries were written after seeing which questions failed, so the synonym gain is optimistic; the blend weights and thresholds were chosen on dev rows only.