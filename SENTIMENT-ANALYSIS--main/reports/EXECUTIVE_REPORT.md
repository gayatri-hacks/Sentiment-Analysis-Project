# Climate-change stance classifier: executive verification report

## Scope

This is a four-class **stance** classifier for climate-change tweets (Anti, Neutral, Pro, News). It does not measure generic positive/negative sentiment or sentiment intensity.

## Dataset and quality checks

- Source: Repository-provided Twitter climate-change stance dataset; labels follow the project dataset documentation.
- CSV: `src/twitter_sentiment_data.csv`; original rows: 43,943; usable rows: 43,943.
- Missing message/label rows: 0; missing tweet IDs: 0; empty messages: 0.
- Exact-text duplicate rows: 3,455; duplicate groups beyond their first row: 2,915. Exact-text groups were kept wholly within one partition.

## Independent held-out test results

- Test rows: 6,611; accuracy: 0.709; macro F1: 0.623; weighted F1: 0.694.

| Stance | Precision | Recall | F1 | Support |
| --- | ---: | ---: | ---: | ---: |
| Anti | 0.684 | 0.405 | 0.509 | 573 |
| Neutral | 0.600 | 0.392 | 0.474 | 1211 |
| Pro | 0.720 | 0.880 | 0.792 | 3458 |
| News | 0.751 | 0.686 | 0.717 | 1369 |

## Method and limitations

Training uses a deterministic grouped exact-message 70/15/15 split. TF-IDF fitting and oversampling occur only on training rows. A sigmoid probability calibration is fit on the validation partition; the test partition is untouched until final evaluation. Displayed probabilities are calibrated model estimates, not guarantees of correctness.

This baseline may be affected by sarcasm, historical Twitter language, annotation noise, retweets, and temporal drift. The evaluation is neither a temporal holdout nor external validation; it is not evidence of production readiness.

## Business reporting use

Use the dashboard to summarize stance mix in a supplied message file, inspect filtered records, and export reproducible CSV results. Escalate high-impact decisions for human review; do not use this model to infer an individual’s beliefs or as a generic sentiment score.
