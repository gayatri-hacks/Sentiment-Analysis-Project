# Climate-change stance dashboard

This is a four-class **climate-change stance** classifier for tweets, not a generic positive/negative sentiment application. It uses the repository's `src/twitter_sentiment_data.csv`, which corresponds to the [Twitter Climate Change Sentiment Dataset](https://www.kaggle.com/datasets/edqian/twitter-climate-change-sentiment-dataset). The supplied CSV has 43,943 rows and `sentiment`, `message`, and `tweetid` columns.

| Label | Meaning |
| --- | --- |
| -1 | Anti — rejects or doubts man-made climate change |
| 0 | Neutral — neither supports nor rejects it |
| 1 | Pro — supports the belief in man-made climate change |
| 2 | News — factual climate-change news |

## Reproducible training

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt pytest
python -m nltk.downloader stopwords
python -m src.sentiment_analysis
pytest -q
streamlit run streamlit_dashboard.py
```

Training uses a fixed seed (`42`) and splits original tweets before vectorization and oversampling. Exact duplicate message groups are kept in only one of train, validation, or test. The approximate partition is 70/15/15; TF-IDF fitting and random oversampling happen only on training rows. Results, split distributions, preprocessing/data-quality counts, label map, and held-out metrics are saved in `models/metadata.json`. Re-running training replaces the committed model artifacts with results from the same real dataset.

The dashboard supports individual stance predictions, read-only exploration of the original dataset, and batch CSV prediction (input needs a `message` column). It uses the exact same preprocessing module as training. Probabilities shown are `LogisticRegression.predict_proba` outputs and are not described as guaranteed confidence.

## Docker

Train first so `models/` contains current artifacts, then:

```bash
docker build -t climate-stance-dashboard .
docker run --rm -p 8080:8080 climate-stance-dashboard
```

Open `http://localhost:8080`. The image includes the real CSV, trained artifacts, and required NLTK resources. Its health endpoint is `/_stcore/health`.

## Tests and CI

`pytest` covers real-data loading, label mapping, preprocessing determinism, duplicate-group leakage prevention, artifact loading, single predictions, batch predictions, and saved evaluation metadata. GitHub Actions retrains with the original dataset, runs the tests, builds the image, and smoke-tests the Streamlit health endpoint.

## Limitations

This bag-of-words baseline can struggle with sarcasm, irony, evolving climate discourse, mislabeled tweets, and retweet-heavy data. Class probabilities are not independently calibrated. The stored evaluation is a grouped held-out split, not a temporal or external validation.
