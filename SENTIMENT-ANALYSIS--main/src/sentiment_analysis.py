"""Train a leakage-resistant climate-change stance classifier."""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import joblib
import pandas as pd
from imblearn.over_sampling import RandomOverSampler
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import CalibratedClassifierCV
from sklearn.frozen import FrozenEstimator
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.model_selection import GroupShuffleSplit

from src.text_preprocessing import clean_text, ensure_nltk_resources

ROOT = Path(__file__).resolve().parents[1]
DATASET_PATH = ROOT / "src" / "twitter_sentiment_data.csv"
MODELS_DIR = ROOT / "models"
LABEL_MAP = {-1: "Anti", 0: "Neutral", 1: "Pro", 2: "News"}
SEED = 42


def load_data(path: Path = DATASET_PATH) -> tuple[pd.DataFrame, dict]:
    """Load and validate the repository-provided Twitter stance CSV."""
    if not path.is_file():
        raise FileNotFoundError(f"Dataset not found: {path}")
    data = pd.read_csv(path)
    required = {"message", "sentiment", "tweetid"}
    missing = required.difference(data.columns)
    if missing:
        raise ValueError(f"Dataset is missing required columns: {sorted(missing)}")
    initial_rows = len(data)
    missing_rows = int(data[["message", "sentiment"]].isna().any(axis=1).sum())
    missing_message_rows = int(data["message"].isna().sum())
    missing_label_rows = int(data["sentiment"].isna().sum())
    missing_tweetid_rows = int(data["tweetid"].isna().sum())
    data = data.dropna(subset=["message", "sentiment"]).copy()
    data["sentiment"] = pd.to_numeric(data["sentiment"], errors="raise").astype(int)
    invalid = sorted(set(data["sentiment"]).difference(LABEL_MAP))
    if invalid:
        raise ValueError(f"Invalid stance labels: {invalid}; expected {sorted(LABEL_MAP)}")
    data["message"] = data["message"].astype(str)
    data["clean_text"] = data["message"].map(clean_text)
    data["split_group"] = data["message"].str.strip().str.casefold()
    stats = {
        "initial_rows": initial_rows,
        "rows_after_missing_value_removal": len(data),
        "rows_with_missing_message_or_sentiment": missing_rows,
        "rows_with_missing_message": missing_message_rows,
        "rows_with_missing_label": missing_label_rows,
        "rows_with_missing_tweetid": missing_tweetid_rows,
        "empty_message_rows": int(data["message"].str.strip().eq("").sum()),
        "unique_exact_message_groups": int(data["split_group"].nunique()),
        "duplicate_message_rows": int(data.duplicated("split_group", keep=False).sum()),
        "duplicate_message_groups": int(data["split_group"].duplicated().sum()),
        "class_distribution": {str(k): int(v) for k, v in data.sentiment.value_counts().sort_index().items()},
    }
    return data.reset_index(drop=True), stats


def grouped_split(data: pd.DataFrame, seed: int = SEED) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Make roughly 70/15/15 splits without splitting identical original tweets."""
    first = GroupShuffleSplit(n_splits=1, test_size=0.30, random_state=seed)
    train_idx, holdout_idx = next(first.split(data, groups=data.split_group))
    train, holdout = data.iloc[train_idx].copy(), data.iloc[holdout_idx].copy()
    second = GroupShuffleSplit(n_splits=1, test_size=0.50, random_state=seed)
    validation_idx, test_idx = next(second.split(holdout, groups=holdout.split_group))
    validation, test = holdout.iloc[validation_idx].copy(), holdout.iloc[test_idx].copy()
    for left, right in ((train, validation), (train, test), (validation, test)):
        if set(left.split_group).intersection(right.split_group):
            raise RuntimeError("Duplicate original tweets crossed a dataset partition")
    return train, validation, test


def distribution(frame: pd.DataFrame) -> dict[str, int]:
    return {str(k): int(v) for k, v in frame.sentiment.value_counts().sort_index().items()}


def write_executive_report(metadata: dict, output_path: Path) -> None:
    """Write a concise, data-derived Markdown hand-off for business stakeholders."""
    quality = metadata["data_quality"]
    evaluation = metadata["evaluation"]
    per_class = evaluation["per_class"]
    rows = [
        "# Climate-change stance classifier: executive verification report",
        "",
        "## Scope",
        "",
        "This is a four-class **stance** classifier for climate-change tweets (Anti, Neutral, Pro, News). It does not measure generic positive/negative sentiment or sentiment intensity.",
        "",
        "## Dataset and quality checks",
        "",
        f"- Source: {metadata['dataset']['provenance']}",
        f"- CSV: `{metadata['dataset']['path']}`; original rows: {quality['initial_rows']:,}; usable rows: {quality['rows_after_missing_value_removal']:,}.",
        f"- Missing message/label rows: {quality['rows_with_missing_message_or_sentiment']:,}; missing tweet IDs: {quality['rows_with_missing_tweetid']:,}; empty messages: {quality['empty_message_rows']:,}.",
        f"- Exact-text duplicate rows: {quality['duplicate_message_rows']:,}; duplicate groups beyond their first row: {quality['duplicate_message_groups']:,}. Exact-text groups were kept wholly within one partition.",
        "",
        "## Independent held-out test results",
        "",
        f"- Test rows: {sum(item['support'] for key, item in per_class.items() if key in LABEL_MAP.values()):,.0f}; accuracy: {evaluation['accuracy']:.3f}; macro F1: {evaluation['macro_f1']:.3f}; weighted F1: {evaluation['weighted_f1']:.3f}.",
        "",
        "| Stance | Precision | Recall | F1 | Support |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for label in LABEL_MAP.values():
        item = per_class[label]
        rows.append(f"| {label} | {item['precision']:.3f} | {item['recall']:.3f} | {item['f1-score']:.3f} | {item['support']:.0f} |")
    rows += [
        "",
        "## Method and limitations",
        "",
        "Training uses a deterministic grouped exact-message 70/15/15 split. TF-IDF fitting and oversampling occur only on training rows. A sigmoid probability calibration is fit on the validation partition; the test partition is untouched until final evaluation. Displayed probabilities are calibrated model estimates, not guarantees of correctness.",
        "",
        "This baseline may be affected by sarcasm, historical Twitter language, annotation noise, retweets, and temporal drift. The evaluation is neither a temporal holdout nor external validation; it is not evidence of production readiness.",
        "",
        "## Business reporting use",
        "",
        "Use the dashboard to summarize stance mix in a supplied message file, inspect filtered records, and export reproducible CSV results. Escalate high-impact decisions for human review; do not use this model to infer an individual’s beliefs or as a generic sentiment score.",
    ]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(rows) + "\n", encoding="utf-8")


def train(dataset_path: Path = DATASET_PATH, models_dir: Path = MODELS_DIR) -> dict:
    ensure_nltk_resources()
    data, data_quality = load_data(dataset_path)
    train_frame, validation_frame, test_frame = grouped_split(data)
    vectorizer = TfidfVectorizer(max_features=30000, ngram_range=(1, 2), min_df=2, max_df=0.95, sublinear_tf=True)
    x_train = vectorizer.fit_transform(train_frame.clean_text)
    x_validation = vectorizer.transform(validation_frame.clean_text)
    x_test = vectorizer.transform(test_frame.clean_text)
    sampler = RandomOverSampler(random_state=SEED)
    x_train_balanced, y_train_balanced = sampler.fit_resample(x_train, train_frame.sentiment)
    base_model = LogisticRegression(max_iter=2000, solver="lbfgs", random_state=SEED)
    base_model.fit(x_train_balanced, y_train_balanced)
    # Validation is deliberately reserved for calibration, never used to fit the
    # vectorizer or base classifier. The test partition remains fully untouched.
    model = CalibratedClassifierCV(estimator=FrozenEstimator(base_model), method="sigmoid")
    model.fit(x_validation, validation_frame.sentiment)
    labels = list(LABEL_MAP)
    validation_prediction = model.predict(x_validation)
    prediction = model.predict(x_test)
    report = classification_report(test_frame.sentiment, prediction, labels=labels,
        target_names=[LABEL_MAP[x] for x in labels], output_dict=True, zero_division=0)
    evaluation = {
        "validation_accuracy": float(accuracy_score(validation_frame.sentiment, validation_prediction)),
        "accuracy": float(accuracy_score(test_frame.sentiment, prediction)),
        "macro_f1": float(f1_score(test_frame.sentiment, prediction, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(test_frame.sentiment, prediction, average="weighted", zero_division=0)),
        "per_class": report,
        "confusion_matrix_labels": labels,
        "confusion_matrix": confusion_matrix(test_frame.sentiment, prediction, labels=labels).tolist(),
    }
    models_dir.mkdir(parents=True, exist_ok=True)
    metadata = {
        "dataset": {"path": str(dataset_path.relative_to(ROOT)), "provenance": "Repository-provided Twitter climate-change stance dataset; labels follow the project dataset documentation."},
        "label_map": {str(k): v for k, v in LABEL_MAP.items()}, "random_seed": SEED,
        "split": "Grouped exact-message split of original rows: approximately 70% train, 15% validation, 15% test. TF-IDF fit and random oversampling are training-only.",
        "calibration": "Sigmoid calibration fit only on the grouped validation partition. Test metrics use the independently held-out grouped test partition.",
        "data_quality": data_quality,
        "split_class_distribution": {"train": distribution(train_frame), "validation": distribution(validation_frame), "test": distribution(test_frame), "train_after_oversampling": {str(k): int(v) for k, v in Counter(y_train_balanced).items()}},
        "evaluation": evaluation,
    }
    joblib.dump(model, models_dir / "model.pkl")
    joblib.dump(vectorizer, models_dir / "tfidf.pkl")
    joblib.dump(LABEL_MAP, models_dir / "label.pkl")
    (models_dir / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    if models_dir.resolve() == MODELS_DIR.resolve():
        write_executive_report(metadata, ROOT / "reports" / "EXECUTIVE_REPORT.md")
    return metadata


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, default=DATASET_PATH)
    parser.add_argument("--models-dir", type=Path, default=MODELS_DIR)
    args = parser.parse_args()
    print(json.dumps(train(args.dataset, args.models_dir)["evaluation"], indent=2))
