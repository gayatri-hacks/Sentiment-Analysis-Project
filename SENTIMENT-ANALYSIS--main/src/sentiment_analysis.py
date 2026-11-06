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
    model = LogisticRegression(max_iter=2000, solver="lbfgs", random_state=SEED)
    model.fit(x_train_balanced, y_train_balanced)
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
        "data_quality": data_quality,
        "split_class_distribution": {"train": distribution(train_frame), "validation": distribution(validation_frame), "test": distribution(test_frame), "train_after_oversampling": {str(k): int(v) for k, v in Counter(y_train_balanced).items()}},
        "evaluation": evaluation,
    }
    joblib.dump(model, models_dir / "model.pkl")
    joblib.dump(vectorizer, models_dir / "tfidf.pkl")
    joblib.dump(LABEL_MAP, models_dir / "label.pkl")
    (models_dir / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return metadata


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, default=DATASET_PATH)
    parser.add_argument("--models-dir", type=Path, default=MODELS_DIR)
    args = parser.parse_args()
    print(json.dumps(train(args.dataset, args.models_dir)["evaluation"], indent=2))
