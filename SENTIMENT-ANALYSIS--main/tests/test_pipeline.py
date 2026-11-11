import json
from pathlib import Path

import pandas as pd

from src.sentiment_analysis import LABEL_MAP, grouped_split, load_data
from src.text_preprocessing import clean_text

ROOT = Path(__file__).resolve().parents[1]

def test_real_dataset_loads_and_has_expected_labels():
    data, quality = load_data()
    assert len(data) == 43943
    assert set(data.sentiment.unique()) == set(LABEL_MAP)
    assert quality["rows_with_missing_message_or_sentiment"] == 0

def test_documented_class_mapping():
    assert LABEL_MAP == {-1: "Anti", 0: "Neutral", 1: "Pro", 2: "News"}

def test_preprocessing_is_deterministic_and_shared():
    sample = "@User Climate-change!!! https://example.test #ActNow 2024"
    assert clean_text(sample) == clean_text(sample)
    assert "actnow" in clean_text(sample)

def test_duplicate_groups_do_not_cross_partitions():
    data, _ = load_data()
    train, validation, test = grouped_split(data)
    assert not set(train.split_group) & set(validation.split_group)
    assert not set(train.split_group) & set(test.split_group)
    assert not set(validation.split_group) & set(test.split_group)

def test_grouped_split_keeps_tiny_fixture_duplicates_together():
    fixture = pd.DataFrame({"split_group": ["same", "same", "different", "third", "fourth", "fifth"], "sentiment": [-1, -1, 0, 1, 2, 1]})
    train, validation, test = grouped_split(fixture, seed=7)
    assert {"same"}.issubset(set(train.split_group) | set(validation.split_group) | set(test.split_group))
    assert sum(frame.split_group.eq("same").sum() for frame in (train, validation, test)) == 2

def test_trained_artifacts_load_and_predict():
    import joblib
    model = joblib.load(ROOT / "models" / "model.pkl")
    vectorizer = joblib.load(ROOT / "models" / "tfidf.pkl")
    labels = joblib.load(ROOT / "models" / "label.pkl")
    prediction = int(model.predict(vectorizer.transform([clean_text("Climate change is real")]))[0])
    assert labels == LABEL_MAP
    assert prediction in LABEL_MAP

def test_batch_processing_shape():
    import joblib
    model = joblib.load(ROOT / "models" / "model.pkl")
    vectorizer = joblib.load(ROOT / "models" / "tfidf.pkl")
    messages = pd.Series(["Climate change is a hoax", "Scientists release a climate report"])
    output = model.predict(vectorizer.transform(messages.map(clean_text)))
    assert len(output) == 2 and set(output).issubset(LABEL_MAP)

def test_metadata_contains_real_evaluation():
    metadata = json.loads((ROOT / "models" / "metadata.json").read_text())
    assert {"accuracy", "macro_f1", "weighted_f1", "per_class", "confusion_matrix"} <= set(metadata["evaluation"])
    assert "Sigmoid calibration" in metadata["calibration"]

def test_executive_report_is_derived_from_real_metadata():
    report = (ROOT / "reports" / "EXECUTIVE_REPORT.md").read_text(encoding="utf-8")
    metadata = json.loads((ROOT / "models" / "metadata.json").read_text())
    assert "original rows: 43,943" in report
    assert f"macro F1: {metadata['evaluation']['macro_f1']:.3f}" in report
    assert "not evidence of production readiness" in report
