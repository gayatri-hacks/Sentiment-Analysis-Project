"""Streamlit explorer for the four-class climate-change stance model."""
from __future__ import annotations
import json
from pathlib import Path
import joblib
import pandas as pd
import plotly.express as px
import streamlit as st
from src.text_preprocessing import clean_text, ensure_nltk_resources

ROOT = Path(__file__).resolve().parent
MODELS_DIR, DATASET_PATH = ROOT / "models", ROOT / "src" / "twitter_sentiment_data.csv"
LABEL_MAP = {-1: "Anti", 0: "Neutral", 1: "Pro", 2: "News"}
COLORS = {"Anti": "#dc3545", "Neutral": "#ffc107", "Pro": "#28a745", "News": "#17a2b8"}
st.set_page_config(page_title="Climate-change stance dashboard", page_icon="📊", layout="wide")

@st.cache_resource
def load_model_artifacts():
    files = [MODELS_DIR / name for name in ("model.pkl", "tfidf.pkl", "label.pkl", "metadata.json")]
    absent = [str(path.relative_to(ROOT)) for path in files if not path.is_file()]
    if absent: raise FileNotFoundError("Missing trained artifacts: " + ", ".join(absent) + ". Run: python -m src.sentiment_analysis")
    model, vectorizer, labels = (joblib.load(files[i]) for i in range(3))
    if labels != LABEL_MAP: raise ValueError(f"Artifact label mapping is invalid: {labels!r}")
    return model, vectorizer, labels, json.loads(files[3].read_text(encoding="utf-8"))

@st.cache_data
def load_dataset() -> pd.DataFrame:
    if not DATASET_PATH.is_file(): raise FileNotFoundError(f"Dataset not found: {DATASET_PATH}")
    data = pd.read_csv(DATASET_PATH)
    if not {"message", "sentiment"}.issubset(data.columns): raise ValueError("Dataset must contain message and sentiment columns.")
    return data

def predict(text: str, model, vectorizer) -> tuple[int, str, float]:
    features = vectorizer.transform([clean_text(text)])
    raw = int(model.predict(features)[0])
    return raw, LABEL_MAP[raw], float(model.predict_proba(features)[0].max())

def labelled_data(data: pd.DataFrame) -> pd.DataFrame:
    view = data[["message", "sentiment", "tweetid"]].copy()
    view["stance"] = view.sentiment.map(LABEL_MAP)
    view["message_length"] = view.message.astype(str).str.len()
    return view

def home(data, metadata):
    st.header("Climate-change stance dashboard")
    st.caption("Classifies tweet stance: Anti, Neutral, Pro, or News — not generic positive/negative sentiment.")
    cols = st.columns(4); cols[0].metric("Original tweets", f"{len(data):,}"); cols[1].metric("Stance classes", data.sentiment.nunique()); cols[2].metric("Duplicate-text rows", metadata["data_quality"]["duplicate_message_rows"]); cols[3].metric("Held-out macro F1", f"{metadata['evaluation']['macro_f1']:.3f}")
    counts = data.sentiment.value_counts().sort_index()
    st.plotly_chart(px.pie(values=counts.values, names=[LABEL_MAP[int(x)] for x in counts.index], color=[LABEL_MAP[int(x)] for x in counts.index], color_discrete_map=COLORS), use_container_width=True)
    st.caption("Metrics come from the independently held-out test partition; they are not dashboard estimates.")
    st.dataframe(labelled_data(data).sample(min(10, len(data)), random_state=42), use_container_width=True)

def prediction_page(model, vectorizer):
    st.header("Predict tweet stance"); text = st.text_area("Tweet text", height=110)
    if st.button("Classify", type="primary"):
        if not text.strip(): st.warning("Enter tweet text first.")
        else:
            raw, label, probability = predict(text, model, vectorizer)
            st.success(f"Predicted stance: {label} (label {raw})")
            st.caption(f"Calibrated model probability for the predicted class: {probability:.1%}. It is an estimated likelihood, not a guarantee or generic sentiment score.")
            st.code(clean_text(text), language=None)

def data_page(data, metadata):
    st.header("Dataset exploration")
    view = labelled_data(data)
    choices = st.multiselect("Stance filter", list(LABEL_MAP.values()), default=list(LABEL_MAP.values()))
    length = st.slider("Maximum message length", 0, int(view.message_length.max()), int(view.message_length.max()))
    filtered = view[view.stance.isin(choices) & view.message_length.le(length)]
    st.caption(f"Showing {len(filtered):,} of {len(view):,} source rows. Filters are applied before export.")
    st.dataframe(filtered, use_container_width=True)
    st.plotly_chart(px.histogram(filtered, x="message_length", color="stance", color_discrete_map=COLORS), use_container_width=True)
    st.download_button("Download filtered source rows", filtered.to_csv(index=False), "filtered_climate_stance_rows.csv", "text/csv")
    st.subheader("Data quality summary")
    st.json(metadata["data_quality"])
    st.download_button("Download reproducibility metadata", json.dumps(metadata, indent=2), "model_verification_metadata.json", "application/json")

def batch_page(model, vectorizer):
    st.header("Batch predictions"); uploaded = st.file_uploader("CSV with a `message` column", type="csv")
    if uploaded and st.button("Classify batch"):
        batch = pd.read_csv(uploaded)
        if "message" not in batch: st.error("The uploaded CSV must have a `message` column."); return
        if batch.message.isna().any() or batch.message.astype(str).str.strip().eq("").any():
            st.error("The uploaded `message` column contains missing or empty values."); return
        matrix = vectorizer.transform(batch.message.map(clean_text)); batch["predicted_label"] = model.predict(matrix).astype(int); batch["predicted_stance"] = batch.predicted_label.map(LABEL_MAP); batch["calibrated_probability"] = model.predict_proba(matrix).max(axis=1)
        st.caption("`calibrated_probability` is an estimated probability for the selected stance, not a guarantee or positive/negative sentiment score.")
        st.dataframe(batch, use_container_width=True); st.download_button("Download results", batch.to_csv(index=False), "stance_predictions.csv", "text/csv")

def main():
    try: ensure_nltk_resources(); model, vectorizer, _, metadata = load_model_artifacts(); data = load_dataset()
    except Exception as error: st.error(f"Unable to start dashboard: {error}"); st.stop()
    page = st.sidebar.radio("Navigation", ["Home", "Predict", "Explore data", "Batch predictions"])
    if page == "Home": home(data, metadata)
    elif page == "Predict": prediction_page(model, vectorizer)
    elif page == "Explore data": data_page(data, metadata)
    else: batch_page(model, vectorizer)

if __name__ == "__main__": main()
