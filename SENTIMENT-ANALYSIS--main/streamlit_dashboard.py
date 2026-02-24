import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import re
import nltk
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import plotly.express as px
import seaborn as sns
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import warnings
warnings.filterwarnings('ignore')

# Download NLTK resources
nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)
nltk.download('omw-1.4', quiet=True)

# Page config
st.set_page_config(
    page_title="Sentiment Analysis Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
.main-header {font-size: 3rem; color: #1f77b4; text-align:center; margin-bottom:2rem;}
.metric-card {background-color:#f0f2f6; padding:1rem; border-radius:0.5rem; border-left:4px solid #1f77b4;}
.sentiment-positive {color:#28a745; font-weight:bold;}
.sentiment-negative {color:#dc3545; font-weight:bold;}
.sentiment-neutral {color:#ffc107; font-weight:bold;}
.sentiment-news {color:#17a2b8; font-weight:bold;}
</style>
""", unsafe_allow_html=True)

# Label mapping
LABEL_MAP = {2: 'News', 1: 'Pro', 0: 'Neutral', -1: 'Anti'}
LABEL_COLORS = {'News': '#17a2b8', 'Pro': '#28a745', 'Neutral': '#ffc107', 'Anti': '#dc3545'}

# ----------------------- Utility Functions -----------------------

@st.cache_data
def load_model_artifacts():
    """Load model, vectorizer, and label encoder"""
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    MODELS_DIR = os.path.join(BASE_DIR, 'models')
    clf = joblib.load(os.path.join(MODELS_DIR, 'model.pkl'))
    tfidf = joblib.load(os.path.join(MODELS_DIR, 'tfidf.pkl'))
    le = joblib.load(os.path.join(MODELS_DIR, 'label.pkl'))
    return clf, tfidf, le

@st.cache_data
def load_dataset():
    """Load dataset"""
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    path = os.path.join(BASE_DIR,'src', 'twitter_sentiment_data.csv')
    if os.path.exists(path):
        return pd.read_csv(path)
    else:
        st.warning("Dataset not found")
        return pd.DataFrame(columns=['message','sentiment'])

def clean_text(text):
    """Preprocess text"""
    stop_words = set(nltk.corpus.stopwords.words('english'))
    text = str(text).lower()
    text = re.sub(r"http\S+|www\S+", "", text)
    text = re.sub(r"@\w+", "", text)
    text = re.sub(r"#", "", text)
    text = re.sub(r"[^A-Za-z\s]", "", text)
    text = re.sub(r'(.)\1{2,}', r'\1', text)
    return ' '.join([w for w in text.split() if w not in stop_words])

def predict_sentiment(text, clf, tfidf, le):
    cleaned = clean_text(text)
    X = tfidf.transform([cleaned])
    raw = int(clf.predict(X)[0])
    label =LABEL_MAP[raw]

    # Approximate confidence: max probability from decision_function
    if hasattr(clf, "decision_function"):
        scores = clf.decision_function(X)
        confidence = max(scores[0]) - min(scores[0])
    else:
        confidence = 1.0
    return raw, label, confidence

def get_sentiment_color(label):
    return LABEL_COLORS.get(label, '#6c757d')

# ----------------------- Page Functions -----------------------

def show_home_page(df):
    st.markdown("## Welcome to Sentiment Analysis Dashboard")
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Tweets", f"{len(df):,}")
    col2.metric("Sentiment Categories", df['sentiment'].nunique())
    col3.metric("Avg Tweet Length", f"{df['message'].str.len().mean():.0f} chars")
    col4.metric("Model Status", " Loaded")
    
    st.markdown("### Sentiment Distribution")
    counts = df['sentiment'].value_counts().sort_index()
    fig = px.pie(
        values=counts.values, names=[LABEL_MAP.get(x,x) for x in counts.index],
        color_discrete_map=LABEL_COLORS
    )
    fig.update_traces(textposition='inside', textinfo='percent+label')
    st.plotly_chart(fig, use_container_width=True)
    
    st.markdown("### Sample Data")
    sample = df.sample(5)[['message','sentiment']].copy()
    sample['sentiment_label'] = sample['sentiment'].map(LABEL_MAP)
    st.dataframe(sample, use_container_width=True)

def show_prediction_page(clf,tfidf,le):
    st.markdown("##  Predict Sentiment")
    text_input = st.text_area("Enter text:", height=100)
    col1, col2 = st.columns([1,3])
    
    with col1:
        if st.button("Analyze Sentiment"):
            if text_input.strip():
                raw, label, conf = predict_sentiment(text_input, clf, tfidf, le)
                color = get_sentiment_color(label)
                st.markdown(f"<div class='metric-card'><h3>Predicted: <span style='color:{color}'>{label}</span></h3><p>Raw: {raw} | Confidence: {conf:.3f}</p></div>", unsafe_allow_html=True)
                with st.expander("Preprocessed Text"): st.text(clean_text(text_input))
            else:
                st.warning("Enter text first")
    
    with col2:
        st.markdown("### Examples")
        examples = [
            "Human activities are clearly driving climate change!",
            "All this climate panic is fake news.",
            "Weather today is nice, sunny and warm",
            "Breaking: UN report warns global temperatures could rise 1.5°C by 2040."
        ]
        for text in examples:
            if st.button(f"Try: {text[:30]}...", key=text[:10]):
                st.session_state['example'] = text
                st.rerun()
        if 'example' in st.session_state: st.text_area("Selected Example", value=st.session_state['example'], height=100)

def show_data_analysis_page(df):
    st.markdown("## 📈 Data Analysis")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**Stats:**")
        st.write(f"Total samples: {len(df)}")
        st.write(f"Unique sentiments: {df['sentiment'].nunique()}")
        st.write(f"Average message length: {df['message'].str.len().mean():.1f}")
        st.write(f"Missing values: {df.isnull().sum().sum()}")
    with col2:
        st.markdown("**Sentiment Distribution:**")
        counts = df['sentiment'].value_counts()
        for s,c in counts.items(): st.write(f"- {LABEL_MAP.get(s,s)}: {c} ({c/len(df)*100:.1f}%)")
    
    st.markdown("### Message Length Analysis")
    df['message_length'] = df['message'].str.len()
    fig = px.histogram(df, x='message_length', color='sentiment', labels={'message_length':'Length'}, color_discrete_map=LABEL_COLORS)
    st.plotly_chart(fig, use_container_width=True)
    
    st.markdown("### Word Cloud")
    sent = st.selectbox("Select sentiment", options=sorted(df['sentiment'].unique()), format_func=lambda x: LABEL_MAP.get(x,x))
    filtered = df[df['sentiment']==sent]
    if len(filtered)>0:
        wc_text = ' '.join(filtered['message'].astype(str))
        wc = WordCloud(width=800, height=400, background_color='white').generate(wc_text)
        fig_wc, ax = plt.subplots(figsize=(10,5))
        ax.imshow(wc, interpolation='bilinear'); ax.axis('off')
        st.pyplot(fig_wc)

# ----------------------- Main App -----------------------

def main():
    st.markdown('<h1 class="main-header">📊 Sentiment Analysis Dashboard</h1>', unsafe_allow_html=True)
    with st.spinner("Loading..."):
        clf, tfidf, le = load_model_artifacts()
        # clf is now a fitted OneVsRestClassifier, so clf.predict(...) works
        df = load_dataset()
    if df.empty or clf is None: st.error("Failed to load data/model"); return
    
    page = st.sidebar.selectbox("Navigation", ["🏠 Home","🔮 Predict Sentiment","📈 Data Analysis"])
    if page=="🏠 Home": show_home_page(df)
    elif page=="🔮 Predict Sentiment": show_prediction_page(clf,tfidf,le)
    elif page=="📈 Data Analysis": show_data_analysis_page(df)

if __name__=="__main__":
    main()
