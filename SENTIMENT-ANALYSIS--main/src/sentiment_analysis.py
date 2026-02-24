import re
import string
import pandas as pd
import joblib
import nltk
import random
import numpy as np
from nltk.corpus import stopwords, wordnet
from nltk.stem import WordNetLemmatizer
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, classification_report
from imblearn.over_sampling import RandomOverSampler
import os

# -------------------------------
# NLTK setup
# -------------------------------
nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)
nltk.download('averaged_perceptron_tagger', quiet=True)

stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

# -------------------------------
# Text preprocessing
# -------------------------------
def get_wordnet_pos(word):
    tag = nltk.pos_tag([word])[0][1][0].upper()
    return {"J": wordnet.ADJ, "N": wordnet.NOUN, "V": wordnet.VERB, "R": wordnet.ADV}.get(tag, wordnet.NOUN)

def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"http\S+|www\S+", "URL", text)
    text = re.sub(r"@\w+", "USER", text)
    text = re.sub(r"#(\w+)", r"\1", text)
    text = re.sub(f"[{re.escape(string.punctuation)}]", "", text)
    text = re.sub(r"\d+", "", text)
    text = re.sub(r"(.)\1{2,}", r"\1", text)
    words = text.split()
    # Simple negation handling
    for i, w in enumerate(words):
        if w in ["not", "no", "never"] and i+1 < len(words):
            words[i+1] = "not_" + words[i+1]
            words[i] = ""
    words = [lemmatizer.lemmatize(w, get_wordnet_pos(w)) for w in words if w and w not in stop_words and len(w) > 2]
    return " ".join(words)

# -------------------------------
# Synonym augmentation
# -------------------------------
def synonym_augment(text, n_aug=1):
    words = text.split()
    augmented_texts = []
    for _ in range(n_aug):
        new_words = words.copy()
        for i, word in enumerate(words):
            syns = wordnet.synsets(word)
            lemmas = [l.name() for s in syns[:1] for l in s.lemmas()]
            lemmas = [l.replace('_','') for l in lemmas if l.lower() != word.lower()]
            if lemmas and random.random() < 0.3:
                new_words[i] = random.choice(lemmas)
        augmented_texts.append(" ".join(new_words))
    return augmented_texts

# -------------------------------
# Load dataset
# -------------------------------
df = pd.read_csv("twitter_sentiment_data.csv")
df = df[['message', 'sentiment']].dropna().reset_index(drop=True)
df['label'] = df['sentiment'].astype(int)
df['clean_text'] = df['message'].apply(clean_text)
print(f"Loaded {len(df)} samples.")

# -------------------------------
# Balance classes via augmentation
# -------------------------------
max_count = df['label'].value_counts().max()
aug_texts, aug_labels = [], []

for label in [-1,0,1,2]:
    subset = df[df['label']==label]
    n_needed = max_count - len(subset)
    if n_needed > 0:
        times = max(1, n_needed // len(subset))
        for _, row in subset.iterrows():
            aug_texts.extend(synonym_augment(row['clean_text'], n_aug=times))
            aug_labels.extend([label]*times)

df_aug = pd.DataFrame({'clean_text': aug_texts, 'label': aug_labels})
df = pd.concat([df, df_aug], ignore_index=True)

# -------------------------------
# TF-IDF vectorization (word + char ngrams)
# -------------------------------
tfidf = TfidfVectorizer(
    max_features=30000,
    ngram_range=(1,3),
    analyzer='word',
    min_df=3,
    max_df=0.9,
    sublinear_tf=True
)
X_tfidf = tfidf.fit_transform(df['clean_text'])
y = df['label']

# -------------------------------
# Optional oversampling for exact balance
# -------------------------------
ros = RandomOverSampler(random_state=42)
X_res, y_res = ros.fit_resample(X_tfidf, y)

print("Class distribution after augmentation + oversampling:")
print(pd.Series(y_res).value_counts())

# -------------------------------
# Train-test split
# -------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X_res, y_res, test_size=0.2, random_state=42, stratify=y_res
)

# -------------------------------
# Logistic Regression with class weights
# -------------------------------
clf = LogisticRegression(
    C=1.0,
    class_weight='balanced',
    max_iter=5000,
    solver='liblinear',
    random_state=42
)
clf.fit(X_train, y_train)

# -------------------------------
# Evaluation
# -------------------------------
y_pred = clf.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
print("F1-score:", f1_score(y_test, y_pred, average='weighted'))
print(classification_report(y_test, y_pred, target_names=['Anti','Neutral','Pro','News']))

# -------------------------------
# Save artifacts
# -------------------------------
models_dir = r"D:\cc final project\sentiment-analysis-project\models"
os.makedirs(models_dir, exist_ok=True)
joblib.dump(clf, os.path.join(models_dir, "model.pkl"))
joblib.dump(tfidf, os.path.join(models_dir, "tfidf.pkl"))
num_to_label = {-1:'Anti', 0:'Neutral', 1:'Pro', 2:'News'}
joblib.dump(num_to_label, os.path.join(models_dir, "label.pkl"))
print("Saved model, tfidf, and label map")
