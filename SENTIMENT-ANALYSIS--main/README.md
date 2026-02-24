# Sentiment Analysis Project

This project implements a comprehensive sentiment analysis system using Python with multiple interfaces including a Streamlit dashboard, FastAPI server, and training scripts. It includes functionalities for data preprocessing, synonym-based augmentation, label encoding, model training, evaluation, and interactive visualization.

## Project Structure

```
sentiment-analysis-project
├── src/
│   ├── sentiment_analysis.py    # Model training script
│   └── __init__.py
├── models/                      # Trained model artifacts
│   ├── sentiment_model_svc.pkl
│   ├── tfidf_vectorizer.pkl
│   └── label_encoder.pkl
├── streamlit_dashboard.py       # Interactive Streamlit dashboard
├── requirements.txt             # Streamlit dependencies         
├── twitter_sentiment_data.csv   # Dataset
└── README.md
```

## Installation

### For Streamlit Dashboard
To set up the Streamlit dashboard, install the required dependencies:

```bash
pip install -r requirements_streamlit.txt
```

### For FastAPI Server
To set up the FastAPI server, install the basic dependencies:

```bash
pip install pandas scikit-learn nltk joblib fastapi uvicorn boto3
```

## Usage

### 1. Streamlit Dashboard (Recommended)
Launch the interactive dashboard:

```bash
streamlit run streamlit_dashboard.py
```

The dashboard provides:
- **Home Page**: Overview of the dataset and sentiment distribution
- **Predict Sentiment**: Interactive text analysis with real-time predictions
- **Data Analysis**: Dataset exploration with visualizations and word clouds
- **Model Performance**: Confusion matrix, classification report, and accuracy metrics
- **Batch Processing**: Upload CSV files or manually input multiple texts for analysis

### 2. FastAPI Server
Start the API server:

```bash
cd src
python app.py
```

Then access the API at `http://localhost:8000`

### 3. Model Training
To train the model from scratch:

```bash
python src/sentiment_analysis.py
```



## Streamlit Dashboard Features

### 🏠 Home Page
- Dataset overview with key metrics
- Sentiment distribution pie chart
- Sample data preview

### 🔮 Predict Sentiment
- Interactive text input for real-time sentiment analysis
- Confidence scores and prediction details
- Example texts to try
- Preprocessed text visualization

### 📈 Data Analysis
- Dataset statistics and overview
- Message length analysis by sentiment
- Interactive word clouds for each sentiment category
- Sentiment distribution analysis

### 📊 Model Performance
- Model accuracy metrics
- Confusion matrix visualization
- Detailed classification report
- Performance statistics

### 📝 Batch Processing
- CSV file upload for bulk analysis
- Manual batch input (one text per line)
- Results export to CSV
- Summary statistics and visualizations

## Dependencies

### Streamlit Dashboard
- streamlit>=1.28.0
- pandas>=1.5.0
- numpy>=1.24.0
- scikit-learn>=1.3.0
- nltk>=3.8.0
- joblib>=1.3.0
- plotly>=5.15.0
- seaborn>=0.12.0
- matplotlib>=3.7.0
- wordcloud>=1.9.0

### Core Dependencies
- pandas
- scikit-learn
- nltk
- joblib

## Contributing

Feel free to contribute to this project by submitting issues or pull requests.

## Label definitions

This project uses a numeric label mapping for tweet stances on climate change. The labels are:

- 2 (News): the tweet links to factual news about climate change.
- 1 (Pro): the tweet supports the belief of man-made climate change.
- 0 (Neutral): the tweet neither supports nor refutes the belief of man-made climate change.
- -1 (Anti): the tweet denies or does not believe in man-made climate change.

When preparing data or interpreting predictions, use these definitions to map numeric labels to human-readable classes.

## Example: mapping model output to labels

If the model returns numeric labels, use the following mapping in your client code (or call the `/label_map` endpoint):

```python
# Example mapping
LABEL_MAP = {2: 'News', 1: 'Pro', 0: 'Neutral', -1: 'Anti'}

# If you get a raw prediction (e.g. 1), convert it to text:
raw = 1
text_label = LABEL_MAP.get(raw, str(raw))
print(text_label)  # -> 'Pro'
```

Example HTTP request to the running FastAPI server:

```powershell
Invoke-RestMethod -Uri 'http://127.0.0.1:8000/predict' -Method Post -ContentType 'application/json' -Body (ConvertTo-Json @{ text = "I support action on climate change" })
```

The server returns JSON with `sentiment` (mapped label) and `raw_label` (original model label).
