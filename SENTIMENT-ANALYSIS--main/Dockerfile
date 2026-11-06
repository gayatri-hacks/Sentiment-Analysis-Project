FROM python:3.12-slim
WORKDIR /app
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 STREAMLIT_SERVER_HEADLESS=true STREAMLIT_SERVER_PORT=8080 STREAMLIT_SERVER_ADDRESS=0.0.0.0
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt && python -m nltk.downloader -d /usr/local/nltk_data stopwords
ENV NLTK_DATA=/usr/local/nltk_data
COPY src ./src
COPY models ./models
COPY streamlit_dashboard.py .
EXPOSE 8080
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8080/_stcore/health', timeout=3)" || exit 1
CMD ["streamlit", "run", "streamlit_dashboard.py", "--server.port=8080", "--server.address=0.0.0.0"]
