# Base Python image
FROM python:3.12-slim

# Set working directory
WORKDIR /app

# Copy requirements and install
COPY requirements.txt .
RUN pip install --upgrade pip && pip install --no-cache-dir -r requirements.txt

# Copy app code and models
COPY streamlit_dashboard.py .
COPY models ./models

# Expose Streamlit port
EXPOSE 8080

# Streamlit environment variables
ENV STREAMLIT_SERVER_HEADLESS=true
ENV STREAMLIT_SERVER_PORT=8080
ENV STREAMLIT_SERVER_ADDRESS=0.0.0.0
ENV STREAMLIT_SERVER_ENABLE_CORS=false

# Run Streamlit
CMD ["streamlit", "run", "streamlit_dashboard.py", "--server.port=8080", "--server.address=0.0.0.0"]
