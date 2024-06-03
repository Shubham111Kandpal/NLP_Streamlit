# # Use the official Python image.

FROM python:3.10.12-slim

WORKDIR /app

COPY requirements.txt .

# Install build tools and dependencies
RUN apt-get update && \
    apt-get install -y build-essential python3-dev && \
    pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

COPY . .

ENV PYTHONPATH=/app

CMD ["streamlit", "run", "app.py", "--server.port", "8000", "--server.headless", "true"]