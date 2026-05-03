# Mock + sklearn training step for reproducible pipeline demos
FROM python:3.12-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY src/pipeline.py ./src/pipeline.py

CMD ["python", "src/pipeline.py"]
