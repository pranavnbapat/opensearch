#FROM python:3.9-slim-buster
FROM python:3.9

WORKDIR /app
#COPY init_opensearch.py /app/init-opensearch.py

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY init_opensearch.py .

# Run the initialization script
CMD ["python", "/app/init_opensearch.py"]