FROM python:3.9

WORKDIR /app
COPY init_opensearch.py /app/init-opensearch.py

# Install dependencies
RUN pip install requests

# Run the initialization script
CMD ["python", "/app/init_opensearch.py"]