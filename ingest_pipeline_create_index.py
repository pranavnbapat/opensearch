import os
import requests
import urllib3
from dotenv import load_dotenv

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# Load environment variables
load_dotenv()

# OpenSearch Configuration
OPENSEARCH_URL = "https://opensearch.nexavion.com"
AUTH = (os.getenv("OPENSEARCH_USR"), os.getenv("OPENSEARCH_PWD"))
HEADERS = {"Content-Type": "application/json"}

# Model IDs (Replace with your actual model IDs)
MODELS = {
    "mpnet": os.getenv("MPNET_MODEL_ID"),  # 768-dimension
    "minilm": os.getenv("MINILM_MODEL_ID")  # 384-dimension
}

INDEX_NAME = os.getenv("INDEX_NAME")
PIPELINE_NAME = os.getenv("PIPELINE_NAME")

# Fields for text embedding
TEXT_FIELDS = ["title", "summary", "projectName", "content.content_pages"]


# **Create Ingest Pipeline for Multiple Models**
def create_ingest_pipeline():
    processors = []

    for model_key, model_id in MODELS.items():
        field_map = {field: f"{field}_embedding_{model_key}" for field in TEXT_FIELDS}
        processors.append({
            "text_embedding": {
                "model_id": model_id,
                "field_map": field_map
            }
        })

    payload = {
        "description": "An NLP ingest pipeline for multiple models",
        "processors": processors
    }

    response = requests.put(
        f"{OPENSEARCH_URL}/_ingest/pipeline/{PIPELINE_NAME}",
        auth=AUTH,
        headers=HEADERS,
        json=payload,
        verify=False
    )

    if response.status_code in [200, 201]:
        print(f"Ingest pipeline '{PIPELINE_NAME}' created successfully.")
    else:
        print(f"Failed to create ingest pipeline: {response.json()}")


# **Check if k-NN Index Exists**
def index_exists(index_name):
    response = requests.head(
        f"{OPENSEARCH_URL}/{index_name}",
        auth=AUTH,
        headers=HEADERS,
        verify=False
    )
    return response.status_code == 200


# **Delete Existing Index**
def delete_index(index_name):
    if index_exists(index_name):
        response = requests.delete(
            f"{OPENSEARCH_URL}/{index_name}",
            auth=AUTH,
            headers=HEADERS,
            verify=False
        )

        if response.status_code in [200, 202]:
            print(f"Deleted existing index '{index_name}' successfully.")
        else:
            print(f"Failed to delete index '{index_name}': {response.json()}")


# **Create k-NN Index for Neural Search**
def create_knn_index():
    delete_index(INDEX_NAME)

    mappings = {
        "properties": {
            "title": {"type": "text"},
            "summary": {"type": "text"},
            "projectName": {"type": "text"},
            "projectAcronym": {"type": "keyword"},
            "keywords": {
                "type": "text",
                "fields": {
                    "raw": {"type": "keyword"}
                }
            },
            "locations": {"type": "text"},
            "content.content_pages": {"type": "text"}
        }
    }

    for model_key in MODELS.keys():
        for field in TEXT_FIELDS:
            mappings["properties"][f"{field}_embedding_{model_key}"] = {
                "type": "knn_vector",
                "dimension": 768 if model_key == "mpnet" else 384,
                "method": {
                    "engine": "lucene",
                    "space_type": "l2",
                    "name": "hnsw",
                    "parameters": {}
                }
            }

    payload = {
        "settings": {
            "index.knn": True,
            "default_pipeline": PIPELINE_NAME
        },
        "mappings": mappings
    }

    response = requests.put(
        f"{OPENSEARCH_URL}/{INDEX_NAME}",
        auth=AUTH,
        headers=HEADERS,
        json=payload,
        verify=False
    )

    if response.status_code in [200, 201]:
        print(f"k-NN index '{INDEX_NAME}' created successfully.")
    else:
        print(f"Failed to create k-NN index: {response.json()}")


# **Main Execution**
if __name__ == "__main__":
    print("Setting up OpenSearch for Neural Search...")

    # Step 1: Create Ingest Pipeline
    create_ingest_pipeline()

    # Step 2: Create k-NN Index
    create_knn_index()
