import time
import requests

# OpenSearch Configuration
# OPENSEARCH_URL = "https://opensearch.nexavion.com"
OPENSEARCH_URL = "http://opensearch-node1:9200"
AUTH = ("os_admin", "strugglersaala")

# Model choices
ENGLISH_MODEL = "huggingface/sentence-transformers/msmarco-distilbert-base-tas-b"
MULTILINGUAL_MODEL = "huggingface/sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"

# Index and pipeline names
INDEX_NAME = "neural_search_index"  # Updated index name
INGEST_PIPELINE = "nlp-ingest-pipeline"  # Ingest pipeline name

# Ignore SSL warnings
import urllib3

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)


def wait_for_opensearch():
    """Wait until OpenSearch is ready, but stop after 30 attempts"""
    for i in range(30):  # Stop after 30 attempts
        try:
            response = requests.get(f"{OPENSEARCH_URL}/_cluster/health", auth=AUTH, verify=False)
            if response.status_code == 200 and response.json().get("status") in ["yellow", "green"]:
                print("OpenSearch is ready!")
                return
        except requests.exceptions.RequestException as e:
            print(f"⚠ Attempt {i + 1}/30: OpenSearch not ready yet ({e})")

        time.sleep(5)  # Wait 5 seconds between attempts

    print("OpenSearch did not become ready in time. Exiting.")
    exit(1)  # Exit script after 30 attempts


def configure_cluster_settings():
    """Set ML cluster settings"""
    payload = {
        "persistent": {
            "plugins.ml_commons.only_run_on_ml_node": "false",
            "plugins.ml_commons.model_access_control_enabled": "true",
            "plugins.ml_commons.native_memory_threshold": "99"
        }
    }
    response = requests.put(f"{OPENSEARCH_URL}/_cluster/settings", json=payload, auth=AUTH, verify=False)
    print("Cluster settings updated:", response.json())


def register_model(model_name):
    """Register an ML model and wait for it to be ready."""
    payload = {
        "name": model_name,
        "version": "1.0.1",
        "model_format": "TORCH_SCRIPT",
        "model_task_type": "TEXT_EMBEDDING"
    }
    response = requests.post(f"{OPENSEARCH_URL}/_plugins/_ml/models/_register", json=payload, auth=AUTH, verify=False)

    if response.status_code != 200:
        print(f"Failed to register model {model_name}: {response.json()}")
        exit(1)

    task_id = response.json()["task_id"]
    print(f"Model {model_name} registered, waiting for it to be ready... (Task ID: {task_id})")

    # Wait for model to be downloaded and fully registered
    model_id = wait_for_model(task_id)

    # Load the model manually
    load_model(model_id)

    return model_id  # Return the actual model ID


def is_model_available(model_id):
    """Check if the model is available in OpenSearch."""
    response = requests.get(f"{OPENSEARCH_URL}/_plugins/_ml/models", auth=AUTH, verify=False)

    if response.status_code != 200:
        print(f"Warning: Failed to fetch model list: {response.json()}")
        return False

    models = response.json().get("models", [])

    for model in models:
        if model["model_id"] == model_id and model["model_state"] == "DEPLOYED":
            return True  # Model is downloaded and ready

    return False  # Model is not yet available


def wait_for_model(task_id):
    """Wait until the model is fully registered AND available before deployment, then return model_id."""
    for i in range(30):  # Wait up to 5 minutes
        response = requests.get(f"{OPENSEARCH_URL}/_plugins/_ml/tasks/{task_id}", auth=AUTH, verify=False)
        data = response.json()

        state = data.get("state", "UNKNOWN")
        model_id = data.get("model_id")  # Extract model_id

        if state == "COMPLETED" and model_id:
            # Confirm model is actually in the available models list
            if is_model_available(model_id):
                print(f"Model {model_id} is fully registered and available!")
                return model_id  # Return model ID
            else:
                print(f"Model {model_id} registered, but still downloading...")

        elif state == "FAILED":
            print(f"Model {task_id} failed to register: {data}")
            exit(1)

        print(f"Model {task_id} is still registering... ({i + 1}/30)")
        time.sleep(10)  # Wait 10 seconds before checking again

    print(f"Model {task_id} did not become ready in time!")
    exit(1)


def load_model(model_id):
    """Load the model into OpenSearch after registration."""
    response = requests.post(f"{OPENSEARCH_URL}/_plugins/_ml/models/{model_id}/_load", auth=AUTH, verify=False)

    if response.status_code != 200:
        print(f"Failed to load model {model_id}: {response.json()}")
        exit(1)

    print(f"Model {model_id} successfully loaded into OpenSearch!")


def deploy_model(model_id):
    """Deploy the registered ML model."""
    for i in range(30):  # ✅ Wait until model is loaded
        response = requests.get(f"{OPENSEARCH_URL}/_plugins/_ml/models/{model_id}", auth=AUTH, verify=False)
        model_data = response.json()

        if model_data.get("model_state") == "DEPLOYED":
            print(f"Model {model_id} is deployed!")
            return

        print(f"⏳ Model {model_id} is still loading... ({i + 1}/30)")
        time.sleep(10)  # ✅ Wait 10 seconds before checking again

    print(f"Model {model_id} did not deploy in time!")
    exit(1)


def create_ingest_pipeline(model_id_en, model_id_multi):
    """Create an ingest pipeline for both English and Multilingual models"""
    pipeline_payload = {
        "description": "Multimodel NLP ingest pipeline",
        "processors": [
            {
                "text_embedding": {
                    "model_id": model_id_en,
                    "field_map": {
                        "text": "embedding_en"
                    }
                }
            },
            {
                "text_embedding": {
                    "model_id": model_id_multi,
                    "field_map": {
                        "text": "embedding_multi"
                    }
                }
            }
        ]
    }
    response = requests.put(f"{OPENSEARCH_URL}/_ingest/pipeline/{INGEST_PIPELINE}", json=pipeline_payload, auth=AUTH,
                            verify=False)
    print("Ingest pipeline created:", response.json())


def create_knn_index():
    """Delete and recreate an OpenSearch k-NN index."""

    # Check if index exists
    response = requests.get(f"{OPENSEARCH_URL}/{INDEX_NAME}", auth=AUTH, verify=False)

    if response.status_code == 200:
        print(f"Index '{INDEX_NAME}' already exists. Deleting it first...")
        delete_response = requests.delete(f"{OPENSEARCH_URL}/{INDEX_NAME}", auth=AUTH, verify=False)
        print(f"Index '{INDEX_NAME}' deleted:", delete_response.json())
        time.sleep(5)  # Wait briefly to ensure deletion completes

    # Create a new index
    index_payload = {
        "settings": {
            "index.knn": True
        },
        "mappings": {
            "properties": {
                "embedding_en": {
                    "type": "knn_vector",
                    "dimension": 768
                },
                "embedding_multi": {
                    "type": "knn_vector",
                    "dimension": 384
                }
            }
        }
    }

    response = requests.put(f"{OPENSEARCH_URL}/{INDEX_NAME}", json=index_payload, auth=AUTH, verify=False)
    print("New k-NN index created:", response.json())


def ingest_sample_data():
    """Index a sample document"""
    doc_payload = {
        "text": "OpenSearch is an open-source search engine."
    }
    response = requests.post(f"{OPENSEARCH_URL}/{INDEX_NAME}/_doc?pipeline={INGEST_PIPELINE}", json=doc_payload,
                             auth=AUTH, verify=False)
    print("Document indexed:", response.json())


# Run everything in order**
wait_for_opensearch()
configure_cluster_settings()

# Register models and wait for them to be ready
model_id_en = register_model(ENGLISH_MODEL)
model_id_multi = register_model(MULTILINGUAL_MODEL)

# Deploy only when models are fully ready
deploy_model(model_id_en)
deploy_model(model_id_multi)

# Create ingest pipeline and k-NN index
create_ingest_pipeline(model_id_en, model_id_multi)
create_knn_index()

# Index sample document
ingest_sample_data()

print("OpenSearch Neural Search setup completed with two models!")
