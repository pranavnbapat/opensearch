import os
import requests
import time
import urllib3
from dotenv import load_dotenv

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# Load environment variables
load_dotenv()

# OpenSearch Configuration
OPENSEARCH_URL = "https://opensearch.nexavion.com"
AUTH = (os.getenv("OPENSEARCH_USR"), os.getenv("OPENSEARCH_PWD"))
HEADERS = {"Content-Type": "application/json"}

MODEL_GROUP_NAME = "euf_search_models"


# 1️⃣ **Check if Model Group Exists & Create If Not**
def get_or_create_model_group(group_name):
    """Retrieve model group ID if exists, else create one."""
    payload = {
        "query": {
            "match": {
                "name": group_name
            }
        }
    }

    response = requests.post(f"{OPENSEARCH_URL}/_plugins/_ml/model_groups/_search",
                             auth=AUTH, headers=HEADERS, json=payload, verify=False)

    if response.status_code == 200:
        model_groups = response.json().get("hits", {}).get("hits", [])
        if model_groups:
            model_group_id = model_groups[0]["_id"]
            print(f"✅ Model Group '{group_name}' exists. ID: {model_group_id}")
            return model_group_id

    # If model group does not exist, create it
    payload = {
        "name": group_name,
        "description": "A model group for EUF search functionality"
    }

    response = requests.post(f"{OPENSEARCH_URL}/_plugins/_ml/model_groups/_register",
                             auth=AUTH, headers=HEADERS, json=payload, verify=False)

    if response.status_code == 200:
        model_group_id = response.json().get("model_group_id")
        print(f"✅ Model Group '{group_name}' created. ID: {model_group_id}")
        return model_group_id
    else:
        print(f"❌ Failed to create model group: {response.json()}")
        return None


# 2️⃣ **Check if Model Exists**
def get_existing_model_id(model_name):
    """Check if the model is already registered."""
    payload = {"query": {"match": {"name": model_name}}}

    response = requests.post(f"{OPENSEARCH_URL}/_plugins/_ml/models/_search",
                             auth=AUTH, headers=HEADERS, json=payload, verify=False)

    if response.status_code == 200:
        models = response.json().get("hits", {}).get("hits", [])
        if models:
            existing_model_id = models[0]["_id"]
            print(f"✅ Model '{model_name}' already exists. Model ID: {existing_model_id}")
            return existing_model_id
    return None


# 3️⃣ **Register a Model If It Doesn't Exist**
def register_model_if_not_exists(model_group_id, model_name, model_version, model_format):
    """Register a model only if it does not already exist."""
    existing_model_id = get_existing_model_id(model_name)
    if existing_model_id:
        return existing_model_id  # Return existing model ID

    payload = {
        "name": model_name,
        "version": model_version,
        "model_group_id": model_group_id,
        "model_format": model_format
    }

    response = requests.post(f"{OPENSEARCH_URL}/_plugins/_ml/models/_register",
                             auth=AUTH, headers=HEADERS, json=payload, verify=False)

    if response.status_code == 200:
        task_id = response.json().get("task_id")
        print(f"✅ Model '{model_name}' registration initiated. Task ID: {task_id}")
        return task_id
    else:
        print(f"❌ Model '{model_name}' registration failed: {response.json()}")
        return None


# 4️⃣ **Monitor Task Until Completion & Retrieve Model ID**
def get_completed_model_id(task_id, timeout=600, interval=10):
    """Wait until the model registration task is completed and return the model ID."""
    start_time = time.time()

    while time.time() - start_time < timeout:
        response = requests.get(f"{OPENSEARCH_URL}/_plugins/_ml/tasks/{task_id}",
                                auth=AUTH, headers=HEADERS, verify=False)

        if response.status_code == 200:
            task_info = response.json()
            state = task_info.get("state")

            print(f"🔄 Task {task_id} Status: {state}")

            if state == "COMPLETED":
                model_id = task_info.get("model_id")
                print(f"✅ Model registration completed. Model ID: {model_id}")
                return model_id
            elif state in ["FAILED", "ERROR"]:
                print(f"❌ Model registration failed: {task_info}")
                return None

        time.sleep(interval)

    print("❌ Timeout reached. Model registration did not complete.")
    return None


# 🚀 **Main Workflow**
if __name__ == "__main__":
    print("🚀 Checking or Creating Model Group...\n")
    model_group_id = get_or_create_model_group(MODEL_GROUP_NAME)

    if not model_group_id:
        print("\n❌ Could not retrieve or create model group. Exiting...")
        exit(1)

    # **List of models to register**
    models_to_register = [
        {"name": "huggingface/sentence-transformers/all-mpnet-base-v2", "version": "1.0.1", "format": "TORCH_SCRIPT"},
        {"name": "huggingface/sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2", "version": "1.0.1",
         "format": "TORCH_SCRIPT"}
    ]

    # **Register multiple models dynamically**
    for model in models_to_register:
        task_id = register_model_if_not_exists(model_group_id, model["name"], model["version"], model["format"])
        if task_id:
            get_completed_model_id(task_id)  # Monitor task status and retrieve Model ID
