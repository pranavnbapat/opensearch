# create_neural_search_index/create_neural_search_index.py

import json
import os
import re
import sys

from datetime import datetime
from transformers import AutoTokenizer

# Add the project root to Python's path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from utils import (BASE_MODEL_CONFIG, client, maybe_lowercase, clean_text_light, clean_text_moderate,
                   clean_text_extensive, remove_extra_quotes, chunk_text_by_tokens, helpers)

dev_mode = True

# Apply `_dev` suffix to index names if dev_mode is True
MODEL_CONFIG = {
    model: {
        **cfg,
        "index": cfg["index"] + ("_dev" if dev_mode else "")
    }
    for model, cfg in BASE_MODEL_CONFIG.items()
}


def get_latest_json_file():
    folder = "../raw_data_dev" if dev_mode else "../raw_data"
    files = [f for f in os.listdir(folder) if f.endswith(".json")]
    if not files:
        raise FileNotFoundError(f"No JSON files found in {folder}!")
    latest_file = max(files, key=lambda x: os.path.getctime(os.path.join(folder, x)))
    return os.path.join(folder, latest_file)


def reset_index(INDEX_NAME, PIPELINE_NAME, VECTOR_DIM):
    try:
        client.indices.delete(index=INDEX_NAME, ignore=[400, 404])  # Delete index if it exists
        print(f"Deleted index: {INDEX_NAME}")

        # Recreate the index with required settings and mappings
        client.indices.create(
            index=INDEX_NAME,
            body={
                "settings": {"index.knn": True, "default_pipeline": PIPELINE_NAME},
                "mappings": {
                    "properties": {
                        "title_embedding": {
                            "type": "knn_vector", "dimension": VECTOR_DIM,
                            "method": {"engine": "lucene", "space_type": "l2", "name": "hnsw",
                                       "parameters": {"ef_construction": 512, "m": 16}}
                        },
                        "summary_embedding": {
                            "type": "knn_vector", "dimension": VECTOR_DIM,
                            "method": {"engine": "lucene", "space_type": "l2", "name": "hnsw",
                                       "parameters": {"ef_construction": 512, "m": 16}}
                        },
                        "content_embedding": {
                            "type": "knn_vector", "dimension": VECTOR_DIM,
                            "method": {"engine": "lucene", "space_type": "l2", "name": "hnsw",
                                       "parameters": {"ef_construction": 512, "m": 16}}
                        },
                        "keywords_embedding": {
                            "type": "knn_vector", "dimension": VECTOR_DIM,
                            "method": {"engine": "lucene", "space_type": "l2", "name": "hnsw",
                                       "parameters": {"ef_construction": 512, "m": 16}}
                        },
                        "locations_embedding": {
                            "type": "knn_vector", "dimension": VECTOR_DIM,
                            "method": {"engine": "lucene", "space_type": "l2", "name": "hnsw",
                                       "parameters": {"ef_construction": 512, "m": 16}}
                        },
                        "project_embedding": {
                            "type": "knn_vector", "dimension": VECTOR_DIM,
                            "method": {"engine": "lucene", "space_type": "l2", "name": "hnsw",
                                       "parameters": {"ef_construction": 512, "m": 16}}
                        },

                        "_orig_id": {"type": "keyword"},
                        "@id": {"type": "keyword", "index": False},
                        "title": {"type": "text"},
                        "summary": {"type": "text"},
                        "content_pages": {"type": "text"},
                        "keywords": {"type": "text", "fields": {"raw": {"type": "keyword"}}},
                        "topics": {"type": "keyword"},
                        "subtopics": {"type": "keyword"},
                        "locations": {"type": "keyword"},
                        "fileType": {"type": "keyword"},
                        "languages": {"type": "keyword"},
                        "dateCreated": {"type": "date"},
                        "creators": {"type": "text"},
                        "projectName": {"type": "text"},
                        "projectAcronym": {"type": "keyword"},
                        "project_id": {"type": "keyword"},
                        "project_type": {"type": "keyword"},
                        "projectURL": {"type": "keyword"},
                    }
                }
            }
        )
        print(f"Recreated index: {INDEX_NAME}")

        # Apply ef_search setting after recreating the index
        client.indices.put_settings(
            index=INDEX_NAME,
            body={"index": {"knn.algo_param.ef_search": 100}}
        )
        print(f"Applied ef_search setting (100) to {INDEX_NAME}")

    except Exception as e:
        print(f"Error resetting index: {e}")


def generate_bulk_actions(documents, index_name):
    """
    Generator that yields bulk index operations for OpenSearch.
    Processes data in batches.
    """
    for doc in documents:
        yield {
            "_index": index_name,
            "_id": doc["_orig_id"],  # Use _orig_id as document ID
            "_source": doc  # The actual document data
        }


# Function to process JSON data for OpenSearch
def process_json_for_opensearch(input_file, tokenizer, model_key):
    """
    Reads a JSON file, applies cleaning functions appropriately, and prepares data for OpenSearch ingestion.
    - Stores both cleaned and original versions of the data.
    - Only indexes cleaned data while keeping original data for search results.
    """

    # Load JSON data
    with open(input_file, "r", encoding="utf-8") as file:
        documents = json.load(file)

    processed_documents = []

    for doc in documents:
        cleaned_doc = {}  # Stores cleaned fields for OpenSearch indexing
        original_doc = {}  # Stores original fields for search results

        title_raw = str(doc.get("title", "")).strip()
        summary_raw = str(doc.get("summary", "")).strip()
        project_name_raw = str(doc.get("projectName", "")).strip()
        project_acronym_raw = str(doc.get("projectAcronym", "")).strip()

        cleaned_doc["title"] = clean_text_light(remove_extra_quotes(title_raw))
        cleaned_doc["summary"] = clean_text_light(remove_extra_quotes(summary_raw))
        cleaned_doc["projectName"] = clean_text_light(project_name_raw)
        cleaned_doc["projectAcronym"] = clean_text_light(project_acronym_raw)

        original_doc["title"] = title_raw
        original_doc["summary"] = summary_raw
        original_doc["projectName"] = project_name_raw
        original_doc["projectAcronym"] = project_acronym_raw

        # Embedding inputs (with lowercasing as needed)
        cleaned_doc["title_embedding_input"] = maybe_lowercase(cleaned_doc["title"], model_key)
        cleaned_doc["summary_embedding_input"] = maybe_lowercase(cleaned_doc["summary"], model_key)
        cleaned_doc["project_embedding_input"] = maybe_lowercase(
            f"{cleaned_doc['projectName']} {cleaned_doc['projectAcronym']}".strip(), model_key
        )

        # Fields that need moderate cleaning (lists remain as lists)
        for key in ["keywords", "topics", "subtopics", "project_type", "locations", "languages"]:
            value = doc.get(key)
            if isinstance(value, list):
                cleaned = [clean_text_moderate(str(v).strip()) for v in value if isinstance(v, str) and v.strip()]
                if cleaned:
                    cleaned_doc[key] = cleaned
                    original_doc[key] = cleaned
            elif isinstance(value, str) and value.strip():
                cleaned = clean_text_moderate(value)
                cleaned_doc[key] = cleaned
                original_doc[key] = value

        # Embedding-friendly inputs from lists
        if "keywords" in cleaned_doc:
            joined = " ".join(cleaned_doc["keywords"])
            cleaned_doc["keywords_embedding_input"] = maybe_lowercase(joined, model_key)
        if "locations" in cleaned_doc:
            joined = " ".join(cleaned_doc["locations"])
            cleaned_doc["locations_embedding_input"] = maybe_lowercase(joined, model_key)

        # Extract and clean content_pages from ko_content
        if "ko_content" in doc and isinstance(doc["ko_content"], list):
            all_chunks = []

            for item in doc["ko_content"]:
                pages = item.get("content", {}).get("content_pages", [])
                for page in pages:
                    if not isinstance(page, str):
                        continue

                    cleaned = clean_text_extensive(page, preserve_numbers=True)
                    cleaned = maybe_lowercase(cleaned, model_key)
                    chunk_objs = chunk_text_by_tokens(cleaned, tokenizer)

                    for chunk in chunk_objs:
                        all_chunks.append(chunk["text"])  # still send only text for embedding

                        # Store metadata per chunk
                        if "content_pages_token_counts" not in cleaned_doc:
                            cleaned_doc["content_pages_token_counts"] = []
                        cleaned_doc["content_pages_token_counts"].append(chunk["token_count"])

            if all_chunks:
                cleaned_doc["content_pages"] = all_chunks

        # Convert dateCreated to ISO 8601 format
        original_date = str(doc.get("dateCreated", "")).strip()
        # print("original_date: ", original_date)
        if original_date:
            try:
                if re.fullmatch(r"\d{4}", original_date):
                    parsed = datetime.strptime(original_date + "-01-01", "%Y-%m-%d")
                elif re.fullmatch(r"\d{4}-\d{2}-\d{2}", original_date):
                    parsed = datetime.strptime(original_date, "%Y-%m-%d")
                elif re.fullmatch(r"\d{2}-\d{2}-\d{4}", original_date):
                    parsed = datetime.strptime(original_date, "%d-%m-%Y")
                else:
                    raise ValueError("Unrecognised date format")
                    # parsed = datetime.strptime(original_date, "%d-%m-%Y")

                cleaned_doc["dateCreated"] = parsed.strftime("%Y-%m-%d")  # For OpenSearch
                original_doc["dateCreated"] = parsed.strftime("%d-%m-%Y")  # For UI

                # print("cleaned_doc[\"dateCreated\"]", cleaned_doc["dateCreated"])
                # print("original_doc[\"dateCreated\"]", original_doc["dateCreated"])
            except Exception as e:
                # Log and skip problematic date in cleaned_doc
                print(f"Skipping bad date: {original_date} → {e}")

                # Skip invalid date
                original_doc["dateCreated"] = original_date  # still show in UI
                cleaned_doc.pop("dateCreated", None)
        else:
            # Don't include empty date in OpenSearch index
            original_doc.pop("dateCreated", None)

        # Store only these fields in the original version (returned in search results)
        search_result_fields = ["title", "creators", "topics", "fileType", "keywords", "dateCreated", "_orig_id", "@id",
                                "project_id", "project_type", "projectAcronym", "project_id", "projectURL"]
        for key in search_result_fields:
            if key in doc:
                original_doc[key] = doc[key]

        # Prepare the final document for OpenSearch ingestion
        processed_doc = {
            **original_doc,  # Original data for returning in search
            **cleaned_doc  # Cleaned data for indexing
        }
        processed_documents.append(processed_doc)

    return processed_documents


for MODEL, CONFIG in MODEL_CONFIG.items():
    print(f"\nProcessing model: {MODEL} \n")

    tokenizer = AutoTokenizer.from_pretrained(CONFIG["tokenizer"])
    INDEX_NAME = CONFIG["index"]
    PIPELINE_NAME = CONFIG["pipeline"]
    VECTOR_DIM = CONFIG["dimension"]

    try:
        latest_file = get_latest_json_file()
        print(f"Using file: {latest_file}")

        processed_data = process_json_for_opensearch(latest_file, tokenizer, MODEL)
        total_docs = len(processed_data)

        reset_index(INDEX_NAME, PIPELINE_NAME, VECTOR_DIM)

        print(f"Starting ingestion of {total_docs} documents...")

        print(f"Indexing into: {INDEX_NAME}")
        batch_size = 20
        for i in range(0, total_docs, batch_size):
            batch = processed_data[i: i + batch_size]
            success_count, errors = helpers.bulk(client, generate_bulk_actions(batch, INDEX_NAME), refresh="wait_for",
                                                 stats_only=False)
            print(f"Batch {i // batch_size + 1}: {success_count} successes.")
            if errors:
                print(f"Batch {i // batch_size + 1}: {len(errors)} errors occurred.")

        print(f"Finished ingestion for: {MODEL}")
    except Exception as e:
        print(f"Error during {MODEL}: {e}")

