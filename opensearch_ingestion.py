# opensearch_ingestion

import json
from datetime import datetime
from utils import *
from transformers import AutoTokenizer

# tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/msmarco-distilbert-base-tas-b")

# INDEX_NAME = os.getenv("INDEX_NAME", "neural_search_index")

MODEL_CONFIG = {
    "minilml12v2": {
        "tokenizer": "sentence-transformers/all-MiniLM-L12-v2",
        "dimension": 384,
        "pipeline": "neural_search_pipeline_minilml12v2",
        "index": "neural_search_index_minilml12v2"
    },
    # "mpnetv2": {
    #     "tokenizer": "sentence-transformers/all-mpnet-base-v2",
    #     "dimension": 768,
    #     "pipeline": "neural_search_pipeline_mpnetv2",
    #     "index": "neural_search_index_mpnetv2"
    # },
    # "msmarco": {
    #     "tokenizer": "sentence-transformers/msmarco-distilbert-base-tas-b",
    #     "dimension": 768,
    #     "pipeline": "neural_search_pipeline",
    #     "index": "neural_search_index_msmarco"
    # }
}


# Function to get the latest file from the 'raw_data' folder
def get_latest_json_file(folder="raw_data"):
    files = [f for f in os.listdir(folder) if f.endswith(".json")]
    if not files:
        raise FileNotFoundError("No JSON files found in the raw_data folder!")
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

# Function to process JSON data for OpenSearch
def process_json_for_opensearch(input_file):
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

        # Fields that need light cleaning
        for key in ["title", "summary", "projectName", "projectAcronym"]:
            if key in doc:
                value = remove_extra_quotes(str(doc[key]))  # Convert to string and remove extra quotes
                value = value.strip()  # Final trim for safety
                if value:
                    cleaned_doc[key] = clean_text_light(value)  # Cleaned version
                    original_doc[key] = value  # Keep cleaned original

        # Fields that need moderate cleaning (lists remain as lists)
        for key in ["keywords", "topics", "subtopics", "project_type", "locations", "languages"]:
            if key in doc:
                if isinstance(doc[key], list):
                    # Step 1: Remove empty values and whitespace BEFORE cleaning
                    pre_cleaned_values = [str(item).strip() for item in doc[key] if
                                          isinstance(item, str) and item.strip()]

                    # Step 2: Apply cleaning to each element
                    cleaned_values = [clean_text_moderate(item) for item in pre_cleaned_values]

                    # Step 3: Remove empty values AGAIN after cleaning
                    cleaned_values = [item for item in cleaned_values if item.strip()]

                    if cleaned_values:  # Only store if there's at least one valid value
                        cleaned_doc[key] = cleaned_values
                        original_doc[key] = cleaned_values
                    elif key in cleaned_doc:  # Remove empty lists if they exist
                        del cleaned_doc[key]
                else:
                    value = str(doc[key]).strip()  # Ensure it's a string and remove spaces
                    if value:  # Only store if not empty
                        cleaned_doc[key] = clean_text_moderate(value)
                        original_doc[key] = value

        # Extract and clean content_pages from ko_content
        if "ko_content" in doc and isinstance(doc["ko_content"], list):
            all_chunks = []

            for item in doc["ko_content"]:
                pages = item.get("content", {}).get("content_pages", [])
                for page in pages:
                    if not isinstance(page, str):
                        continue
                    cleaned = clean_text_extensive(page)

                    chunk_objs = chunk_text_by_tokens(cleaned, tokenizer)
                    for chunk in chunk_objs:
                        all_chunks.append(chunk["text"])  # still send only text for embedding

                        # Store metadata per chunk
                        if "content_pages_token_counts" not in cleaned_doc:
                            cleaned_doc["content_pages_token_counts"] = []
                        cleaned_doc["content_pages_token_counts"].append(chunk["token_count"])

            if all_chunks:
                cleaned_doc["content_pages"] = all_chunks

        # Store only these fields in the original version (returned in search results)
        search_result_fields = ["title", "creators", "topics", "fileType", "keywords", "dateCreated", "_orig_id", "@id",
                                "project_id", "project_type", "projectAcronym", "project_id", "projectURL"]
        for key in search_result_fields:
            if key in doc:
                original_doc[key] = doc[key]

        # Convert dateCreated to ISO 8601 format
        if "dateCreated" in doc:
            original_date = str(doc["dateCreated"]).strip()
            try:
                if "-" in original_date and len(original_date.split("-")[0]) == 4:
                    # Already in YYYY-MM-DD format, no need to convert
                    cleaned_doc["dateCreated"] = original_date
                else:
                    # Convert from DD-MM-YYYY to YYYY-MM-DD
                    parsed_date = datetime.strptime(original_date, "%d-%m-%Y").strftime("%Y-%m-%d")
                    cleaned_doc["dateCreated"] = parsed_date
            except ValueError:
                cleaned_doc["dateCreated"] = original_date  # Use original if parsing fails

            original_doc["dateCreated"] = original_date  # Keep the original format

        # Prepare the final document for OpenSearch ingestion
        processed_doc = {
            **cleaned_doc,  # Cleaned data for indexing
            **original_doc  # Original data for returning in search
        }
        processed_documents.append(processed_doc)

    return processed_documents


# Function to generate bulk actions
def generate_bulk_actions(documents):
    """
    Generator that yields bulk index operations for OpenSearch.
    Processes data in batches.
    """
    for doc in documents:
        # Convert `dateCreated` if it exists
        if "dateCreated" in doc and isinstance(doc["dateCreated"], str):
            doc["dateCreated"] = fix_date_format(doc["dateCreated"])
            if doc["dateCreated"] is None:
                del doc["dateCreated"]  # Remove field if conversion fails

        yield {
            "_index": INDEX_NAME,
            "_id": doc["_orig_id"],  # Use _orig_id as document ID
            "_source": doc  # The actual document data
        }


# Function to reformat date
def fix_date_format(date_str):
    """
    Converts dates to the format YYYY-MM-DD.
    - If the input is "DD-MM-YYYY", it converts it to "YYYY-MM-DD".
    - If the input is only a year (e.g., "2023"), it assumes "01-01-YYYY".
    - If invalid, returns None.
    """
    date_str = date_str.strip()  # Ensure no leading/trailing spaces

    # Check if the date is only a year (e.g., "2023")
    if re.fullmatch(r"\d{4}", date_str):  # Matches exactly 4 digits
        return f"{date_str}-01-01"  # Convert "2023" → "2023-01-01"

    # Convert from DD-MM-YYYY to YYYY-MM-DD
    try:
        return datetime.strptime(date_str, "%d-%m-%Y").strftime("%Y-%m-%d")
    except ValueError:
        print(f"Warning: Invalid date format detected: {date_str}. Skipping conversion.")
        return None  # Return None for invalid dates



for MODEL, CONFIG in MODEL_CONFIG.items():
    print(f"\nProcessing model: {MODEL} \n")

    tokenizer = AutoTokenizer.from_pretrained(CONFIG["tokenizer"])
    INDEX_NAME = CONFIG["index"]
    PIPELINE_NAME = CONFIG["pipeline"]
    VECTOR_DIM = CONFIG["dimension"]

    try:
        latest_file = get_latest_json_file()
        print(f"Using file: {latest_file}")

        processed_data = process_json_for_opensearch(latest_file)
        total_docs = len(processed_data)

        reset_index(INDEX_NAME, PIPELINE_NAME, VECTOR_DIM)

        print(f"Starting ingestion of {total_docs} documents...")

        print(f"Indexing into: {INDEX_NAME}")
        batch_size = 10
        for i in range(0, total_docs, batch_size):
            batch = processed_data[i: i + batch_size]
            success_count, errors = helpers.bulk(client, generate_bulk_actions(batch), refresh="wait_for",
                                                 stats_only=False)
            print(f"Batch {i // batch_size + 1}: {success_count} successes.")
            if errors:
                print(f"Batch {i // batch_size + 1}: {len(errors)} errors occurred.")

        print(f"Finished ingestion for: {MODEL}")
    except Exception as e:
        print(f"Error during {MODEL}: {e}")
