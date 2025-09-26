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

dev_mode = False

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
                        "subtitle_embedding": {
                            "type": "knn_vector", "dimension": VECTOR_DIM,
                            "method": {"engine": "lucene", "space_type": "l2", "name": "hnsw",
                                       "parameters": {"ef_construction": 512, "m": 16}}
                        },
                        "description_embedding": {
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
                        "subtitle": {"type": "text"},
                        "description": {"type": "text"},

                        "keywords": {"type": "text", "fields": {"raw": {"type": "keyword"}}},
                        "topics": {"type": "keyword"},
                        "themes": {"type": "keyword"},
                        "locations": {"type": "keyword"},
                        "category": {"type": "keyword"},
                        "subcategory": {"type": "keyword"},
                        "languages": {"type": "keyword"},
                        "intended_purposes": {"type": "keyword"},
                        "date_of_completion": {"type": "date"},
                        "creators": {"type": "text"},

                        "projectName": {"type": "text"},
                        "projectAcronym": {"type": "keyword"},
                        "project_id": {"type": "keyword"},
                        "project_type": {"type": "keyword"},
                        "projectURL": {"type": "keyword"},

                        "parent_id": {"type": "keyword"},
                        "chunk_index": {"type": "integer"},
                        "content_chunk": {"type": "text"},
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


def _make_meta_doc(doc, cleaned_doc):
    return {
        "_orig_id": f"{doc.get('_orig_id')}::meta",
        "parent_id": doc.get("_orig_id"),
        "chunk_index": -1,
        "content_chunk": "",

        "@id": doc.get("@id"),
        "title": cleaned_doc.get("title"),
        "subtitle": cleaned_doc.get("subtitle"),
        "description": cleaned_doc.get("description"),
        "keywords": cleaned_doc.get("keywords"),
        "topics": cleaned_doc.get("topics"),
        "themes": cleaned_doc.get("themes"),
        "locations": cleaned_doc.get("locations"),
        "languages": cleaned_doc.get("languages"),
        "category": doc.get("category"),
        "subcategory": doc.get("subcategory"),
        "date_of_completion": cleaned_doc.get("date_of_completion"),
        "creators": doc.get("creators"),
        "intended_purposes": doc.get("intended_purposes"),
        "projectName": doc.get("projectName"),
        "projectAcronym": doc.get("projectAcronym"),
        "project_id": doc.get("project_id"),
        "project_type": doc.get("project_type"),
        "projectURL": doc.get("projectURL"),
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
        if not doc.get("_orig_id"):
            print("[WARN] Missing _orig_id; skipping doc")
            continue

        cleaned_doc = {}  # Stores cleaned fields for OpenSearch indexing
        original_doc = {}  # Stores original fields for search results

        title_raw = str(doc.get("title", "")).strip()
        subtitle_raw = str(doc.get("subtitle", "")).strip()
        description_raw = str(doc.get("description", "")).strip()
        project_name_raw = str(doc.get("projectName", "")).strip()
        project_acronym_raw = str(doc.get("projectAcronym", "")).strip()

        if not project_name_raw and project_acronym_raw:
            project_name_raw = project_acronym_raw
        elif not project_acronym_raw and project_name_raw:
            project_acronym_raw = project_name_raw
        elif not project_name_raw and not project_acronym_raw:
            # Skip this document completely
            continue

        cleaned_doc["title"] = clean_text_light(remove_extra_quotes(title_raw))
        cleaned_doc["subtitle"] = clean_text_light(remove_extra_quotes(subtitle_raw))
        cleaned_doc["description"] = clean_text_light(remove_extra_quotes(description_raw))

        original_doc["title"] = title_raw
        original_doc["description"] = description_raw

        original_doc["projectName"] = project_name_raw
        original_doc["projectAcronym"] = project_acronym_raw

        # Embedding inputs (with lowercasing as needed)
        cleaned_doc["title_embedding_input"] = maybe_lowercase(cleaned_doc["title"], model_key)
        cleaned_doc["subtitle_embedding_input"] = maybe_lowercase(cleaned_doc["subtitle"], model_key)
        cleaned_doc["description_embedding_input"] = maybe_lowercase(cleaned_doc["description"], model_key)

        proj_for_embed = " ".join(
            s for s in (project_name_raw, project_acronym_raw) if s
        ).strip()

        if proj_for_embed:
            cleaned_doc["project_embedding_input"] = maybe_lowercase(proj_for_embed, model_key)

        # Fields that need moderate cleaning (lists remain as lists)
        for key in ["keywords", "topics", "themes", "project_type", "languages"]:
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

        locations_raw = doc.get("locations", [])
        loc_names = []
        if isinstance(locations_raw, list):
            for item in locations_raw:
                if isinstance(item, dict) and item.get("name"):
                    loc_names.append(item["name"])
                elif isinstance(item, str):
                    loc_names.append(item)

        seen = set()
        loc_names_dedup = []
        for n in loc_names:
            k = n.lower().strip()
            if k not in seen:
                seen.add(k)
                loc_names_dedup.append(n.strip())

        loc_clean = [clean_text_moderate(s) for s in loc_names_dedup if s]

        if loc_clean:
            cleaned_doc["locations"] = loc_clean
            original_doc["locations"] = loc_names_dedup

        # Embedding-friendly inputs from lists
        if "keywords" in cleaned_doc:
            joined = " ".join(cleaned_doc["keywords"])
            cleaned_doc["keywords_embedding_input"] = maybe_lowercase(joined, model_key)

        doc_dcomp_raw = str(doc.get("date_of_completion", "")).strip()
        cleaned_doc["date_of_completion"] = None
        if doc_dcomp_raw:
            try:
                # Accept common inputs and normalise to YYYY-MM-DD
                if re.fullmatch(r"\d{4}-\d{2}-\d{2}", doc_dcomp_raw):
                    dt = datetime.strptime(doc_dcomp_raw, "%Y-%m-%d")
                elif re.fullmatch(r"\d{4}", doc_dcomp_raw):
                    dt = datetime.strptime(doc_dcomp_raw + "-01-01", "%Y-%m-%d")
                elif re.fullmatch(r"\d{2}-\d{2}-\d{4}", doc_dcomp_raw):
                    dt = datetime.strptime(doc_dcomp_raw, "%d-%m-%Y")
                else:
                    raise ValueError("Unrecognised date_of_completion format")

                cleaned_doc["date_of_completion"] = dt.strftime("%Y-%m-%d")
            except Exception as e:
                print(f"[WARN] Bad date_of_completion '{doc_dcomp_raw}' for _orig_id={doc.get('_orig_id')} → {e}")



        flat_pages = doc.get("ko_content_flat", [])

        # Coerce a stray string to a one-item list (defensive but non-destructive)
        if isinstance(flat_pages, str):
            if flat_pages.strip().lower() in {"no content present", "none", "null", ""}:
                print(
                    f"[WARN] 'ko_content_flat' says '{flat_pages}' for _orig_id={doc.get('_orig_id')}; treating as empty")
                processed_documents.append(_make_meta_doc(doc, cleaned_doc))
                continue
            else:
                print(f"[WARN] 'ko_content_flat' is a string for _orig_id={doc.get('_orig_id')}; coercing to list")
                flat_pages = [flat_pages]

        # If it's not a list now, treat as no-content and emit meta
        if not isinstance(flat_pages, list):
            print(
                f"[WARN] 'ko_content_flat' is not a list for _orig_id={doc.get('_orig_id')}, type={type(flat_pages)}; indexing meta only")
            processed_documents.append(_make_meta_doc(doc, cleaned_doc))
            continue

        # Empty list ⇒ index meta-only
        if not flat_pages:
            print(f"[WARN] 'ko_content_flat' empty for _orig_id={doc.get('_orig_id')}; indexing meta only")
            processed_documents.append(_make_meta_doc(doc, cleaned_doc))
            continue

        # Non-empty list ⇒ process pages → chunks
        all_chunks = []
        for page in flat_pages:
            if not isinstance(page, str):
                continue
            cleaned = clean_text_extensive(page, preserve_numbers=True)
            cleaned = maybe_lowercase(cleaned, model_key)

            # Token-aware chunking; each item: {"text": ..., "token_count": ...}
            for ch in chunk_text_by_tokens(cleaned, tokenizer):
                # Belt-and-braces: never let a chunk exceed 512 tokens
                if ch["token_count"] > 512:
                    print(
                        f"[WARN] Overlong chunk (>512) for _orig_id={doc.get('_orig_id')}; truncation recommended in chunker")
                all_chunks.append(ch["text"])

        if all_chunks:
            for i, ch_text in enumerate(all_chunks):
                processed_documents.append({
                    "_orig_id": f"{doc.get('_orig_id')}::c{i}",
                    "parent_id": doc.get("_orig_id"),
                    "chunk_index": i,

                    "content_chunk": ch_text,
                    "content_embedding_input": ch_text,

                    "@id": doc.get("@id"),
                    "title": cleaned_doc.get("title"),
                    "subtitle": cleaned_doc.get("subtitle"),
                    "description": cleaned_doc.get("description"),
                    "keywords": cleaned_doc.get("keywords"),
                    "topics": cleaned_doc.get("topics"),
                    "themes": cleaned_doc.get("themes"),
                    "locations": cleaned_doc.get("locations"),
                    "languages": cleaned_doc.get("languages"),
                    "category": doc.get("category"),
                    "subcategory": doc.get("subcategory"),
                    "date_of_completion": cleaned_doc.get("date_of_completion"),
                    "creators": doc.get("creators"),
                    "intended_purposes": doc.get("intended_purposes"),
                    "projectName": original_doc.get("projectName"),
                    "projectAcronym": original_doc.get("projectAcronym"),
                    "project_id": doc.get("project_id"),
                    "project_type": doc.get("project_type"),
                    "projectURL": doc.get("projectURL"),

                    "title_embedding_input": cleaned_doc.get("title_embedding_input"),
                    "subtitle_embedding_input": cleaned_doc.get("subtitle_embedding_input"),
                    "description_embedding_input": cleaned_doc.get("description_embedding_input"),
                    "keywords_embedding_input": cleaned_doc.get("keywords_embedding_input"),
                    "project_embedding_input": cleaned_doc.get("project_embedding_input"),
                })
            # Done with this KO
            continue
        else:
            # Pages existed but yielded no usable chunks (all non-strings or cleaned to empty)
            print(f"[WARN] No usable chunks for _orig_id={doc.get('_orig_id')}; indexing meta only")
            processed_documents.append(_make_meta_doc(doc, cleaned_doc))
            continue

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

