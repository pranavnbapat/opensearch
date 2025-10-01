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
    """
    HNSW (k-NN) quick reference for the vector fields below:

    • dimension (int):
        Must equal your model's embedding size (e.g. 768). If it doesn't match, indexing fails.

    • method.parameters.m (int):
        Max neighbours per node (graph degree). Typical = 16 (8–32 common).
        Higher m ⇒ larger index + slower build, slightly better recall.

    • method.parameters.ef_construction (int):
        Candidate list size during graph build. Typical = 128–512.
        Higher ef_construction ⇒ more RAM + slower build, better recall.

    • index.knn.algo_param.ef_search (int)  [set via indices.put_settings below]:
        Candidate list size at query time. Typical = 64–200; we use 100 here.
        Higher ef_search ⇒ slower queries, better recall.

    Notes:
      - These are HNSW graph params (not token lengths, not model dims—except 'dimension').
      - Tuning order: try raising ef_search first (cheap). Changing m/ef_construction requires reindex.
      - Distance: 'space_type' may be 'l2' or 'cosinesimil'. If you switch to cosine, ensure vectors are normalised.
    """

    try:
        client.indices.delete(index=INDEX_NAME, ignore=[400, 404])  # Delete index if it exists
        print(f"Deleted index: {INDEX_NAME}")

        # Recreate the index with required settings and mappings
        client.indices.create(
            index=INDEX_NAME,
            body={
                "settings": {
                    "index.knn": True,
                    "index.default_pipeline": PIPELINE_NAME
                },
                "mappings": {
                    "properties": {
                        "title_embedding": {
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
                        "subcategories": {"type": "keyword"},
                        "languages": {"type": "keyword"},
                        "intended_purposes": {"type": "keyword"},

                        # "dateCreated": {"type": "date"},
                        "date_of_completion": {"type": "date"},

                        "creators": {"type": "text"},

                        "project_name": {"type": "text"},
                        # "projectName": {"type": "text"},

                        "project_acronym": {"type": "keyword"},
                        # "projectAcronym": {"type": "keyword"},

                        "project_id": {"type": "keyword"},
                        "project_type": {"type": "keyword"},
                        "project_display_name": {"type": "keyword"},

                        "project_url": {"type": "keyword"},
                        # "projectURL": {"type": "keyword"},

                        "parent_id": {"type": "keyword"},
                        "chunk_index": {"type": "integer"},
                        "content_chunk": {"type": "text"},
                        "page_index": {"type": "integer"},  # which page in ko_content_flat
                        "within_page_chunk_index": {"type": "integer"},  # order of this chunk within that page
                        "chunk_token_count": {"type": "integer"},  # token count reported by your chunker
                        "content_char_len": {"type": "integer"},  # len(content_chunk) after cleaning
                        "ko_id": {"type": "keyword"},  # stable KO id to group chunks
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
        "ko_id": doc.get("_orig_id"),
        "chunk_index": -1,
        "page_index": -1,
        "within_page_chunk_index": -1,
        "chunk_token_count": 0,
        "content_char_len": 0,
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
        "subcategories": doc.get("subcategories") or doc.get("subcategory"),

        # "dateCreated": cleaned_doc.get("dateCreated"),
        "date_of_completion": cleaned_doc.get("date_of_completion"),

        "creators": doc.get("creators"),
        "intended_purposes": doc.get("intended_purposes"),

        # "projectName": doc.get("project_name"),
        "project_name": doc.get("project_name"),

        # "projectAcronym": doc.get("project_acronym"),
        "project_acronym": doc.get("project_acronym"),

        "project_id": doc.get("project_id"),
        "project_type": doc.get("project_type"),
        "project_display_name": doc.get("project_display_name"),

        # "projectURL": doc.get("project_url"),
        "project_url": doc.get("project_url"),

        "title_embedding_input": cleaned_doc.get("title_embedding_input"),
        "subtitle_embedding_input": cleaned_doc.get("subtitle_embedding_input"),
        "description_embedding_input": cleaned_doc.get("description_embedding_input"),
        "project_embedding_input": cleaned_doc.get("project_embedding_input"),
        "keywords_embedding_input": cleaned_doc.get("keywords_embedding_input"),
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
        project_name_raw = str(doc.get("project_name", "")).strip()
        project_acronym_raw = str(doc.get("project_acronym", "")).strip()

        # If one of (name/acronym) is missing, mirror the other. If both empty, skip the KO.
        if not project_name_raw and project_acronym_raw:
            project_name_raw = project_acronym_raw
        elif not project_acronym_raw and project_name_raw:
            project_acronym_raw = project_name_raw
        elif not project_name_raw and not project_acronym_raw:
            continue  # no project identity at all → skip

        cleaned_doc["title"] = clean_text_light(remove_extra_quotes(title_raw))
        cleaned_doc["subtitle"] = clean_text_light(remove_extra_quotes(subtitle_raw))
        cleaned_doc["description"] = clean_text_light(remove_extra_quotes(description_raw))

        original_doc["title"] = title_raw
        original_doc["description"] = description_raw

        original_doc["project_name"] = project_name_raw
        original_doc["project_acronym"] = project_acronym_raw

        # Ensure display name is available if present in JSON
        original_doc["project_display_name"] = str(doc.get("project_display_name", "")).strip()

        # proj_url_val = doc.get("project_url") or doc.get("projectURL")
        proj_url_val = doc.get("project_url")

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
        cleaned_doc["dateCreated"] = None
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

                cleaned_doc["dateCreated"] = dt.strftime("%Y-%m-%d")
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
        for page_idx, page in enumerate(flat_pages):
            if not isinstance(page, str):
                continue
            cleaned = clean_text_extensive(page, preserve_numbers=True)
            cleaned = maybe_lowercase(cleaned, model_key)

            within_idx = 0
            chunks = chunk_text_by_tokens(cleaned, tokenizer)
            for ch in chunks:
                txt = ch.get("text", "")
                tok = int(ch.get("token_count", 0))

                if not txt.strip():
                    continue
                if tok > 512:
                    print(
                        f"[WARN] Overlong chunk (>512) for _orig_id={doc.get('_orig_id')} page={page_idx} idx={within_idx}")

                all_chunks.append({
                    "text": txt,
                    "token_count": tok,
                    "page_index": page_idx,
                    "within_page_chunk_index": within_idx,
                    "char_len": len(txt),
                })
                within_idx += 1

        if all_chunks:
            ko_id_val = doc.get("_orig_id")
            for i, ch in enumerate(all_chunks):
                token_count = int(ch.get("token_count", 0))
                page_idx = ch.get("page_index", 0)
                within_idx = ch.get("within_page_chunk_index", i)

                processed_documents.append({
                    "_orig_id": f"{ko_id_val}::c{i}",
                    "parent_id": ko_id_val,
                    "ko_id": ko_id_val,

                    "chunk_index": i,
                    "page_index": page_idx,
                    "within_page_chunk_index": within_idx,
                    "chunk_token_count": token_count,
                    "content_char_len": int(ch.get("char_len", 0)),

                    "content_chunk": ch["text"],
                    "content_embedding_input": ch["text"],

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
                    # "subcategory": doc.get("subcategory"),
                    "subcategories": doc.get("subcategories") or doc.get("subcategory"),

                    # "dateCreated": cleaned_doc.get("dateCreated"),
                    "date_of_completion": cleaned_doc.get("date_of_completion"),

                    "creators": doc.get("creators"),
                    "intended_purposes": doc.get("intended_purposes"),

                    # "projectName": original_doc.get("project_name"),
                    "project_name": original_doc.get("project_name"),

                    "project_display_name": original_doc.get("project_display_name"),

                    # "projectAcronym": original_doc.get("project_acronym"),
                    "project_acronym": original_doc.get("project_acronym"),

                    "project_id": doc.get("project_id"),
                    "project_type": doc.get("project_type"),

                    "project_url": proj_url_val,
                    # "projectURL": proj_url_val,
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

