# create_neural_search_index/create_neural_search_index.py

"""
create_neural_search_index.py — incremental OpenSearch ingestion for neural search.

What it does
- Finds the most recently modified JSON in ../raw_data_dev (or ../raw_data when dev_mode=False).
- Cleans fields and chunks `ko_content_flat` using the model’s tokenizer; prepares *embedding inputs* (actual vectors are created by the ingest pipeline).
- Ensures the target index exists with k-NN (HNSW) mappings and sets `index.default_pipeline` to the model’s pipeline.
- Emits **one meta document per KO** (`chunk_index == -1`) and zero or more **chunk documents** (`chunk_index >= 0`).

Incremental behaviour
- Loads existing meta docs (`term` query on `chunk_index: -1`).
- A KO is considered **changed** if **any** of these differ vs. what’s indexed: `ko_updated_at`, `proj_updated_at`, or `source_hash` (hash of title/subtitle/description/keywords/topics/themes and the ordered `ko_content_flat`).
- **Changed KOs**: delete old docs by `ko_id` (scoped `_delete_by_query`), then bulk upsert new meta + chunks.
- **New KOs**: bulk upsert only (no delete). **Unchanged KOs** are skipped.

IDs & grouping
- Meta ID: `"{_orig_id}::meta"`, Chunk IDs: `"{_orig_id}::c{n}"`.
- All docs for a KO carry `ko_id == _orig_id` for grouping and targeted deletes.
- Meta stores `source_hash` and `ingested_at` (UTC) for change detection and audit.
"""

import json

from collections import defaultdict
from datetime import datetime, UTC

from opensearchpy.helpers import scan
from transformers import AutoTokenizer

from utils import *

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
    latest_file = max(files, key=lambda x: os.path.getmtime(os.path.join(folder, x)))
    return os.path.join(folder, latest_file)


def ensure_index_exists(INDEX_NAME: str, PIPELINE_NAME: str, VECTOR_DIM: int):
    if client.indices.exists(index=INDEX_NAME):
        return

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

                    "date_of_completion": {"type": "date"},

                    "creators": {"type": "text"},

                    "project_name": {"type": "text"},

                    "project_acronym": {"type": "keyword"},

                    "project_id": {"type": "keyword"},
                    "project_type": {"type": "keyword"},
                    "project_display_name": {"type": "keyword"},

                    "project_url": {"type": "keyword"},

                    "parent_id": {"type": "keyword"},
                    "chunk_index": {"type": "integer"},
                    "content_chunk": {"type": "text"},
                    "page_index": {"type": "integer"},  # which page in ko_content_flat
                    "within_page_chunk_index": {"type": "integer"},  # order of this chunk within that page
                    "chunk_token_count": {"type": "integer"},
                    "content_char_len": {"type": "integer"},  # len(content_chunk) after cleaning
                    "ko_id": {"type": "keyword"},  # stable KO id to group chunks

                    "ko_created_at": {"type": "date", "format": "strict_date_optional_time||epoch_millis"},
                    "ko_updated_at": {"type": "date", "format": "strict_date_optional_time||epoch_millis"},
                    "proj_created_at": {"type": "date", "format": "strict_date_optional_time||epoch_millis"},
                    "proj_updated_at": {"type": "date", "format": "strict_date_optional_time||epoch_millis"},
                }
            }
        }
    )
    print(f"Created index: {INDEX_NAME}")

    # Apply ef_search setting after recreating the index
    client.indices.put_settings(
        index=INDEX_NAME,
        body={"index": {"knn.algo_param.ef_search": 100}}
    )
    print(f"Applied ef_search setting (100) to {INDEX_NAME}")


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

        "date_of_completion": cleaned_doc.get("date_of_completion"),

        "creators": doc.get("creators"),
        "intended_purposes": doc.get("intended_purposes"),

        "project_name": doc.get("project_name"),

        "project_acronym": doc.get("project_acronym"),

        "project_id": doc.get("project_id"),
        "project_type": doc.get("project_type"),
        "project_display_name": doc.get("project_display_name"),

        "project_url": doc.get("project_url"),

        "ko_created_at": doc.get("ko_created_at"),
        "ko_updated_at": doc.get("ko_updated_at"),
        "proj_created_at": doc.get("proj_created_at"),
        "proj_updated_at": doc.get("proj_updated_at"),

        "source_hash": compute_source_hash(doc),
        "ingested_at": datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
    }

def load_existing_meta(index_name: str) -> dict:
    """
    Return dict: ko_id -> {"ko_updated_at":..., "proj_updated_at":..., "source_hash":...}
    Reads only meta docs (_orig_id endswith '::meta').
    """
    out = {}
    try:
        for hit in scan(
            client,
                index=index_name,
                query={"query": {"term": {"chunk_index": -1}}},
                _source=["ko_id", "ko_updated_at", "proj_updated_at", "source_hash"],
                size=2000,
                preserve_order=False,
                clear_scroll=True
        ):
            src = hit.get("_source", {})
            ko_id = src.get("ko_id")
            if ko_id:
                out[ko_id] = {
                    "ko_updated_at": src.get("ko_updated_at"),
                    "proj_updated_at": src.get("proj_updated_at"),
                    "source_hash": src.get("source_hash"),
                }
    except Exception as e:
        print(f"[WARN] Could not load existing meta from {index_name}: {e}")
    return out

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

        proj_url_val = doc.get("project_url")

        # Embedding inputs (with lowercasing)
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
                    "subcategories": doc.get("subcategories") or doc.get("subcategory"),

                    "date_of_completion": cleaned_doc.get("date_of_completion"),

                    "creators": doc.get("creators"),
                    "intended_purposes": doc.get("intended_purposes"),

                    "project_name": original_doc.get("project_name"),

                    "project_display_name": original_doc.get("project_display_name"),

                    "project_acronym": original_doc.get("project_acronym"),

                    "project_id": doc.get("project_id"),
                    "project_type": doc.get("project_type"),

                    "project_url": proj_url_val,

                    "ko_created_at": doc.get("ko_created_at"),
                    "ko_updated_at": doc.get("ko_updated_at"),
                    "proj_created_at": doc.get("proj_created_at"),
                    "proj_updated_at": doc.get("proj_updated_at"),
                })

            processed_documents.append(_make_meta_doc(doc, cleaned_doc))

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
    tokenizer.model_max_length = 10 ** 9
    INDEX_NAME = CONFIG["index"]
    PIPELINE_NAME = CONFIG["pipeline"]
    VECTOR_DIM = CONFIG["dimension"]

    try:
        latest_file = get_latest_json_file()
        print(f"Using file: {latest_file}")

        processed_data = process_json_for_opensearch(latest_file, tokenizer, MODEL)
        ensure_index_exists(INDEX_NAME, PIPELINE_NAME, VECTOR_DIM)
        existing_meta = load_existing_meta(INDEX_NAME)

        by_ko = defaultdict(list)
        meta_docs = {}

        for d in processed_data:
            ko = d.get("ko_id") or d.get("_orig_id")
            # meta docs end with ::meta
            if d.get("_orig_id", "").endswith("::meta"):
                meta_docs[ko] = d
            else:
                by_ko[ko].append(d)

        # Ensure KOs with only meta doc are considered
        for ko in set(meta_docs.keys()) - set(by_ko.keys()):
            by_ko[ko] = []  # no chunks, but we still want to decide/ingest the meta

        new_ko_ids = []
        changed_ko_ids = []
        unchanged_ko_ids = []
        to_write_docs = []
        removed_ko_ids = []

        for ko_id, chunks in by_ko.items():
            meta = meta_docs.get(ko_id)
            if not meta:
                # Defensive: synthesise meta if missing
                meta = _make_meta_doc({"_orig_id": ko_id}, {
                    "title": "", "subtitle": "", "description": "",
                    "keywords": [], "topics": [], "themes": [], "languages": [], "locations": []
                })

            prev = existing_meta.get(ko_id)

            if prev is None:
                # NEW KO: don't add to delete list; just write it
                new_ko_ids.append(ko_id)
                to_write_docs.append(meta)
                to_write_docs.extend(chunks)
                continue

            # else:
            #     if (meta.get("ko_updated_at") and meta["ko_updated_at"] != prev.get("ko_updated_at")) \
            #             or (meta.get("proj_updated_at") and meta["proj_updated_at"] != prev.get("proj_updated_at")) \
            #             or (meta.get("source_hash") and meta["source_hash"] != prev.get("source_hash")):
            #         changed = True

            # CHANGED?
            changed = (
                    (meta.get("ko_updated_at") and meta["ko_updated_at"] != prev.get("ko_updated_at")) or
                    (meta.get("proj_updated_at") and meta["proj_updated_at"] != prev.get("proj_updated_at")) or
                    (meta.get("source_hash") and meta["source_hash"] != prev.get("source_hash"))
            )

            if changed:
                changed_ko_ids.append(ko_id)
                to_write_docs.append(meta)
                to_write_docs.extend(chunks)
            else:
                unchanged_ko_ids.append(ko_id)

        # ---- PRUNE: delete KOs that no longer exist in the current JSON ----
        PRUNE_REMOVED_KOS = True  # set False if you ever ingest delta files (not full snapshots)

        if PRUNE_REMOVED_KOS:
            existing_ko_ids = set(existing_meta.keys())
            current_ko_ids = set(meta_docs.keys())  # meta_docs has one entry per KO you plan to keep
            removed_ko_ids = sorted(existing_ko_ids - current_ko_ids)

            if removed_ko_ids:
                print(f"[PRUNE] Will remove {len(removed_ko_ids)} KO(s) missing from current snapshot.")
                try:
                    # Delete all docs belonging to each removed KO
                    delete_kos_by_id(INDEX_NAME, removed_ko_ids, reason="missing from new JSON")
                except Exception as e:
                    print(f"[WARN] prune (removed KOs) failed: {e}")
            else:
                print("[PRUNE] No removed KOs detected.")

        total_docs = len(to_write_docs)

        to_delete_ko_ids = changed_ko_ids

        print(
            f"Plan → new={len(new_ko_ids)}, changed={len(changed_ko_ids)}, "
            f"unchanged={len(unchanged_ko_ids)}, removed={len(removed_ko_ids)}"
        )

        if to_delete_ko_ids:
            q = {"terms": {"ko_id": to_delete_ko_ids}}
            try:
                resp = client.delete_by_query(
                    index=INDEX_NAME,
                    body={"query": q},
                    refresh=True,
                    conflicts="proceed",
                    slices="auto"
                )
                print(f"Deleted old docs for {len(to_delete_ko_ids)} KO(s): deleted={resp.get('deleted')}")
            except Exception as e:
                print(f"[WARN] delete_by_query failed: {e}")

        if not to_write_docs:
            print("Nothing changed — skipping ingestion.")
            continue

        print(f"Starting ingestion of {total_docs} documents...")
        print(f"Indexing into: {INDEX_NAME}")

        batch_size = 20
        for i in range(0, total_docs, batch_size):
            batch = to_write_docs[i: i + batch_size]
            success_count, errors = helpers.bulk(
                client,
                generate_bulk_actions_upsert(batch, INDEX_NAME, PIPELINE_NAME),
                refresh="wait_for",
                stats_only=False
            )
            print(f"Batch {i // batch_size + 1}: {success_count} successes.")
            if errors:
                print(f"Batch {i // batch_size + 1}: {len(errors)} errors occurred.")

        print(f"Finished ingestion for: {MODEL}")
    except Exception as e:
        print(f"Error during {MODEL}: {e}")

