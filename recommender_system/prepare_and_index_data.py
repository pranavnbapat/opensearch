# recommender_system/prepare_and_index_data.py

import json
import logging
import sys
import urllib3

from datetime import datetime
from dotenv import load_dotenv
from pathlib import Path
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

sys.path.append(str(Path(__file__).resolve().parents[1]))

from utils import client, RECOMM_SYS_SUPPORTED_MODELS, safe_join, chunked_bulk_upload

log_dir = Path(__file__).resolve().parent / "../logs"
log_dir.mkdir(parents=True, exist_ok=True)
log_file = log_dir / f"indexing_log_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.log"

logging.basicConfig(
    filename=log_file,
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger()

# Disable SSL warnings (for self-signed certs)
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# Load .env from parent directory
env_path = Path(__file__).resolve().parents[1] / '.env'
load_dotenv(dotenv_path=env_path)

with open('../raw_data/final_output_07_05-2025_15-57-25.json', 'r', encoding='utf-8') as f:
    data = json.load(f)

for model_key, model_path in RECOMM_SYS_SUPPORTED_MODELS.items():
    index_name = f"recomm_sys_{model_key}"
    print(f"\nProcessing model: {model_key} | Index: {index_name}")

    # Delete existing index (if any)
    if client.indices.exists(index=index_name):
        client.indices.delete(index=index_name)
        print(f"Deleted existing index: {index_name}")

    model = SentenceTransformer(model_path)
    embedding_dim = model.get_sentence_embedding_dimension()

    client.indices.create(index=index_name, body={
        "settings": {
            "index": {
                "knn": True
            }
        },
        "mappings": {
            "properties": {
                "title": {"type": "text"},
                "project_name": {"type": "text"},
                "project_acronym": {"type": "text"},
                "summary": {"type": "text"},
                "topics": {"type": "text"},
                "subtopics": {"type": "text"},
                "content_pages": {"type": "text"},
                "embedding": {"type": "knn_vector", "dimension": embedding_dim}
            }
        }
    })
    print(f"Created index: {index_name}")

    model = SentenceTransformer(model_path)

    # Prepare and index documents
    def prepare_doc(obj):
        try:
            # Safely get content_pages
            ko_content = obj.get("ko_content", [])
            content_pages = []
            if isinstance(ko_content, list) and len(ko_content) > 0:
                content_pages = ko_content[0].get("content", {}).get("content_pages", [])
                if not isinstance(content_pages, list):
                    content_pages = [str(content_pages)]

            text_parts = [
                obj.get("title", "") or "",
                obj.get("projectName", "") or "",
                obj.get("projectAcronym", "") or "",
                obj.get("summary", "") or "",
                safe_join(obj.get("topics", [])),
                safe_join(obj.get("subtopics", [])),
                safe_join(content_pages),
                safe_join(obj.get("keywords", [])),
                safe_join(obj.get("purpose", []))
            ]

            full_text = " ".join(text_parts).strip()

            if not full_text:
                raise ValueError("Skipping empty document with no usable content.")

            vector = model.encode(full_text).tolist()

            return {
                "_index": index_name,
                "_source": {
                    "title": obj.get("title", ""),
                    "project_name": obj.get("projectName", ""),
                    "project_acronym": obj.get("projectAcronym", ""),
                    "summary": obj.get("summary", ""),
                    "content_pages": content_pages,
                    "creators": obj.get("creators", []),
                    "file_type": obj.get("fileType", ""),
                    "languages": obj.get("languages", []),
                    "keywords": obj.get("keywords", []),
                    "locations": obj.get("locations", []),
                    "purpose": obj.get("purpose", []),
                    "embedding": vector
                }
            }

        except Exception as e:
            print(f"Error preparing document ID {obj.get('@id', 'N/A')}: {e}")
            return None


    docs = [prepare_doc(obj) for obj in tqdm(data)]
    docs = [d for d in docs if d is not None]

    if docs:
        chunked_bulk_upload(docs, logger=logger)
        logger.info(f"Successfully indexed {len(docs)} documents into {index_name} using model '{model_key}'.")
    else:
        logger.warning(f"No documents indexed for model: {model_key}.")
