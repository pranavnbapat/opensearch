import os
import json
from tqdm import tqdm
from dotenv import load_dotenv
from pathlib import Path
from sentence_transformers import SentenceTransformer
from opensearchpy import OpenSearch, helpers

# Load .env from parent directory
env_path = Path(__file__).resolve().parents[1] / '.env'
load_dotenv(dotenv_path=env_path)

OPENSEARCH_URL = os.getenv("OPENSEARCH_URL")
AUTH = (os.getenv("OPENSEARCH_USR"), os.getenv("OPENSEARCH_PWD"))
INDEX_NAME = os.getenv("TEST_RECOMM_INDEX_NAME", "test_recomm")

if not OPENSEARCH_URL:
    raise ValueError("OPENSEARCH_URL is missing in your .env file!")

if not all(AUTH):
    raise ValueError("OpenSearch credentials (OPENSEARCH_USR & OPENSEARCH_PWD) are missing!")

# === Create OpenSearch client ===
client = OpenSearch(
    hosts=[OPENSEARCH_URL],
    http_auth=AUTH,
    use_ssl=True,
    verify_certs=False,
    timeout=60,
    max_retries=3,
    retry_on_timeout=True,
)

model = SentenceTransformer('distilbert-base-nli-stsb-mean-tokens')

with open('../raw_data/final_output_16_04-2025_18-09-24.json', 'r', encoding='utf-8') as f:
    data = json.load(f)

# Define index mapping
if not client.indices.exists(index=INDEX_NAME):
    client.indices.create(index=INDEX_NAME, body={
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
                "embedding": {"type": "knn_vector", "dimension": 768}
            }
        }
    })

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

        # Safe string and list conversion
        def safe_join(field):
            return " ".join(field) if isinstance(field, list) else str(field or "")

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
            "_index": INDEX_NAME,
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

helpers.bulk(client, docs)
print("Indexing complete.")
