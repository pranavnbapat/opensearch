import os
import json
from dotenv import load_dotenv
import requests
from requests.auth import HTTPBasicAuth
from opensearchpy import OpenSearch

load_dotenv()

OPENSEARCH_URL = os.getenv("OPENSEARCH_URL")
AUTH = (os.getenv("OPENSEARCH_USR"), os.getenv("OPENSEARCH_PWD"))
INDEX_NAME = os.getenv("RAG_INDEX_NAME", "euf_rag")

if not all(AUTH) or not OPENSEARCH_URL:
    raise ValueError("Missing OpenSearch credentials or URL. Check your .env file!")

client = OpenSearch(
    hosts=[OPENSEARCH_URL],
    http_auth=AUTH,
    use_ssl=True,
    timeout=60,
    max_retries=3,
    retry_on_timeout=True,
    verify_certs=False  # Set to True if using proper SSL certs
)

JSON_FILE = "raw_data/final_output_16_04-2025_18-09-24.json"
BULK_BATCH_SIZE = 500

# ---- Load JSON data ----
with open(JSON_FILE, 'r', encoding='utf-8') as f:
    documents = json.load(f)

def prepare_document(doc):
    # Extract content pages if available
    content_pages = []
    if "ko_content" in doc and isinstance(doc["ko_content"], list):
        for item in doc["ko_content"]:
            pages = item.get("content", {}).get("content_pages", [])
            content_pages.extend(pages)

    # Prepare final OpenSearch document
    return {
        "@id": doc.get("@id", ""),
        "title": doc.get("title", ""),
        "summary": doc.get("summary", ""),
        "dateCreated": doc.get("dateCreated", ""),
        "languages": doc.get("languages", []),
        "locations": doc.get("locations", []),
        "keywords": doc.get("keywords", []),
        "purpose": doc.get("purpose", []),
        "creators": doc.get("creators", []),
        "projectName": doc.get("projectName", ""),
        "projectAcronym": doc.get("projectAcronym", ""),
        "fileType": doc.get("fileType", ""),
        "fileTypeCategories": doc.get("fileTypeCategories", []),
        "topics": doc.get("topics", []),
        "subtopics": doc.get("subtopics", []),
        "text": "\n\n".join(content_pages)
    }

# ---- Helper function to bulk index documents ----
def bulk_index(docs):
    bulk_payload = ""
    for doc in docs:
        action_metadata = {
            "index": {
                "_index": INDEX_NAME,
            }
        }
        bulk_payload += json.dumps(action_metadata) + "\n"
        bulk_payload += json.dumps(doc) + "\n"

    # Perform bulk request
    response = requests.post(
        f"{OPENSEARCH_URL}/_bulk",
        headers={"Content-Type": "application/json"},
        data=bulk_payload,
        auth=HTTPBasicAuth(AUTH[0], AUTH[1])
    )

    if response.status_code == 200:
        result = response.json()
        if result.get('errors'):
            print("⚠️ Some documents failed to ingest.")
        else:
            print(f"✅ Successfully ingested {len(docs)} documents.")
    else:
        print(f"❌ Bulk request failed with status {response.status_code}: {response.text}")

# ---- Main ingestion loop ----
batch = []
for idx, doc in enumerate(documents, 1):
    prepared = prepare_document(doc)
    batch.append(prepared)

    if idx % BULK_BATCH_SIZE == 0:
        print(f"Uploading batch ending at document {idx}...")
        bulk_index(batch)
        batch = []

# Upload remaining documents
if batch:
    print(f"Uploading final batch of {len(batch)} documents...")
    bulk_index(batch)

print("🎯 All documents processed.")
