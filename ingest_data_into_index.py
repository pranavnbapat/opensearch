import os
import requests
import urllib3
import json
import time
from dotenv import load_dotenv

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# Load environment variables
load_dotenv()

# OpenSearch Configuration
OPENSEARCH_URL = "https://opensearch.nexavion.com"
AUTH = (os.getenv("OPENSEARCH_USR"), os.getenv("OPENSEARCH_PWD"))
HEADERS = {"Content-Type": "application/json"}

INDEX_NAME = os.getenv("INDEX_NAME")  # Neural search index name
RETRY_LIMIT = 5  # Maximum retries per document
TIMEOUT = 10  # Timeout per request


# **Function to extract & transform document**
def transform_document(raw_doc):
    """Transforms a raw document into OpenSearch format"""
    content_pages = ""

    if isinstance(raw_doc.get("ko_content", []), list) and raw_doc["ko_content"]:
        content = raw_doc["ko_content"][0].get("content", {})
        content_pages = " ".join(content.get("content_pages", []))

    return {
        "title": raw_doc.get("title", ""),
        "summary": raw_doc.get("summary", ""),
        "projectName": raw_doc.get("projectName", ""),
        "projectAcronym": raw_doc.get("projectAcronym", ""),
        "keywords": raw_doc.get("keywords", []),
        "locations": raw_doc.get("locations", []),
        "content": {"content_pages": content_pages},
    }


# **Function to check if OpenSearch is online**
def check_opensearch_health():
    """Check if OpenSearch is reachable before inserting data"""
    url = f"{OPENSEARCH_URL}/_cluster/health"
    try:
        response = requests.get(url, auth=AUTH, headers=HEADERS, timeout=TIMEOUT, verify=False)
        if response.status_code == 200:
            return True
    except requests.exceptions.RequestException:
        return False
    return False


# **Function to check if document already exists**
def document_exists(doc_id):
    """Returns True if the document exists, otherwise False"""
    url = f"{OPENSEARCH_URL}/{INDEX_NAME}/_doc/{doc_id}"
    response = requests.head(url, auth=AUTH, headers=HEADERS, timeout=TIMEOUT, verify=False)
    return response.status_code == 200


# **Function to insert a document into OpenSearch**
def insert_document(doc_id, document):
    """Inserts a single document into OpenSearch with retries"""
    url = f"{OPENSEARCH_URL}/{INDEX_NAME}/_doc/{doc_id}"
    retries = 0

    while retries < RETRY_LIMIT:
        try:
            response = requests.put(url, auth=AUTH, headers=HEADERS, json=document, timeout=TIMEOUT, verify=False)

            if response.status_code in [200, 201]:
                return True  # Success

            elif response.status_code == 429:  # Too Many Requests (Rate Limit)
                wait_time = 2 ** retries  # Exponential backoff
                print(f"⚠️ Rate limited. Retrying in {wait_time}s...")
                time.sleep(wait_time)

            else:
                print(f"❌ Error {response.status_code} for doc {doc_id}: {response.text}")
                return False

        except requests.exceptions.ConnectionError:
            print(f"⚠️ Connection error for {doc_id}. Retrying in 5s...")
            time.sleep(5)

        except requests.exceptions.Timeout:
            print(f"⚠️ Timeout for {doc_id}. Retrying in 3s...")
            time.sleep(3)

        retries += 1

    print(f"❌ Skipped document {doc_id} after {RETRY_LIMIT} failed attempts.")
    return False


# **Function to ingest all documents into OpenSearch one by one**
def ingest_documents(documents):
    """Inserts documents into OpenSearch, one by one, skipping already processed ones"""
    total_docs = len(documents)
    inserted_count = 0
    skipped_count = 0

    if not check_opensearch_health():
        print("🚨 OpenSearch is unreachable! Check your connection and restart the script.")
        return

    for i, raw_doc in enumerate(documents, 1):
        doc_id = raw_doc["_id"]

        # **Check if document already exists before inserting**
        if document_exists(doc_id):
            skipped_count += 1
            print(f"⏩ Skipping already existing document: {doc_id}")  # ✅ Prints skipped document ID
            continue

        transformed_doc = transform_document(raw_doc)
        success = insert_document(doc_id, transformed_doc)

        if success:
            inserted_count += 1

        # Log progress every 100 documents
        if i % 100 == 0 or i == total_docs:
            print(f"✅ Inserted {inserted_count}/{i} | Skipped {skipped_count} already existing documents...")

        # Small delay to avoid overloading OpenSearch
        time.sleep(0.1)

    print(f"🎉 Finished! Successfully inserted {inserted_count}/{total_docs} documents. Skipped {skipped_count} existing ones.")


# **Main Execution**
if __name__ == "__main__":
    print("🚀 Ingesting documents into OpenSearch...")

    # Load documents
    with open("final_output.json", "r", encoding="utf-8") as f:
        data = json.load(f)

    # Ingest documents one by one
    ingest_documents(data)
