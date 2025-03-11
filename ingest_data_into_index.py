import os
import json
import re
import urllib3
from datetime import datetime
from dotenv import load_dotenv
from opensearchpy import OpenSearch, helpers

# Disable SSL warnings (if using self-signed certificates)
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# Load environment variables
load_dotenv()

# OpenSearch Configuration
OPENSEARCH_URL = "https://opensearch.nexavion.com"
AUTH = (os.getenv("OPENSEARCH_USR"), os.getenv("OPENSEARCH_PWD"))

# Ensure credentials are loaded
if not all(AUTH):
    raise ValueError("OpenSearch credentials (OPENSEARCH_USR & OPENSEARCH_PWD) are missing!")

# Connect to OpenSearch
client = OpenSearch(
    hosts=[OPENSEARCH_URL],
    http_auth=AUTH,
    use_ssl=True,
    verify_certs=False,  # Set to True if using trusted SSL certs
)

# Load JSON data from file
file_path = "opensearch_data_ingestion.json"
INDEX_NAME = os.getenv("INDEX_NAME", "neural_search_index")

with open(file_path, "r", encoding="utf-8") as file:
    data = json.load(file)

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
        print(f"⚠️ Warning: Invalid date format detected: {date_str}. Skipping conversion.")
        return None  # Return None for invalid dates

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


# Process data in batches of 10
batch_size = 10
total_docs = len(data)
print(f"Starting ingestion of {total_docs} documents in batches of {batch_size}...")

for i in range(0, total_docs, batch_size):
    batch = data[i: i + batch_size]  # Extract batch of 10 documents
    success, failed = helpers.bulk(client, generate_bulk_actions(batch))

    print(f"Batch {i // batch_size + 1}: {success} documents indexed, {failed} failed.")

print("✅ All documents successfully ingested into OpenSearch!")
