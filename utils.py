# utils.py

import os
import re
import unicodedata
import urllib3

from dotenv import load_dotenv
from opensearchpy import OpenSearch
from opensearchpy.helpers import bulk

# Disable SSL warnings (for self-signed certs)
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# Load environment variables from .env
load_dotenv()

# Extract OpenSearch settings
OPENSEARCH_URL = os.getenv("OPENSEARCH_URL")
AUTH = (os.getenv("OPENSEARCH_USR"), os.getenv("OPENSEARCH_PWD"))

# Ensure credentials exist
if not all(AUTH):
    raise ValueError("OpenSearch credentials (OPENSEARCH_USR & OPENSEARCH_PWD) are missing!")

# Create OpenSearch client
client = OpenSearch(
    hosts=[OPENSEARCH_URL],
    http_auth=AUTH,
    use_ssl=True,
    verify_certs=False,
    timeout=60,
    max_retries=3,
    retry_on_timeout=True,
)

RECOMM_SYS_SUPPORTED_MODELS = {
    "mpnet": "sentence-transformers/all-mpnet-base-v2",
    "minilm": "sentence-transformers/all-MiniLM-L6-v2",
    "e5": "intfloat/e5-base",
    "bge": "BAAI/bge-base-en-v1.5",
    "distilbert": "distilbert-base-nli-stsb-mean-tokens",
}


def clean_text_light(text):
    """
    Light cleaning: For title, summary, project name, acronym.
    - Lowercase conversion
    - Trim whitespace
    - Remove extra spaces and newlines
    - Preserve alphanumeric characters, accents, and essential punctuation
    """
    if not isinstance(text, str):
        return text

    text = text.strip()  # Remove leading/trailing spaces
    text = re.sub(r'\s+', ' ', text)  # Normalize spaces
    return text


def clean_text_moderate(text):
    """
    Moderate cleaning: For keywords, topics, subtopics, locations, languages, file type.
    - Lowercase conversion
    - Remove excessive punctuation (keep hyphens & underscores)
    - Normalize spaces
    """
    if not isinstance(text, str):
        return text

    text = text.lower().strip()
    text = re.sub(r'\s+', ' ', text)  # Normalize spaces
    text = re.sub(r'[“”"‘’]', '', text)  # Remove fancy quotes
    text = re.sub(r'[|]', '', text)  # Remove pipe characters
    text = re.sub(r'[^\w\s\-_]', '', text)  # Remove all other punctuation except hyphen/underscore
    return text


def chunk_text_by_tokens(text, tokenizer, max_tokens=480, overlap=80):
    # tokens = tokenizer.encode(text, truncation=False)

    # Tokenize the text into token IDs (no truncation, returns warning if >512)
    tokens = tokenizer.encode(text, add_special_tokens=False)

    chunks = []
    start = 0
    while start < len(tokens):
        end = min(start + max_tokens, len(tokens))
        chunk_tokens = tokens[start:end]

        # Safety check: never let a chunk exceed 512 tokens (just in case overlap math breaks it)
        if len(chunk_tokens) > 512:
            chunk_tokens = chunk_tokens[:512]

        # Decode token IDs back into text
        chunk_text = tokenizer.decode(chunk_tokens, skip_special_tokens=True)

        token_count = len(tokenizer.encode(chunk_text, add_special_tokens=False))
        if token_count > 512:
            print(f"Warning: Overlong chunk detected ({token_count} tokens)")

        chunks.append({
            "text": chunk_text.strip(),
            "token_count": token_count
        })

        # chunks.append(chunk_text.strip())

        # Move to next chunk with overlap
        start += (max_tokens - overlap)
    return chunks


def clean_text_extensive(text):
    """
    Extensive cleaning: For content pages.
    - Convert to lowercase
    - Remove all special characters except hyphens and underscores
    - Remove standalone numbers
    - Remove escape sequences (\n, \r, \t, etc.)
    - Normalize accented characters
    - Remove extra spaces
    """
    if not isinstance(text, str):
        return text

    text = text.lower().strip()

    # Normalize Unicode characters (preserves accents)
    text = unicodedata.normalize('NFKC', text)

    # Remove all escape sequences (e.g., \n, \r, \t, \x, \u)
    text = re.sub(r'\\[a-zA-Z0-9]+', '', text)

    # Remove newlines, tabs, carriage returns explicitly
    text = re.sub(r'[\n\r\t\f\v]', ' ', text)

    # Remove quotation marks, pipes, and other unnecessary punctuation
    text = re.sub(r'[“”"‘’]', '', text)  # Remove fancy quotes
    text = re.sub(r'[|]', '', text)  # Remove pipe characters

    # Remove any remaining non-alphanumeric characters (except hyphens and underscores)
    text = re.sub(r'[^a-zA-Z0-9\s\-_]', '', text)

    # Remove standalone numbers
    text = re.sub(r'\b\d+\b', '', text)

    # Normalize spaces
    text = re.sub(r'\s+', ' ', text)

    return text.strip()


def remove_extra_quotes(text):
    """
    Removes extra leading/trailing double quotes but keeps quotes within the text.
    """
    if isinstance(text, str):
        text = text.strip()
        text = re.sub(r'^[\'"]+|[\'"]+$', '', text)  # Removes extra quotes from both ends
    return text


def safe_join(field):
    return " ".join(field) if isinstance(field, list) else str(field or "")


def chunked_bulk_upload(docs, chunk_size=10):
    for i in range(0, len(docs), chunk_size):
        chunk = docs[i:i+chunk_size]
        try:
            success, errors = bulk(client, chunk)
            print(f"✅ Chunk {i//chunk_size + 1}: {success} documents indexed.")
            if errors:
                print(f"⚠️ Chunk {i//chunk_size + 1} had errors: {errors}")
        except Exception as e:
            print(f"❌ Error indexing chunk {i//chunk_size + 1}: {e}")
