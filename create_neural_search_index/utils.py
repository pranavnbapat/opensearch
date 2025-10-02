# utils.py

import hashlib
import os
import re
import unicodedata
import urllib3

from dotenv import load_dotenv

from opensearchpy import OpenSearch, helpers


# Disable SSL warnings (for self-signed certs)
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# Load environment variables from .env
load_dotenv()

BASE_MODEL_CONFIG = {
    # "minilml12v2": {
    #     "tokenizer": "sentence-transformers/all-MiniLM-L12-v2",
    #     "dimension": 384,
    #     "pipeline": "neural_search_pipeline_minilml12v2",
    #     "index": "neural_search_index_minilml12v2"
    # },
    "mpnetv2": {
        "tokenizer": "sentence-transformers/all-mpnet-base-v2",
        "dimension": 768,
        "pipeline": "neural_search_pipeline_mpnetv2",
        "index": "neural_search_index_mpnetv2"
    },
    # "msmarco": {
    #     "tokenizer": "sentence-transformers/msmarco-distilbert-base-tas-b",
    #     "dimension": 768,
    #     "pipeline": "neural_search_pipeline_msmarco_distilbert_v2",
    #     "index": "neural_search_index_msmarco_distilbert"
    # }
}

MODEL_LOWERCASE_REQUIRED = {
    "minilm": True,
    "mpnet": True,
    "msmarco": True,
    "distilbert": True,
    "e5": False,
    "bge": False
}

# Extract OpenSearch settings
OPENSEARCH_URL = os.getenv("OPENSEARCH_URL")
AUTH = (os.getenv("OPENSEARCH_USR"), os.getenv("OPENSEARCH_PWD"))

# Ensure credentials exist
if not all(AUTH):
    raise ValueError("OpenSearch credentials (OPENSEARCH_USR & OPENSEARCH_PWD) are missing!")

if not OPENSEARCH_URL:
    raise ValueError("Missing OPENSEARCH_URL in .env file")

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

def compute_source_hash(doc: dict) -> str:
    """
    Hash the parts that should force a re-embed if they change.
    Keep it stable across runs by joining in a fixed order.
    """
    parts = [
        str(doc.get("title", "")),
        str(doc.get("subtitle", "")),
        str(doc.get("description", "")),
        "||".join(doc.get("keywords") or []),
        "||".join(doc.get("topics") or []),
        "||".join(doc.get("themes") or []),
    ]
    # Fold in flattened pages if present
    flat = doc.get("ko_content_flat") or []
    if isinstance(flat, list):
        parts.append("||".join(str(p) for p in flat))
    elif isinstance(flat, str):
        parts.append(flat)

    h = hashlib.sha256("␟".join(parts).encode("utf-8")).hexdigest()
    return h

def maybe_lowercase(text, model_key):
    if not isinstance(text, str):
        return text
    return text.lower() if MODEL_LOWERCASE_REQUIRED.get(model_key, False) else text

def normalise_whitespace(text):
    """Trim and normalise all internal spaces."""
    if not isinstance(text, str):
        return text
    return re.sub(r'\s+', ' ', text.strip())

def remove_quotes_and_junk(text):
    """Remove common quote marks, pipes, accents."""
    return re.sub(r'[“”‘’"`´|]', '', text)

def preserve_apostrophes_only_in_words(text):
    """Removes all punctuation except in-word apostrophes."""
    return re.sub(r"(?!\B'\b)(?!\b'\B)[^\w\s\-_']", '', text)

def restore_urls(text, url_map):
    """Replace URL placeholders with the original URLs."""
    for placeholder, url in url_map.items():
        text = text.replace(placeholder, url)
    return text

def extract_and_mask_urls(text):
    """Find all URLs and replace them with placeholders like URL_0."""
    url_pattern = r'https?://\S+'
    urls = re.findall(url_pattern, text)
    url_map = {}
    for i, url in enumerate(urls):
        placeholder = f" URL_{i} "  # Space-padded placeholder
        text = text.replace(url, placeholder)
        url_map[placeholder.strip()] = url  # Strip to match during restore
    return text, url_map

def clean_text_light(text):
    """
    Light cleaning: For title, summary, project name, acronym.
    - Trim leading/trailing whitespace
    - Normalize spaces (replace \n, \t, multiple spaces with single space)
    - Preserve original casing and punctuation
    """
    if not isinstance(text, str):
        return text

    return normalise_whitespace(text)

def clean_text_moderate(text):
    """
    Recommended for: keywords, topics, subtopics, file types
    Handles:
      - Excessive punctuation (except hyphen, underscore, apostrophes in contractions)
      - Fancy quote removal
      - Pipe character and stray accents
    """
    if not isinstance(text, str):
        return text

    text = normalise_whitespace(text)
    text = remove_quotes_and_junk(text)
    text = preserve_apostrophes_only_in_words(text)
    return text

def clean_text_extensive(text, preserve_numbers=True):
    """
    Extensive cleaning: For content pages.
    - Remove all special characters except hyphens and underscores
    - Remove standalone numbers
    - Remove escape sequences (\n, \r, \t, etc.)
    - Normalize accented characters
    - Remove extra spaces
    """
    if not isinstance(text, str):
        return text

    text, url_map = extract_and_mask_urls(text)

    # Normalize Unicode characters (preserves accents)
    text = unicodedata.normalize('NFKC', text)

    text = re.sub(r'(?<=\S)[\n\t](?=\S)', ' ', text)  # Replace \n or \t between non-spaces with space
    text = re.sub(r'[\n\t\r\f\v]', ' ', text)  # Remove standalone escape characters

    text = remove_quotes_and_junk(text)

    text = preserve_apostrophes_only_in_words(text)

    if not preserve_numbers:
        text = re.sub(r'\b\d+\b', '', text)

    text = normalise_whitespace(text)

    text = restore_urls(text, url_map)

    return text

def remove_extra_quotes(text):
    """
    Removes extra leading/trailing double quotes but keeps quotes within the text.
    """
    if isinstance(text, str):
        text = text.strip()
        text = re.sub(r'^[\'"]+|[\'"]+$', '', text)  # Removes extra quotes from both ends
    return text

def chunk_text_by_tokens(text, tokenizer, max_tokens=480, overlap=80):
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

        token_count = len(tokenizer.encode(chunk_text, add_special_tokens=False, truncation=True, max_length=512))
        if token_count > 512:
            print(f"Warning: Overlong chunk detected ({token_count} tokens)")

        chunks.append({
            "text": chunk_text.strip(),
            "token_count": token_count
        })

        # Move to next chunk with overlap
        start += (max_tokens - overlap)
    return chunks

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

def generate_bulk_actions_upsert(docs, index_name, pipeline_name):
    """
    Upsert all docs. We send the full _source as 'doc' and set doc_as_upsert=True.
    We also pass 'pipeline' so ingest runs on creation (and often on update).
    """
    for doc in docs:
        yield {
            "_op_type": "update",
            "_index": index_name,
            "_id": doc["_orig_id"],
            "doc": doc,
            "doc_as_upsert": True,
            "pipeline": pipeline_name,
        }

