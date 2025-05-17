# rag/ingest_data_rag.py

import json
import nltk
import sys

nltk.download('punkt', quiet=True)
from nltk.tokenize import sent_tokenize

from opensearchpy.helpers import bulk

from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[1]))

from sentence_transformers import SentenceTransformer

from utils import client

index_name = "rag_paraphrase_multilingual_minilm_l12_v2"
model = SentenceTransformer("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")

JSON_FILE = "../raw_data/final_output_10_05-2025_21-09-42.json"
BULK_BATCH_SIZE = 300

with open(JSON_FILE, 'r', encoding='utf-8') as f:
    documents = json.load(f)

# Delete index if it exists
if client.indices.exists(index=index_name):
    print(f"Index '{index_name}' already exists. Deleting...")
    client.indices.delete(index=index_name)
    print(f"Index '{index_name}' deleted.")

# Create new index with full mapping
dimension = 384  # embedding size from MiniLM-L12-v2
mapping = {
    "settings": {
        "index": {
            "knn": True,
            "knn.algo_param.ef_search": 100
        }
    },
    "mappings": {
        "properties": {
            "@id": {"type": "keyword"},
            "title": {"type": "text"},
            "summary": {"type": "text"},
            "keywords": {"type": "text"},
            "creators": {"type": "text"},
            "projectName": {"type": "text"},
            "projectAcronym": {"type": "keyword"},
            "projectURL": {"type": "keyword"},
            "fileType": {"type": "keyword"},
            "topics": {"type": "text"},
            "subtopics": {"type": "text"},
            "project_type": {"type": "keyword"},
            "content_chunk": {"type": "text"},
            "embedding": {
                "type": "knn_vector",
                "dimension": dimension,
                "method": {
                    "name": "hnsw",
                    "space_type": "cosinesimil",
                    "engine": "nmslib"
                }
            }
        }
    }
}

print(f"Creating index '{index_name}'...")
client.indices.create(index=index_name, body=mapping)
print(f"Index '{index_name}' created.")


# Chunking function
def chunk_text(text, max_tokens=200, overlap=30):
    sentences = sent_tokenize(text)
    chunks = []
    current_chunk = []
    current_len = 0

    for sentence in sentences:
        tokens = sentence.split()
        if current_len + len(tokens) > max_tokens:
            chunks.append(" ".join(current_chunk))
            current_chunk = current_chunk[-overlap:]  # retain overlap
            current_len = sum(len(s.split()) for s in current_chunk)
        current_chunk.append(sentence)
        current_len += len(tokens)

    if current_chunk:
        chunks.append(" ".join(current_chunk))

    return chunks


# Bulk index function
def bulk_index_rag_docs(docs):
    actions = [
        {
            "_index": index_name,
            "_source": doc
        }
        for doc in docs
    ]
    success, failed = bulk(client, actions, stats_only=False)

    print(f"✅ Successfully ingested {success} chunks.")
    if failed:
        print("⚠️ Some documents failed.")

buffer = []
total_chunks = 0
doc_count = 0

for idx, doc in enumerate(documents, 1):
    title = doc.get("title", "")
    summary = doc.get("summary", "")
    langs = doc.get("languages", [])
    locs = doc.get("locations", [])
    project = doc.get("projectName", "")

    # Extract and validate content
    content_pages = []
    for item in doc.get("ko_content", []):
        pages = item.get("content", {}).get("content_pages", [])
        if isinstance(pages, list):
            content_pages.extend(pages)

    full_text = "\n".join(content_pages).strip()
    if not full_text:
        continue

    chunks = chunk_text(full_text)
    if not chunks:
        continue

    embeddings = model.encode(chunks, show_progress_bar=False)

    doc_count += 1
    print(f"📄 Processed document {doc_count}/{len(documents)}: {title[:60]}")

    for chunk, emb in zip(chunks, embeddings):
        buffer.append({
            "@id": doc.get("@id", ""),
            "title": title,
            "summary": summary,
            "keywords": doc.get("keywords", []),
            "creators": doc.get("creators", []),
            "projectName": project,
            "projectAcronym": doc.get("projectAcronym", ""),
            "projectURL": doc.get("projectURL", ""),
            "fileType": doc.get("fileType", ""),
            "topics": doc.get("topics", []),
            "subtopics": doc.get("subtopics", []),
            "project_type": doc.get("project_type", ""),
            "content_chunk": chunk,
            "embedding": emb.tolist()
        })

    if len(buffer) >= BULK_BATCH_SIZE:
        bulk_index_rag_docs(buffer)
        total_chunks += len(buffer)
        buffer = []

# Upload any remaining
if buffer:
    bulk_index_rag_docs(buffer)
    total_chunks += len(buffer)

print(f"🎯 All done. Total chunks indexed: {total_chunks}")
print(f"📊 Total documents processed: {doc_count}")
