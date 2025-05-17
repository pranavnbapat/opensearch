# rag/retriever.py

import sys

from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[1]))

from sentence_transformers import SentenceTransformer
from typing import List, Dict
from utils import client

INDEX_NAME = "rag_paraphrase_multilingual_minilm_l12_v2"
EMBEDDING_DIM = 384
TOP_K = 5

model = SentenceTransformer("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")


def retrieve_top_chunks(query: str, k: int = TOP_K) -> List[Dict]:
    # 1. Embed query
    query_vector = model.encode(query).tolist()

    # 2. Construct vector search query
    search_query = {
        "size": k,
        "query": {
            "knn": {
                "embedding": {
                    "vector": query_vector,
                    "k": k
                }
            }
        },
        "_source": [
            "content_chunk", "title", "summary", "projectName", "projectAcronym", "@id"
        ]
    }

    # 3. Execute query
    response = client.search(index=INDEX_NAME, body=search_query)

    # 4. Return results
    hits = response["hits"]["hits"]
    return [
        {
            "chunk": hit["_source"]["content_chunk"],
            "title": hit["_source"].get("title"),
            "summary": hit["_source"].get("summary"),
            "project": hit["_source"].get("projectName"),
            "id": hit["_source"].get("@id"),
            "score": hit["_score"]
        }
        for hit in hits
    ]

if __name__ == "__main__":
    query = "What are the main challenges in biosecurity for direct selling farms?"
    results = retrieve_top_chunks(query)

    for i, r in enumerate(results, 1):
        print(f"\n--- Result {i} ---")
        print(f"Title: {r['title']}")
        print(f"Score: {r['score']:.2f}")
        print(r['chunk'][:500], "...")

