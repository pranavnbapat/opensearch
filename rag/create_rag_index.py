# rag/create_rag_index.py

import sys

from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[1]))

from utils import client

index_name = "rag_paraphrase_multilingual_minilm_l12_v2"
dimension = 384

mapping = {
    "settings": {
        "index": {
            "knn": True,
            "knn.algo_param.ef_search": 100
        }
    },
    "mappings": {
        "properties": {
            "title": {"type": "text"},
            "summary": {"type": "text"},
            "content_chunk": {"type": "text"},
            "embedding": {
                "type": "knn_vector",
                "dimension": dimension,
                "method": {
                    "name": "hnsw",
                    "space_type": "cosinesimil",
                    "engine": "nmslib"
                }
            },
            "language": {"type": "keyword"},
            "locations": {"type": "keyword"},
            "projectName": {"type": "keyword"}
        }
    }
}

if not client.indices.exists(index=index_name):
    client.indices.create(index=index_name, body=mapping)
