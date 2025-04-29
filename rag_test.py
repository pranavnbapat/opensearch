import os
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

def create_conversation_memory():
    payload = {
        "name": "Conversation with OpenAI GPT-3.5"
    }
    response = requests.post(
        f"{OPENSEARCH_URL}/_plugins/_ml/memory/",
        headers={"Content-Type": "application/json"},
        auth=HTTPBasicAuth(AUTH[0], AUTH[1]),
        json=payload
    )
    if response.status_code == 200:
        memory_id = response.json()["memory_id"]
        print(f"🧠 Created conversation memory: {memory_id}")
        return memory_id
    else:
        raise Exception(f"❌ Failed to create memory: {response.text}")

MEMORY_ID = create_conversation_memory()

while True:
    query_text = input("❓ Enter your question (or type 'exit' to quit): ").strip()
    if query_text.lower() == "exit":
        print("👋 Exiting conversation.")
        break

    search_payload = {
        "query": {
            "match": {
                "text": query_text
            }
        },
        "ext": {
            "generative_qa_parameters": {
                "llm_model": "gpt-3.5-turbo",
                "llm_question": query_text,
                "memory_id": MEMORY_ID,
                "context_size": 3,
                "message_size": 5,
                "timeout": 30
            }
        }
    }

    response = requests.post(
        f"{OPENSEARCH_URL}/{INDEX_NAME}/_search",
        headers={"Content-Type": "application/json"},
        auth=HTTPBasicAuth(AUTH[0], AUTH[1]),
        json=search_payload
    )

    if response.status_code == 200:
        result = response.json()

        retrieved_contexts = result.get("hits", {}).get("hits", [])
        combined_context = ""

        # Limit to top 2 results only (to avoid overloading the model)
        for hit in retrieved_contexts[:2]:  # change [:2] if you want more or less
            context_text = hit["_source"].get("text", "")
            combined_context += context_text + "\n\n"

        # Final safety net: Limit combined context to ~4000 characters
        if len(combined_context) > 4000:
            combined_context = combined_context[:4000]

        # Prepare manual prompt if no direct RAG answer
        if combined_context:
            print("\n🧠 Context found, sending to LLM...")
            print("\nContext snippet:\n", combined_context[:500], "...")  # Show part of context
        else:
            print("\n⚠️ No context retrieved from search!")

        # Now fetch the generated RAG answer
        answer = result.get("ext", {}).get("retrieval_augmented_generation", {}).get("answer", None)

        if answer:
            print("\n🧠 RAG Answer from Local Ollama:")
            print(answer)
        else:
            print(
                "\n⚠️ No RAG answer generated. Model may be overloaded, context may be too large, or tunnel may be offline.")
    else:
        print(f"❌ Query failed with status {response.status_code}: {response.text}")

