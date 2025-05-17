import requests

from prompt_builder import build_prompt
from retriever import retrieve_top_chunks

query = "Tell me something about crop rotation in EU."
retrieved_chunks = retrieve_top_chunks(query)

prompt = build_prompt(retrieved_chunks, query)

def query_mistral_ollama(prompt: str) -> str:
    response = requests.post(
        "https://v9zpf5kzgkvf1r-11434.proxy.runpod.net/api/generate",
        json={
            "model": "mistral",
            "prompt": prompt,
            "stream": False
        }
    )
    if response.status_code != 200:
        raise Exception(f"Ollama error {response.status_code}: {response.text}")
    return response.json()["response"]

response = query_mistral_ollama(prompt)

seen = set()
sources = []
for chunk in retrieved_chunks:
    link = chunk.get("id")
    title = chunk.get("title", "View source")
    if link and link not in seen:
        seen.add(link)
        sources.append(f"- [{title}]({link})")
    if len(sources) == 5:
        break

print("\n🔍 Answer:\n", response)

if sources:
    print("\n📚 You may find these resources useful:")
    print("\n".join(sources))
