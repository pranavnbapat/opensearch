from retriever import retrieve_top_chunks
from prompt_builder import build_prompt

query = "What are the common biosecurity issues in direct selling farms?"
results = retrieve_top_chunks(query)
prompt = build_prompt(results, query)

print(prompt)
