# rag/prompt_builder.py

from typing import List, Dict


def build_prompt(chunks: List[Dict], user_query: str, instructions: str = None) -> str:
    """
    Construct a RAG-style prompt from retrieved chunks and a user query.

    :param chunks: List of dicts with at least the 'chunk' field (from retriever)
    :param user_query: The user's natural language question
    :param instructions: (Optional) Custom instruction header
    :return: Formatted prompt string
    """

    # Default instructions if none provided
    if not instructions:
        instructions = (
            "You are a helpful and articulate assistant. Use only the context provided below to answer the user's question. "
            "Your answer should be natural, clear, and well-structured. Use bullet points only when absolutely needed to clarify information. "
            "If you are unsure or the answer is not in the context, respond with: "
            "\"I don’t know based on the provided context.\""
        )

    # Combine chunks into context sections
    context = "\n\n".join([
        f"---\n{c['chunk']}" for c in chunks if c.get("chunk")
    ])

    # Final prompt
    prompt = (
        f"{instructions}\n\n"
        f"Context:\n{context}\n\n"
        f"Question:\n{user_query.strip()}\n\n"
        f"Answer:"
    )

    return prompt
