import os
import requests
import numpy as np
from sentence_transformers import SentenceTransformer

# Switch to a stronger embedding model for better accuracy
embedder = SentenceTransformer("all-mpnet-base-v2")

def get_groq_api_key():
    # Try environment variable first
    key = os.environ.get("GROQ_API_KEY")
    if key:
        return key
    # Try Streamlit secrets if available
    try:
        import streamlit as st
        return st.secrets["GROQ_API_KEY"]
    except (ImportError, AttributeError, KeyError):
        raise RuntimeError("GROQ_API_KEY not found in environment or Streamlit secrets.")

def get_embedding(text):
    embedding = embedder.encode([text])[0]
    # Normalize the embedding to a unit vector
    return embedding / np.linalg.norm(embedding)

def ask_groq(context, question):
    api_key = get_groq_api_key()
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    # Truncate context to 4000 characters to avoid payload too large error
    max_context_length = 4000
    safe_context = context[:max_context_length]
    data = {
        "model": "llama-3.1-8b-instant",
        "messages": [
            {
                "role": "system",
                "content": (
                    "You are an assistant that answers strictly using the provided context.\n"
                    "- If the answer is in the context, use only that information.\n"
                    "- If the context does not contain the answer, reply with: "
                    "\"The answer is not available in the provided text.\"\n"
                    "- Do not use outside knowledge, do not make assumptions, and do not hallucinate.\n"
                    "- Keep answers clear and elaborate."
                ),
            },
            {
                "role": "user",
                "content": f"Context:\n{safe_context}\n\nQuestion: {question}"
            }
        ]
    }

    response = requests.post("https://api.groq.com/openai/v1/chat/completions", json=data, headers=headers)
    response.raise_for_status()
    return response.json()["choices"][0]["message"]["content"]