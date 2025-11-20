# from fastapi import FastAPI
# from fastapi.middleware.cors import CORSMiddleware
# from pydantic import BaseModel
# from src.groq_api import get_embedding, ask_groq
# from src.text_processor import create_chunks
# from src.app import fallback_to_gemini, search_similar_chunks

# import json
# import faiss
# import os
# import re
# from urllib.parse import quote
# from pathlib import Path

# # -------------------------------------
# # 🚀 FastAPI Setup
# # -------------------------------------
# app = FastAPI(title="AI Chatbot Backend")

# app.add_middleware(
#     CORSMiddleware,
#     allow_origins=["*"],  # change to frontend URL in production
#     allow_credentials=True,
#     allow_methods=["*"],
#     allow_headers=["*"],
# )

# # -------------------------------------
# # 📦 Load FAISS Index + Metadata
# # -------------------------------------
# if not os.path.exists("faiss.index") or not os.path.exists("chunks_metadata.json"):
#     raise FileNotFoundError("FAISS index or chunks_metadata.json not found!")

# with open("chunks_metadata.json", "r", encoding="utf-8") as f:
#     chunks_with_metadata = json.load(f)

# index = faiss.read_index("faiss.index")

# # -------------------------------------
# # 🧠 Request Model
# # -------------------------------------
# class ChatRequest(BaseModel):
#     question: str
#     history: list = []

# # -------------------------------------
# # 💬 Chat Endpoint
# # -------------------------------------
# @app.post("/chat")
# async def chat(req: ChatRequest):
#     query_embedding = get_embedding(req.question)
#     results = search_similar_chunks(query_embedding)

#     # combine retrieved chunks as context
#     context = "\n\n".join([r["text"] for r in results]) if results else ""

#     # prepare Groq prompt
#     prompt = (
#         f"Answer the question below using only the provided context. "
#         f"If the answer cannot be found, say 'No relevant information found in the context.'\n\n"
#         f"Question: {req.question}\n\nContext:\n{context}"
#     )

#     # get Groq’s answer
#     response = ask_groq(prompt, req.question)
#     response_source = "Website Knowledge Base"

#     # 🔎 if FAISS returned nothing — directly fallback to Gemini
#     if not results:
#         print("⚠️ No FAISS matches — fallback to Gemini.")
#         gemini_text = fallback_to_gemini(req.question)
#         # try to extract link from Gemini response if present
#         match = re.search(r'(https?://[^\s)]+)', gemini_text)
#         gemini_link = match.group(1) if match else f"https://www.google.com/search?q={quote(req.question)}"
#         response = (
#             f"Answer :\n{gemini_text}\n\n"
#             f"---\n🧭Source: Google Gemini  \n🔗 [Find out more]({gemini_link})"
#         )
#         return {"answer": response}

#     # 🧩 if FAISS result exists but Groq answer is insufficient
#     if (
#         not response
#         or any(kw in response.lower() for kw in [
#             "not available", "not enough information", "no relevant", "i don't have enough"
#         ])
#     ):
#         print("⚠️ Insufficient FAISS answer — fallback to Gemini.")
#         gemini_text = fallback_to_gemini(req.question)
#         match = re.search(r'(https?://[^\s)]+)', gemini_text)
#         gemini_link = match.group(1) if match else f"https://www.google.com/search?q={quote(req.question)}"
#         response = (
#             f"Answer:\n{gemini_text}\n\n"
#             f"---\n🧭 Source: Google Gemini  \n{gemini_link}"
#         )
#         return {"answer": response}

#     # ✅ valid FAISS answer — attach search link for source file
#     top_chunk = results[0]
#     file_name = top_chunk.get("source", "")
#     filename_stem = Path(file_name).stem if file_name else quote(req.question)
#     nitr_link = f"https://www.google.com/search?q=site:nitrkl.ac.in+{quote(filename_stem)}"

#     final_response = (
#         f"Answer: \n{response}\n\n"
#         f"---\n🧭 Source: Website Knowledge Base  \n{nitr_link} "
#     )

#     return {"answer": final_response}

# # -------------------------------------
# # 🩵 Health Check
# # -------------------------------------
# @app.get("/")
# async def root():
#     return {"message": "AI Backend is running smoothly ✅"}


# from fastapi import FastAPI
# from fastapi.middleware.cors import CORSMiddleware
# from pydantic import BaseModel
# from src.groq_api import get_embedding, ask_groq
# from src.text_processor import create_chunks
# from src.app import fallback_to_gemini, search_similar_chunks

# import json
# import faiss
# import os
# import re
# from urllib.parse import quote, quote_plus
# from pathlib import Path

# # -------------------------------------
# # 🚀 FastAPI Setup
# # -------------------------------------
# app = FastAPI(title="AI Chatbot Backend")

# app.add_middleware(
#     CORSMiddleware,
#     allow_origins=["*"],  # change to frontend URL in production
#     allow_credentials=True,
#     allow_methods=["*"],
#     allow_headers=["*"],
# )

# # -------------------------------------
# # 📦 Load FAISS Index + Metadata
# # -------------------------------------
# if not os.path.exists("faiss.index") or not os.path.exists("chunks_metadata.json"):
#     raise FileNotFoundError("FAISS index or chunks_metadata.json not found!")

# with open("chunks_metadata.json", "r", encoding="utf-8") as f:
#     chunks_with_metadata = json.load(f)

# index = faiss.read_index("faiss.index")

# # -------------------------------------
# # 🧠 Request Model
# # -------------------------------------
# class ChatRequest(BaseModel):
#     question: str
#     history: list = []  # past chat messages (optional)

# # -------------------------------------
# # 💬 Chat Endpoint
# # -------------------------------------
# @app.post("/chat")
# async def chat(req: ChatRequest):
#     # 🧠 Add conversation context if history exists
#     context_text = ""
#     if req.history and len(req.history) > 0:
#         last_msgs = req.history[-6:]  # last few for context
#         context_text = "\n".join([
#             f"{m['sender'].capitalize()}: {m['text']}" for m in last_msgs
#         ])

#     # 💬 Handle name-intent instantly (no Groq/Gemini call)
#     q_lower = req.question.strip().lower()
#     if any(keyword in q_lower for keyword in ["your name", "who are you", "what is your name", "introduce yourself"]):
#         name_response = (
#             "👋 I'm Dristi — Digital Retrieval and Intelligent Search Technology Interface, "
#             "the official AI assistant for NIT Rourkela."
#         )
#         return {"answer": name_response}

#     # 🔍 Rephrase user query with context
#     contextual_prompt = (
#         f"Given the conversation below, rewrite the latest user question so it is complete and self-contained.\n"
#         f"Replace pronouns like 'he', 'she', or 'it' with their actual subjects.\n\n"
#         f"Conversation history:\n{context_text}\n\n"
#         f"Latest user question:\n{req.question}\n\n"
#         f"Return only the rephrased question:"
#     )

#     try:
#         contextual_question = ask_groq(contextual_prompt, req.question)
#         if not contextual_question or len(contextual_question.strip()) < 3:
#             contextual_question = req.question
#     except Exception:
#         contextual_question = req.question

#     query_embedding = get_embedding(contextual_question)
#     results = search_similar_chunks(query_embedding)

#     # combine retrieved chunks as context
#     context = "\n\n".join([r["text"] for r in results]) if results else ""

#     # 🧠 Updated Prompt for Fresh / Current Info
#     prompt = (
#         f"You are Drishti — Digital Retrieval and Intelligent Search Technology Interface for NIT Rourkela. "
#         f"Use the provided context as background info. "
#         f"If there’s a chance the data is outdated, clearly mention it and focus on the most recent or likely current situation.\n\n"
#         f"If the context doesn't contain enough info, say 'This information might be outdated — fetching latest updates.'\n\n"
#         f"Be clear, accurate, and concise.\n\n"
#         f"Conversation context:\n{context_text}\n\n"
#         f"Question: {contextual_question}\n\n"
#         f"Context (may contain old data):\n{context}"
#     )

#     # get Groq’s answer
#     response = ask_groq(prompt, contextual_question)
#     response_source = "Website Knowledge Base"

#     # 🔎 if FAISS returned nothing — directly fallback to Gemini
#     if not results:
#         print("⚠️ No FAISS matches — fallback to Gemini.")
#         gemini_text = fallback_to_gemini(contextual_question)
#         match = re.search(r'(https?://[^\s)]+)', gemini_text)
#         gemini_link = match.group(1) if match else f"https://www.google.com/search?q={quote_plus(contextual_question)}"
#         response = (
#             f"Answer:\n{gemini_text}\n\n"
#             f"---\n🧭 Source: Google Gemini  \n🔗 [Find out more]({gemini_link})"
#         )
#         return {"answer": response}

#     # 🧩 if FAISS result exists but Groq answer is insufficient
#     if (
#         not response
#         or any(kw in response.lower() for kw in [
#             "not available", "not enough information", "no relevant", "i don't have enough"
#         ])
#     ):
#         print("⚠️ Insufficient FAISS answer — fallback to Gemini.")
#         gemini_text = fallback_to_gemini(contextual_question)
#         match = re.search(r'(https?://[^\s)]+)', gemini_text)
#         gemini_link = match.group(1) if match else f"https://www.google.com/search?q={quote_plus(contextual_question)}"
#         response = (
#             f"Answer:\n{gemini_text}\n\n"
#             f"---\n🧭 Source: Google Gemini  \n{gemini_link}"
#         )
#         return {"answer": response}

#     # ✅ valid FAISS answer — attach NITR link
#     nitr_link = f"https://www.nitrkl.ac.in/Search/?q={quote_plus(contextual_question)}"

#     final_response = (
#         f"Answer:\n{response}\n\n"
#         f"---\n🧭 Source: Website Knowledge Base  \n{nitr_link}"
#     )

#     return {"answer": final_response}

# # -------------------------------------
# # 🩵 Health Check
# # -------------------------------------
# @app.get("/")
# async def root():
#     return {"message": "AI Backend is running smoothly ✅"}


from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from src.groq_api import get_embedding, ask_groq
from src.text_processor import create_chunks
from src.app import fallback_to_gemini, search_similar_chunks

import json
import faiss
import os
import re
import requests
from urllib.parse import quote_plus
from pathlib import Path

# -------------------------------------
# 🚀 FastAPI Setup
# -------------------------------------
app = FastAPI(title="AI Chatbot Backend")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # change to frontend URL in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -------------------------------------
# 📦 Load FAISS Index + Metadata
# -------------------------------------
if not os.path.exists("faiss.index") or not os.path.exists("chunks_metadata.json"):
    raise FileNotFoundError("FAISS index or chunks_metadata.json not found!")

with open("chunks_metadata.json", "r", encoding="utf-8") as f:
    chunks_with_metadata = json.load(f)

index = faiss.read_index("faiss.index")

# -------------------------------------
# 🔍 Google CSE-Based NITR Link Finder
# -------------------------------------
def find_first_nitrkl_link(query: str) -> str:
    """
    Use Google Custom Search JSON API to search 'site:nitrkl.ac.in <query>'
    and return the first nitrkl.ac.in link.
    If none found, return NITR homepage.
    """
    clean_query = query.replace("?", "").replace(".", "").strip()

    api_key = "AIzaSyB2nDikJtawfwBzm9ziXdq0PWSqGmCn3Xw"  # your CSE API key
    cse_id = "20d27d17fa71343dc"  # your CSE ID

    search_url = f"https://www.googleapis.com/customsearch/v1?key={api_key}&cx={cse_id}&q={quote_plus('site:nitrkl.ac.in ' + clean_query)}"
    print(f"[DEBUG] Query sent to Google CSE: site:nitrkl.ac.in {clean_query}")

    try:
        resp = requests.get(search_url, timeout=10)
        if resp.status_code != 200:
            print(f"[API Error] {resp.status_code}: {resp.text}")
            return "https://www.nitrkl.ac.in/"

        data = resp.json()
        items = data.get("items", [])
        for item in items:
            link = item.get("link", "")
            if link.startswith("https://") and "nitrkl.ac.in" in link:
                return link

    except Exception as e:
        print(f"[Search Error] {e}")

    return "https://www.nitrkl.ac.in/"

# -------------------------------------
# 🧠 Request Model
# -------------------------------------
class ChatRequest(BaseModel):
    question: str
    history: list = []  # past chat messages (optional)

# -------------------------------------
# 💬 Chat Endpoint
# -------------------------------------
@app.post("/chat")
async def chat(req: ChatRequest):
    # 🧠 Add conversation context if history exists
    context_text = ""
    if req.history and len(req.history) > 0:
        last_msgs = req.history[-6:]  # last few for context
        context_text = "\n".join([
            f"{m['sender'].capitalize()}: {m['text']}" for m in last_msgs
        ])

    # 💬 Handle name-intent instantly
    q_lower = req.question.strip().lower()
    if any(keyword in q_lower for keyword in ["your name", "who are you", "what is your name", "introduce yourself"]):
        name_response = (
            "👋 I'm Drishti — Digital Retrieval and Intelligent Search Technology Interface, "
            "the official AI assistant for NIT Rourkela."
        )
        return {"answer": name_response}

    # 🔍 Rephrase user query with context
    contextual_prompt = (
        f"Given the conversation below, rewrite the latest user question so it is complete and self-contained.\n"
        f"Replace pronouns like 'he', 'she', or 'it' with their actual subjects.\n\n"
        f"Conversation history:\n{context_text}\n\n"
        f"Latest user question:\n{req.question}\n\n"
        f"Return only the rephrased question:"
    )

    try:
        contextual_question = ask_groq(contextual_prompt, req.question)
        if not contextual_question or len(contextual_question.strip()) < 3:
            contextual_question = req.question
    except Exception:
        contextual_question = req.question

    query_embedding = get_embedding(contextual_question)
    results = search_similar_chunks(query_embedding)

    # combine retrieved chunks as context
    context = "\n\n".join([r["text"] for r in results]) if results else ""

    # 🧠 Updated Prompt for Fresh / Current Info
    prompt = (
        f"You are Drishti — Digital Retrieval and Intelligent Search Technology Interface for NIT Rourkela. "
        f"Use the provided context as background info. "
        f"If there’s a chance the data is outdated, clearly mention it and focus on the most recent or likely current situation.\n\n"
        f"If the context doesn't contain enough info, say 'This information might be outdated — fetching latest updates.'\n\n"
        f"Be clear, accurate, and concise.\n\n"
        f"Conversation context:\n{context_text}\n\n"
        f"Question: {contextual_question}\n\n"
        f"Context (may contain old data):\n{context}"
    )

    # get Groq’s answer
    response = ask_groq(prompt, contextual_question)
    response_source = "Website Knowledge Base"

    # 🔎 if FAISS returned nothing — directly fallback to Gemini
    if not results:
        print("⚠️ No FAISS matches — fallback to Gemini.")
        gemini_text = fallback_to_gemini(contextual_question)
        google_link = f"https://www.google.com/search?q={quote_plus(contextual_question)}"
        response = (
            f"Answer:\n{gemini_text}\n\n"
            f"---\n🧭 Source: Google Gemini  \n{google_link}"
        )
        return {"answer": response}

    # 🧩 if FAISS result exists but Groq answer is insufficient
    if (
        not response
        or any(kw in response.lower() for kw in [
            "not available", "not enough information", "no relevant", "i don't have enough"
        ])
    ):
        print("⚠️ Insufficient FAISS answer — fallback to Gemini.")
        gemini_text = fallback_to_gemini(contextual_question)
        google_link = f"https://www.google.com/search?q={quote_plus(contextual_question)}"
        response = (
            f"Answer:\n{gemini_text}\n\n"
            f"---\n🧭 Source: Google Gemini  \n{google_link}"
        )
        return {"answer": response}


    # ✅ valid FAISS answer — attach live NITR link
    nitr_link = find_first_nitrkl_link(contextual_question)

    final_response = (
        f"Answer:\n{response}\n\n"
        f"---\n🧭 Source: Website Knowledge Base  \n{nitr_link}"
    )

    return {"answer": final_response}

# -------------------------------------
# 🩵 Health Check
# -------------------------------------
@app.get("/")
async def root():
    return {"message": "AI Backend is running smoothly ✅"}

