# import os
# from pathlib import Path
# os.environ["STREAMLIT_FILE_WATCHER_TYPE"] = "none"  # Prevent torch.classes bug in Streamlit

# import streamlit as st
# import glob
# import faiss
# import numpy as np
# import json
# import tempfile
# from gtts import gTTS
# from streamlit_mic_recorder import speech_to_text
# from src.groq_api import get_embedding, ask_groq
# from src.text_processor import create_chunks
# from tqdm import tqdm

# from google import genai

# os.environ["GOOGLE_API_KEY"] = "AIzaSyBOdcqq-BgGF2QmIbmLOY_zzIs9YAqRFP0"
# gemini_client = genai.Client(api_key=os.environ["GOOGLE_API_KEY"])

# st.title("RP Chatbot")

# # Project root (two levels up from this file: project_root/src/app.py -> project_root)
# ROOT_DIR = Path(__file__).resolve().parent.parent
# print(f"Project root resolved to: {ROOT_DIR}")

# def get_file_mods():
#     # Build paths relative to the repository root so Streamlit's cwd won't matter
#     base_dir = str(ROOT_DIR / 'set1_text')
#     file_paths = glob.glob(os.path.join(base_dir, '*.txt'))
#     # Return mtime and size so we can detect real content changes vs environment differences
#     return tuple((fp, (os.path.getmtime(fp), os.path.getsize(fp))) for fp in sorted(file_paths))


# def load_and_embed_files(file_mods):
#     print("Loading and embedding files... (only on first run)")

#     base_dir = str(ROOT_DIR / 'set1_text')
#     files = []
#     for file_path in glob.glob(os.path.join(base_dir, '*.txt')):
#         with open(file_path, 'r', encoding='utf-8') as f:
#             content = f.read()
#             files.append({
#                 'path': file_path,
#                 'content': content
#             })

#     chunks_with_metadata = []
#     embeddings = []

#     for file in tqdm(files, desc="Processing files"):
#         print(f"Embedding file: {file['path']}")  # Display the filename being embedded
#         try:
#             # Use overlapping chunking
#             chunks = create_chunks(file['content'])

#             # Store chunks with metadata
#             for i, chunk in enumerate(chunks):
#                 chunks_with_metadata.append({
#                     'text': chunk,
#                     'source': file['path'],
#                     'chunk_index': i
#                 })

#             # Get embeddings for each chunk with progress
#             for chunk in tqdm(chunks, desc=f"Embedding chunks for {file['path']}"):
#                 embeddings.append(get_embedding(chunk))

#         except Exception as e:
#             print(f"Error processing file {file['path']}: {str(e)}")

#     embeddings_array = np.vstack(embeddings)
#     return chunks_with_metadata, embeddings_array


# def save_data(chunks_with_metadata, embeddings_array, index):
#     # Save metadata
#     chunks_path = str(ROOT_DIR / "chunks_metadata.json")
#     with open(chunks_path, "w", encoding="utf-8") as f:
#         json.dump(chunks_with_metadata, f, ensure_ascii=False, indent=2)
#     # Save embeddings
#     np.save(str(ROOT_DIR / "embeddings.npy"), embeddings_array)
#     # Save FAISS index
#     faiss.write_index(index, str(ROOT_DIR / "faiss.index"))
#     print("Data saved successfully.")


# def load_and_embed_files_incremental(new_files):
#     """
#     Only embed new or changed files, return their chunks and embeddings.
#     """
#     chunks_with_metadata = []
#     embeddings = []
#     for file_path in new_files:
#         print(f"\n🔹 Embedding new/changed file: {file_path}")  # <-- added print
#         with open(file_path, 'r', encoding='utf-8') as f:
#             content = f.read()
#             # Split content into chunks
#             chunks = [content[i:i+1000] for i in range(0, len(content), 1000)]
#             for i, chunk in enumerate(chunks):
#                 chunks_with_metadata.append({
#                     'text': chunk,
#                     'source': file_path,
#                     'chunk_index': i
#                 })
#                 embeddings.append(get_embedding(chunk))
#                 print(f"  └─ Embedded chunk {i+1}/{len(chunks)} from {os.path.basename(file_path)}")  # <-- added print
#     if embeddings:
#         embeddings_array = np.vstack(embeddings)
#     else:
#         embeddings_array = np.zeros((0, 384))  # fallback shape if no embeddings
#     return chunks_with_metadata, embeddings_array


# def load_data():
#     mods_path = str(ROOT_DIR / "file_mods.json")
#     file_mods = get_file_mods()
#     current_files = dict(file_mods)
#     # Load previous state if exists
#     chunks_path = str(ROOT_DIR / "chunks_metadata.json")
#     embeds_path = str(ROOT_DIR / "embeddings.npy")
#     faiss_path = str(ROOT_DIR / "faiss.index")

#     print(f"Looking for cached files:\n - {chunks_path}: {os.path.exists(chunks_path)}\n - {embeds_path}: {os.path.exists(embeds_path)}\n - {faiss_path}: {os.path.exists(faiss_path)}\n - {mods_path}: {os.path.exists(mods_path)}")

#     if (
#         os.path.exists(chunks_path)
#         and os.path.exists(embeds_path)
#         and os.path.exists(faiss_path)
#         and os.path.exists(mods_path)
#     ):
#         print("Checking for new or changed files...")
#         with open(mods_path, "r", encoding="utf-8") as f:
#             saved_mods = dict(json.load(f))
#         with open(chunks_path, "r", encoding="utf-8") as f:
#             chunks_with_metadata = json.load(f)
#         embeddings_array = np.load(embeds_path)
#         index = faiss.read_index(faiss_path)
#         # Find new or changed files. saved_mods may be in older formats (mtime only) or new format (mtime,size)
#         def file_changed(fp, current_val):
#             saved = saved_mods.get(fp)
#             if saved is None:
#                 return True
#             # saved could be a list [mtime, size] (from json) or a single mtime value
#             if isinstance(saved, list) and len(saved) == 2:
#                 try:
#                     saved_tuple = (float(saved[0]), int(saved[1]))
#                 except Exception:
#                     return True
#                 return saved_tuple != current_val
#             else:
#                 # fallback: compare only mtime
#                 try:
#                     return float(saved) != float(current_val[0])
#                 except Exception:
#                     return True

#         new_or_changed = [fp for fp, cur in current_files.items() if file_changed(fp, cur)]
#         if new_or_changed:
#             print(f"Embedding {len(new_or_changed)} new/changed files:")
#             print(new_or_changed)
#             new_chunks, new_embeds = load_and_embed_files_incremental(new_or_changed)
#             # Append new data
#             chunks_with_metadata.extend(new_chunks)
#             if len(new_embeds) > 0:
#                 embeddings_array = np.vstack([embeddings_array, new_embeds])
#                 index.add(new_embeds)
#             # Save updated data
#             with open(chunks_path, "w", encoding="utf-8") as f:
#                 json.dump(chunks_with_metadata, f, ensure_ascii=False, indent=2)
#             np.save(embeds_path, embeddings_array)
#             faiss.write_index(index, faiss_path)
#             with open(mods_path, "w", encoding="utf-8") as f:
#                 json.dump(list(current_files.items()), f, ensure_ascii=False, indent=2)
#         else:
#             print("No new or changed files. Using cached embeddings.")
#         return chunks_with_metadata, embeddings_array, index
#     else:
#         print("No saved data found. Computing embeddings for all files...")
#         all_files = list(current_files.keys())
#         chunks_with_metadata, embeddings_array = load_and_embed_files_incremental(all_files)
#         index = create_faiss_index(embeddings_array)
#         with open(chunks_path, "w", encoding="utf-8") as f:
#             json.dump(chunks_with_metadata, f, ensure_ascii=False, indent=2)
#         np.save(embeds_path, embeddings_array)
#         faiss.write_index(index, faiss_path)
#         with open(mods_path, "w", encoding="utf-8") as f:
#             json.dump(list(current_files.items()), f, ensure_ascii=False, indent=2)
#         return chunks_with_metadata, embeddings_array, index


# def create_faiss_index(_embeddings_array):
#     dimension = _embeddings_array.shape[1]
#     # Switch to IndexFlatIP for cosine similarity
#     index = faiss.IndexFlatIP(dimension)
#     index.add(_embeddings_array)
#     return index


# def rerank_candidates(candidates, query_embedding):
#     """Rerank candidates based on additional criteria."""
#     # Example: prioritize shorter distances and higher text relevance
#     for candidate in candidates:
#         candidate['rerank_score'] = -candidate['distance']  # Lower distance is better

#     # Sort by rerank_score (descending)
#     candidates.sort(key=lambda x: x['rerank_score'], reverse=True)
#     return candidates

# # Update search_similar_chunks to include reranking
# def search_similar_chunks(query_embedding, k=5):
#     # perform search using the session_state index
#     idx_obj = st.session_state.get("index")
#     if idx_obj is None:
#         raise RuntimeError("Index not initialized in session_state")
#     print("Index dim:", idx_obj.d)
#     print("Query embedding shape:", np.array(query_embedding).shape)

#     D, I = idx_obj.search(np.array([query_embedding]), k)
#     results = []
#     chunks = st.session_state.get("chunks_with_metadata", [])
#     for hit_idx, distance in zip(I[0], D[0]):
#         if hit_idx < len(chunks):
#             chunk = chunks[hit_idx].copy()
#             chunk['distance'] = float(distance)
#             results.append(chunk)

#     # Rerank the top candidates
#     results = rerank_candidates(results, query_embedding)
#     return results


# def fallback_to_gemini(query):
#     """
#     Uses Gemini to answer the question, restricting answers to NIT Rourkela context only.
#     """
#     try:
#         prompt = (
#             f"Answer this question accurately and concisely, but only using information relevant to "
#             f"National Institute of Technology Rourkela (NIT Rourkela):\n\n{query}"
#         )

#         response = gemini_client.models.generate_content(
#             model="gemini-2.5-flash",
#             contents=prompt
#         )
#         return response.text
#     except Exception as e:
#         print(f"[Gemini Error] {e}")
#         return "Sorry, I couldn't fetch real-time data right now."



# from urllib.parse import quote
# import re

# def process_question(question):
#     query_embedding = get_embedding(question)
#     results = search_similar_chunks(query_embedding)

#     # prepare context from FAISS chunks
#     context = "\n\n".join([chunk['text'] for chunk in results]) if results else ""

#     # chat history context
#     history = st.session_state.get("messages", [])
#     history_context = "\n".join([f"{msg['role'].capitalize()}: {msg['content']}" for msg in history[-10:]])

#     # main prompt for Groq
#     prompt = (
#         f"You are a helpful assistant. Here's recent conversation history:\n{history_context}\n\n"
#         f"Use only the provided context to answer accurately.\n"
#         f"If not enough info is available, say: 'No relevant information found in the context.'\n\n"
#         f"Question: {question}\n\nContext:\n{context}"
#     )

#     response = ask_groq(prompt, question)

#     # --- If FAISS has no results, go straight to Gemini ---
#     if not results:
#         print("⚠️ No FAISS results — fallback to Gemini.")
#         gemini_text, gemini_link = fallback_to_gemini(question), None
#         source_link = gemini_link or f"https://www.google.com/search?q={quote(question)}"
#         return (
#             f"**[Answer from Google Gemini]**\n{gemini_text}\n\n"
#             f"---\n🧭 **Source:** Google Gemini  \n🔗 [Find out more]({source_link})",
#             ""
#         )

#     # --- If FAISS response is insufficient ---
#     if (
#         not response
#         or any(kw in response.lower() for kw in [
#             "no relevant", "not available", "not enough", "i don't have enough"
#         ])
#     ):
#         print("⚠️ FAISS insufficient — switching to Gemini.")
#         gemini_text, gemini_link = fallback_to_gemini(question), None
#         source_link = gemini_link or f"https://www.google.com/search?q={quote(question)}"
#         return (
#             f"**[Answer from Google Gemini]**\n{gemini_text}\n\n"
#             f"---\n🧭 **Source:** Google Gemini  \n🔗 [Find out more]({source_link})",
#             ""
#         )

#     # --- If FAISS gives a valid response, link to source file on Google search ---
#     top_chunk = results[0]
#     file_name = top_chunk.get("source", "")
#     filename_stem = Path(file_name).stem if file_name else quote(question)
#     nitr_link = f"https://www.google.com/search?q=site:nitrkl.ac.in+{quote(filename_stem)}"

#     response_final = (
#         f"**Answer (from Website Knowledge Base):**\n{response}\n\n"
#         f"---\n🧭 **Source:** Website Knowledge Base  \n🔗 [Find out more]({nitr_link})"
#     )
#     return response_final, context





# # --- Main code ---
# # Initialize data once per Streamlit session to avoid repeated embedding/checks on reruns
# if "data_initialized" not in st.session_state:
#     print("Initializing data (calling load_data) and storing in session_state...")
#     chunks_with_metadata, embeddings_array, index = load_data()
#     st.session_state["chunks_with_metadata"] = chunks_with_metadata
#     st.session_state["embeddings_array"] = embeddings_array
#     st.session_state["index"] = index
#     st.session_state["data_initialized"] = True
# else:
#     chunks_with_metadata = st.session_state["chunks_with_metadata"]
#     embeddings_array = st.session_state["embeddings_array"]
#     index = st.session_state["index"]

# # Chat interface
# st.markdown("### Chat")

# if "messages" not in st.session_state:
#     st.session_state.messages = []

# # Display chat history
# for message in st.session_state.messages:
#     with st.chat_message(message["role"]):
#         st.markdown(message["content"])

# # --- New input block with mic + text ---
# st.write("#### Enter your query (type or speak)")

# # Option 1: Type
# user_input = st.chat_input("Type your query here")

# # Option 2: Speak
# st.write("Or click below to speak:")
# voice_query = speech_to_text(language='en', use_container_width=True, just_once=True, key="voice")

# # Pick whichever input is given
# prompt = user_input or voice_query

# if prompt:
#     st.session_state.messages.append({"role": "user", "content": prompt})
#     with st.chat_message("user"):
#         st.markdown(prompt)

#     with st.chat_message("assistant"):
#         message_placeholder = st.empty()
#         response, context = process_question(prompt)
#         message_placeholder.markdown(response)
#         st.session_state.messages.append({"role": "assistant", "content": response})

#         # Convert response to speech and play it
#         try:
#             tts = gTTS(response)
#             with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as tmp_file:
#                 tts.save(tmp_file.name)
#                 st.audio(tmp_file.name, format="audio/mp3")
#         except Exception as e:
#             st.error(f"Voice output failed: {e}")



import os
from pathlib import Path
os.environ["STREAMLIT_FILE_WATCHER_TYPE"] = "none"

import streamlit as st
import glob
import faiss
import numpy as np
import json
import tempfile
from urllib.parse import quote_plus
from gtts import gTTS
from streamlit_mic_recorder import speech_to_text
from src.groq_api import get_embedding, ask_groq
from src.text_processor import create_chunks
from tqdm import tqdm
from google import genai
import re

# ------------------- Gemini Setup -------------------
os.environ["GOOGLE_API_KEY"] = "AIzaSyBOdcqq-BgGF2QmIbmLOY_zzIs9YAqRFP0"
gemini_client = genai.Client(api_key=os.environ["GOOGLE_API_KEY"])

st.title("RP Chatbot")

ROOT_DIR = Path(__file__).resolve().parent.parent

# ============================================================
#                    FILE & INDEX HELPERS
# ============================================================

def get_file_mods():
    base_dir = str(ROOT_DIR / 'set1_text')
    file_paths = glob.glob(os.path.join(base_dir, '*.txt'))
    return tuple((fp, (os.path.getmtime(fp), os.path.getsize(fp))) for fp in sorted(file_paths))


def load_and_embed_files_incremental(new_files):
    chunks_with_metadata = []
    embeddings = []
    for file_path in new_files:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
            chunks = [content[i:i + 1000] for i in range(0, len(content), 1000)]
            file_name = Path(file_path).name
            for i, chunk in enumerate(chunks):
                chunks_with_metadata.append({
                    'text': chunk,
                    'source': file_name,
                    'chunk_index': i
                })
                embeddings.append(get_embedding(chunk))
    embeddings_array = np.vstack(embeddings) if embeddings else np.zeros((0, 384))
    return chunks_with_metadata, embeddings_array


def create_faiss_index(_embeddings_array):
    dimension = _embeddings_array.shape[1]
    index = faiss.IndexFlatIP(dimension)
    if _embeddings_array.shape[0] > 0:
        index.add(_embeddings_array)
    return index


def load_data():
    mods_path = str(ROOT_DIR / "file_mods.json")
    file_mods = get_file_mods()
    current_files = dict(file_mods)

    chunks_path = str(ROOT_DIR / "chunks_metadata.json")
    embeds_path = str(ROOT_DIR / "embeddings.npy")
    faiss_path = str(ROOT_DIR / "faiss.index")

    if (
        os.path.exists(chunks_path)
        and os.path.exists(embeds_path)
        and os.path.exists(faiss_path)
        and os.path.exists(mods_path)
    ):
        with open(chunks_path, "r", encoding="utf-8") as f:
            chunks_with_metadata = json.load(f)
        embeddings_array = np.load(embeds_path)
        index = faiss.read_index(faiss_path)
        return chunks_with_metadata, embeddings_array, index
    else:
        all_files = list(current_files.keys())
        chunks_with_metadata, embeddings_array = load_and_embed_files_incremental(all_files)
        index = create_faiss_index(embeddings_array)
        with open(chunks_path, "w", encoding="utf-8") as f:
            json.dump(chunks_with_metadata, f, ensure_ascii=False, indent=2)
        np.save(embeds_path, embeddings_array)
        faiss.write_index(index, faiss_path)
        with open(mods_path, "w", encoding="utf-8") as f:
            json.dump(list(current_files.items()), f, ensure_ascii=False, indent=2)
        return chunks_with_metadata, embeddings_array, index


def rerank_candidates(candidates, query_embedding):
    for candidate in candidates:
        candidate['rerank_score'] = -candidate['distance']
    candidates.sort(key=lambda x: x['rerank_score'], reverse=True)
    return candidates


def search_similar_chunks(query_embedding, k=5):
    idx_obj = st.session_state.get("index")
    if idx_obj is None:
        raise RuntimeError("Index not initialized in session_state")
    D, I = idx_obj.search(np.array([query_embedding]), k)
    results = []
    chunks = st.session_state.get("chunks_with_metadata", [])
    for hit_idx, distance in zip(I[0], D[0]):
        if 0 <= hit_idx < len(chunks):
            chunk = chunks[hit_idx].copy()
            chunk['distance'] = float(distance)
            results.append(chunk)
    results = rerank_candidates(results, query_embedding)
    return results


# ============================================================
#                    GEMINI FALLBACK
# ============================================================

def fallback_to_gemini(query):
    """Fetch response from Gemini and remove any fake or unwanted URLs."""
    try:
        prompt = (
            f"Answer this question accurately about NIT Rourkela (nitrkl.ac.in). "
            f"Provide a factual answer without adding fake or outdated links.\n\nQuestion:\n{query}"
        )
        response = gemini_client.models.generate_content(
            model="gemini-2.5-flash",
            contents=prompt
        )
        text = response.text.strip()
        # Remove any URLs Gemini may include
        text = re.sub(r'https?://\S+', '', text)
        return text
    except Exception as e:
        print(f"[Gemini Error] {e}")
        return "Sorry, I couldn't fetch real-time data right now."


# ============================================================
#                    QUESTION PROCESSOR
# ============================================================

INSUFFICIENT_PATTERNS = [
    "not available",
    "no relevant",
    "not enough",
    "i don't have enough",
    "no relevant information",
    "cannot be found",
    "no information"
]


def faiss_response_is_insufficient(text: str) -> bool:
    if not text:
        return True
    low = text.lower()
    return any(pat in low for pat in INSUFFICIENT_PATTERNS)


def process_question(question):
    # 🧠 Maintain conversation context
    context_text = ""
    if "messages" in st.session_state and len(st.session_state["messages"]) > 0:
        history = [
            f"{m['role'].capitalize()}: {m['content']}"
            for m in st.session_state["messages"][-6:]
        ]
        context_text = "\n".join(history)

    # 🧩 Rephrase question clearly
    contextual_question_prompt = (
        f"Given the conversation below, rewrite the latest question so it is complete and self-contained.\n"
        f"Replace pronouns like 'he', 'she', or 'it' with their actual subjects.\n\n"
        f"Conversation history:\n{context_text}\n\n"
        f"Latest user question:\n{question}\n\n"
        f"Return only the rephrased, clear question:"
    )

    try:
        contextual_question = ask_groq(contextual_question_prompt, question)
        if not contextual_question or len(contextual_question.strip()) < 3:
            contextual_question = question
    except Exception:
        contextual_question = question

    query_embedding = get_embedding(contextual_question)
    results = search_similar_chunks(query_embedding)

    # --- If no FAISS results, Gemini fallback ---
    if not results:
        gemini_text = fallback_to_gemini(contextual_question)
        google_link = f"https://www.google.com/search?q={quote_plus(contextual_question)}"
        return (
            f"**Rephrased question for clarity:** {contextual_question}\n\n"
            f"**Answer (from Google Gemini):**\n{gemini_text.strip()}\n\n"
            f"---\n🧭 ({google_link})",
            ""
        )

    # --- Use FAISS Knowledge Base ---
    context = "\n\n".join([chunk['text'] for chunk in results])
    prompt = (
        f"Answer the following question using only the provided context. "
        f"If the answer cannot be found, say exactly: 'No relevant information found in the context.'\n\n"
        f"Question: {contextual_question}\n\nContext:\n{context}"
    )
    faiss_answer = ask_groq(prompt, contextual_question)

    if faiss_response_is_insufficient(faiss_answer):
        gemini_text = fallback_to_gemini(contextual_question)
        google_link = f"https://www.google.com/search?q={quote_plus(contextual_question)}"
        return (
            f"**Rephrased question for clarity:** {contextual_question}\n\n"
            f"**Answer (from Google Gemini):**\n{gemini_text.strip()}\n\n"
            f"---\n🧭 [Link to find out more]({google_link})",
            ""
        )

    nitr_link = f"https://www.nitrkl.ac.in/Search/?q={quote_plus(contextual_question)}"
    return (
        f"**Rephrased question for clarity:** {contextual_question}\n\n"
        f"**Answer (from Website Knowledge Base):**\n{faiss_answer.strip()}\n\n"
        f"---\n🧭 [Link to find out more]({nitr_link})",
        ""
    )


# ============================================================
#                    STREAMLIT UI
# ============================================================

if "data_initialized" not in st.session_state:
    chunks_with_metadata, embeddings_array, index = load_data()
    st.session_state["chunks_with_metadata"] = chunks_with_metadata
    st.session_state["embeddings_array"] = embeddings_array
    st.session_state["index"] = index
    st.session_state["data_initialized"] = True

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

st.markdown("### Chat")
st.write("#### Enter your query (type or speak)")
user_input = st.chat_input("Type your query here")
st.write("Or click below to speak:")
voice_query = speech_to_text(language='en', use_container_width=True, just_once=True, key="voice")

prompt = user_input or voice_query

if prompt:
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        response, _ = process_question(prompt)
        message_placeholder.markdown(response)
        st.session_state.messages.append({"role": "assistant", "content": response})

        try:
            tts = gTTS(response)
            with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as tmp_file:
                tts.save(tmp_file.name)
                st.audio(tmp_file.name, format="audio/mp3")
        except Exception as e:
            st.error(f"Voice output failed: {e}")
