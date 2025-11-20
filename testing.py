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
from groq_api import get_embedding, ask_groq
from text_processor import create_chunks
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
            f"---\n🧭 [Link to find out more]({google_link})",
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
