import re

import requests
import streamlit as st
from youtube_transcript_api import (
    YouTubeTranscriptApi,
    TranscriptsDisabled,
    NoTranscriptFound,
)
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.prompts import PromptTemplate
from langchain_groq import ChatGroq

# =====================================================================
# 👇👇👇  PASTE YOUR GROQ API KEY BETWEEN THE QUOTES BELOW  👇👇👇
GROQ_API_KEY = ""
# 👆👆👆  Get one free at https://console.groq.com/keys  👆👆👆
# (If you leave it empty, the app will ask for the key in the sidebar.)
# =====================================================================

DEFAULT_GROQ_MODEL = "llama-3.3-70b-versatile"  # used only if the live model list cannot be fetched

st.set_page_config(page_title="YouTube RAG Chat", page_icon="🎬", layout="wide")

PROMPT = PromptTemplate.from_template(
    """You are a helpful assistant that answers questions using the provided transcript context.

Instructions:
- Answer the question directly using ONLY the context.
- Summarize and organize relevant information in your own words.
- Do not copy the entire transcript.
- If the context does not contain the answer, say "I don't know."
- Use a short introduction followed by a numbered list when appropriate.
- Explain technical terms clearly.
- Do not include unrelated information.

Transcript context:
{context}

Question: {question}

Answer:"""
)


# ---------- helpers ----------
def extract_video_id(text: str):
    """Accept a full YouTube URL (watch, youtu.be, shorts, embed) or a bare 11-char ID."""
    text = text.strip()
    if re.fullmatch(r"[\w-]{11}", text):
        return text
    match = re.search(r"(?:v=|youtu\.be/|embed/|shorts/|live/)([\w-]{11})", text)
    return match.group(1) if match else None


@st.cache_data(show_spinner=False)
def fetch_transcript(video_id: str, languages: tuple) -> str:
    fetched = YouTubeTranscriptApi().fetch(video_id, languages=list(languages))
    return " ".join(snippet.text for snippet in fetched)


@st.cache_resource(show_spinner="Loading embedding model...")
def load_embeddings():
    return HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")


@st.cache_resource(show_spinner=False)
def build_index(transcript: str, chunk_size: int, chunk_overlap: int):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, chunk_overlap=chunk_overlap
    )
    docs = splitter.create_documents([transcript])
    return FAISS.from_documents(docs, load_embeddings()), len(docs)


@st.cache_data(ttl=600, show_spinner=False)
def list_groq_models(api_key: str):
    """Ask Groq which chat models your key can actually use."""
    r = requests.get(
        "https://api.groq.com/openai/v1/models",
        headers={"Authorization": f"Bearer {api_key}"},
        timeout=10,
    )
    r.raise_for_status()
    ids = [m["id"] for m in r.json()["data"]]
    skip = ("whisper", "tts", "guard", "orpheus", "playai", "embed")
    return sorted(i for i in ids if not any(x in i for x in skip))


def get_model(api_key: str, model_name: str):
    return ChatGroq(
        model=model_name,
        api_key=api_key,
        temperature=0.2,
        max_tokens=400,
    )


# ---------- session state ----------
for key, default in [("store", None), ("video_id", None), ("messages", []),
                     ("transcript", ""), ("n_chunks", 0)]:
    st.session_state.setdefault(key, default)


# ---------- sidebar ----------
with st.sidebar:
    st.header("⚙️ Settings")
    url = st.text_input("YouTube URL or video ID", placeholder="https://youtu.be/ugppjNn8uIE")
    languages = st.text_input("Caption language(s)", value="en", help="Comma-separated, e.g. en,hi")

    api_key = GROQ_API_KEY.strip()
    if api_key:
        st.success("Groq API key loaded from code ✅")
    else:
        api_key = st.text_input("Groq API key", type="password")
    try:
        available = list_groq_models(api_key) if api_key else []
    except Exception:
        available = []
    if available:
        prefs = ["llama-3.3-70b-versatile", "llama-3.1-8b-instant", "openai/gpt-oss-20b"]
        default = next((m for m in prefs if m in available), available[0])
        model_name = st.selectbox("Groq model", available, index=available.index(default))
    else:
        model_name = st.text_input("Groq model", value=DEFAULT_GROQ_MODEL)

    with st.expander("Advanced"):
        chunk_size = st.slider("Chunk size", 300, 2000, 1000, 100)
        chunk_overlap = st.slider("Chunk overlap", 0, 400, 200, 50)
        top_k = st.slider("Chunks to retrieve (k)", 1, 6, 3)

    load_clicked = st.button("Load video", type="primary", use_container_width=True)

    if st.session_state.video_id:
        st.divider()
        st.video(f"https://www.youtube.com/watch?v={st.session_state.video_id}")
        st.caption(f"{st.session_state.n_chunks} chunks indexed")
        if st.button("Clear chat", use_container_width=True):
            st.session_state.messages = []
            st.rerun()


# ---------- load video ----------
if load_clicked:
    video_id = extract_video_id(url)
    langs = tuple(l.strip() for l in languages.split(",") if l.strip()) or ("en",)
    if not video_id:
        st.sidebar.error("Couldn't find a video ID in that URL.")
    else:
        try:
            with st.spinner("Fetching transcript..."):
                transcript = fetch_transcript(video_id, langs)
            with st.spinner("Building search index..."):
                store, n_chunks = build_index(transcript, chunk_size, chunk_overlap)
            st.session_state.update(
                video_id=video_id, store=store, transcript=transcript,
                n_chunks=n_chunks, messages=[],
            )
            st.rerun()
        except TranscriptsDisabled:
            st.sidebar.error("Captions are disabled for this video.")
        except NoTranscriptFound:
            st.sidebar.error(f"No transcript found for language(s): {', '.join(langs)}.")
        except Exception as e:
            st.sidebar.error(f"Could not load video: {e}")


# ---------- main chat ----------
st.title("🎬 Chat with a YouTube video")

if st.session_state.store is None:
    st.info("Paste a YouTube link in the sidebar and click **Load video** to get started.")
    st.stop()

with st.expander("View transcript"):
    st.write(st.session_state.transcript)

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg.get("sources"):
            with st.expander("Sources used"):
                for i, src in enumerate(msg["sources"], 1):
                    st.markdown(f"**Chunk {i}**")
                    st.caption(src)

if question := st.chat_input("Ask something about the video..."):
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        if not api_key:
            st.error("Paste your Groq API key in the code (GROQ_API_KEY) or in the sidebar.")
            st.stop()
        try:
            retriever = st.session_state.store.as_retriever(
                search_type="similarity", search_kwargs={"k": top_k}
            )
            docs = retriever.invoke(question)
            context = "\n\n".join(d.page_content for d in docs)
            prompt = PROMPT.format(context=context, question=question)
            model = get_model(api_key, model_name)

            def token_stream():
                for chunk in model.stream(prompt):
                    yield chunk.content

            answer = st.write_stream(token_stream())  # streams and returns full text
        except Exception as e:
            st.error(f"Something went wrong: {e}")
            st.stop()

        sources = [d.page_content for d in docs]
        with st.expander("Sources used"):
            for i, src in enumerate(sources, 1):
                st.markdown(f"**Chunk {i}**")
                st.caption(src)

    st.session_state.messages.append(
        {"role": "assistant", "content": answer, "sources": sources}
    )