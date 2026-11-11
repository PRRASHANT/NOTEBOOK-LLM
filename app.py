"""Streamlit interface for document retrieval and grounded question answering."""

import os
from hashlib import sha256

import streamlit as st
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from pypdf.errors import PdfReadError

from answer_utils import answer_question
from document_utils import index_document
from embedding_utils import MODEL_NAME
from pdf_utils import extract_pdf_text
from retrieval_utils import build_retrieval_query
from search_utils import search_chunks


@st.cache_resource(show_spinner=False)
def get_embedding_model():
    """Reuse the embedding model across Streamlit reruns."""
    return HuggingFaceEmbeddings(model_name=MODEL_NAME)


@st.cache_resource(show_spinner=False)
def get_answer_model(api_key: str):
    """Reuse the Groq chat model without storing credentials in source code."""
    return ChatGroq(model="openai/gpt-oss-20b", api_key=api_key, temperature=0)


def get_api_key():
    """Read credentials from the environment or Streamlit secrets."""
    key = os.environ.get("GROQ_API_KEY")
    if key:
        return key
    try:
        return st.secrets.get("GROQ_API_KEY")
    except Exception:
        # Streamlit may raise when no secrets file is configured.
        return None


st.set_page_config(page_title="Notebook LLM", page_icon="📚")
st.title("📚 Notebook LLM")
st.write("Upload a PDF and ask questions about its contents.")

uploaded_file = st.file_uploader("Choose a PDF", type=["pdf"])

if uploaded_file is None:
    st.session_state.pop("search_state", None)
    st.session_state.pop("chat_turns", None)
    st.info("Upload a PDF to begin.")
else:
    pdf_bytes = uploaded_file.getvalue()
    fingerprint = sha256(pdf_bytes).hexdigest()
    st.write(f"File name: {uploaded_file.name}")
    st.write(f"File size: {uploaded_file.size / 1024:.1f} KB")

    # Never reuse indexed content or chat turns after a document change.
    if st.session_state.get("search_state", {}).get("fingerprint") != fingerprint:
        st.session_state.pop("search_state", None)
        st.session_state.pop("chat_turns", None)

    if st.button("Process document", type="primary"):
        st.session_state.pop("search_state", None)
        st.session_state.pop("chat_turns", None)
        try:
            page_count, text = extract_pdf_text(pdf_bytes)
        except (PdfReadError, ValueError, OSError) as exc:
            st.error(f"Could not read this PDF: {exc}")
        else:
            if not text:
                st.warning("No selectable text found. Scanned PDFs may require OCR.")
            else:
                with st.spinner("Generating embeddings and building search index..."):
                    try:
                        chunks, index, dimensions = index_document(
                            text, get_embedding_model(),
                        )
                    except Exception:
                        st.error("Could not index this PDF. Check dependencies and model access.")
                    else:
                        st.session_state["search_state"] = {
                            "fingerprint": fingerprint,
                            "page_count": page_count,
                            "preview": text[:5000],
                            "characters": len(text),
                            "chunks": chunks,
                            "dimensions": dimensions,
                            "index": index,
                        }

    state = st.session_state.get("search_state")
    if state is not None:
        st.success(f"Indexed {state['page_count']} page(s) and {len(state['chunks'])} chunks.")
        st.write(f"Extracted characters: {state['characters']:,}")
        st.write(f"Vector dimensions: {state['dimensions']:,}")
        with st.expander("Preview extracted text"):
            st.text_area("Text preview", value=state["preview"], height=240)

        turns = st.session_state.setdefault("chat_turns", [])
        if turns and st.button("Clear conversation"):
            turns.clear()

        with st.form("document_search"):
            question = st.text_input("Ask your document", placeholder="What does this document say about...?")
            submitted = st.form_submit_button("Ask question")
        if submitted:
            if not question.strip():
                st.warning("Enter a question to search the document.")
            else:
                with st.spinner("Searching document..."):
                    try:
                        retrieval_query = build_retrieval_query(
                            question, [turn["question"] for turn in turns],
                        )
                        results = search_chunks(
                            state["index"], state["chunks"], retrieval_query,
                            get_embedding_model(), k=3,
                        )
                    except Exception:
                        st.error("Search failed. Please try again.")
                    else:
                        answer = None
                        error = None
                        api_key = get_api_key()
                        if api_key and results:
                            history = [
                                (turn["question"], turn["answer"])
                                for turn in turns if turn["answer"]
                            ]
                            with st.spinner("Generating answer..."):
                                try:
                                    answer = answer_question(
                                        question, results, get_answer_model(api_key),
                                        history=history,
                                    )
                                except Exception:
                                    error = "Could not generate an answer. Please try again."
                        turns.append({
                            "question": question.strip(),
                            "results": results,
                            "answer": answer,
                            "error": error,
                            "has_api_key": bool(api_key),
                        })

        for turn in turns:
            with st.chat_message("user"):
                st.write(turn["question"])
            with st.chat_message("assistant"):
                if turn["answer"]:
                    st.markdown(turn["answer"])
                elif turn["error"]:
                    st.error(turn["error"])
                elif not turn["has_api_key"]:
                    st.info("Set GROQ_API_KEY to enable AI answers. Search results are available below.")
                else:
                    st.info("No relevant passages were found.")

                with st.expander("Relevant passages"):
                    for number, (passage, similarity) in enumerate(turn["results"], start=1):
                        st.caption(f"Match {number} · cosine similarity {similarity:.3f}")
                        st.text(passage)
