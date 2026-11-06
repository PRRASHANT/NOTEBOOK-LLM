"""Streamlit interface for PDF processing and semantic passage search."""

from hashlib import sha256

import streamlit as st
from langchain_huggingface import HuggingFaceEmbeddings
from pypdf.errors import PdfReadError

from embedding_utils import MODEL_NAME, embed_chunks
from pdf_utils import extract_pdf_text
from search_utils import build_search_index, search_chunks
from text_utils import chunk_text


@st.cache_resource(show_spinner=False)
def get_embedding_model():
    """Reuse the embedding model across Streamlit reruns."""
    return HuggingFaceEmbeddings(model_name=MODEL_NAME)


st.set_page_config(page_title="Notebook LLM", page_icon="📚")
st.title("📚 Notebook LLM")
st.write("Upload a PDF to find relevant passages using semantic search.")

uploaded_file = st.file_uploader("Choose a PDF", type=["pdf"])

if uploaded_file is None:
    st.session_state.pop("search_state", None)
    st.info("Upload a PDF to begin.")
else:
    pdf_bytes = uploaded_file.getvalue()
    fingerprint = sha256(pdf_bytes).hexdigest()
    st.write(f"File name: {uploaded_file.name}")
    st.write(f"File size: {uploaded_file.size / 1024:.1f} KB")

    # Never reuse search results from a different uploaded document.
    if st.session_state.get("search_state", {}).get("fingerprint") != fingerprint:
        st.session_state.pop("search_state", None)

    if st.button("Process document", type="primary"):
        st.session_state.pop("search_state", None)
        try:
            page_count, text = extract_pdf_text(pdf_bytes)
        except (PdfReadError, ValueError, OSError) as exc:
            st.error(f"Could not read this PDF: {exc}")
        else:
            if not text:
                st.warning("No selectable text found. Scanned PDFs may require OCR.")
            else:
                chunks = chunk_text(text)
                with st.spinner("Generating embeddings and building search index..."):
                    try:
                        model = get_embedding_model()
                        vectors = embed_chunks(chunks, model)
                        index = build_search_index(chunks, vectors)
                    except Exception:
                        st.error("Could not index this PDF. Check dependencies and model access.")
                    else:
                        st.session_state["search_state"] = {
                            "fingerprint": fingerprint,
                            "page_count": page_count,
                            "preview": text[:5000],
                            "characters": len(text),
                            "chunks": chunks,
                            "dimensions": len(vectors[0]),
                            "index": index,
                        }

    state = st.session_state.get("search_state")
    if state is not None:
        st.success(f"Indexed {state['page_count']} page(s) and {len(state['chunks'])} chunks.")
        st.write(f"Extracted characters: {state['characters']:,}")
        st.write(f"Vector dimensions: {state['dimensions']:,}")
        with st.expander("Preview extracted text"):
            st.text_area("Text preview", value=state["preview"], height=240)

        with st.form("document_search"):
            question = st.text_input("Search your document", placeholder="What does this document say about...?")
            submitted = st.form_submit_button("Find relevant passages")
        if submitted:
            if not question.strip():
                st.warning("Enter a question to search the document.")
            else:
                with st.spinner("Searching document..."):
                    try:
                        results = search_chunks(
                            state["index"], state["chunks"], question,
                            get_embedding_model(), k=3,
                        )
                    except Exception:
                        st.error("Search failed. Please try again.")
                    else:
                        st.subheader("Relevant passages")
                        for number, (passage, similarity) in enumerate(results, start=1):
                            st.caption(f"Match {number} · cosine similarity {similarity:.3f}")
                            st.text(passage)
