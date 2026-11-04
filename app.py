"""Streamlit interface for uploading and preparing PDF documents."""

import streamlit as st
from pypdf.errors import PdfReadError

from pdf_utils import extract_pdf_text
from text_utils import chunk_text

st.set_page_config(page_title="Notebook LLM", page_icon="📚")
st.title("📚 Notebook LLM")
st.write("Upload a PDF to extract and prepare its text.")

uploaded_file = st.file_uploader("Choose a PDF", type=["pdf"])

if uploaded_file is not None:
    st.write(f"File name: {uploaded_file.name}")
    st.write(f"File size: {uploaded_file.size / 1024:.1f} KB")

    if st.button("Process document", type="primary"):
        try:
            page_count, text = extract_pdf_text(uploaded_file.getvalue())
        except (PdfReadError, ValueError, OSError) as exc:
            st.error(f"Could not read this PDF: {exc}")
        else:
            st.success(f"Processed {page_count} page(s).")
            if text:
                chunks = chunk_text(text)
                st.write(f"Extracted characters: {len(text):,}")
                st.write(f"Text chunks: {len(chunks):,}")
                st.text_area("Text preview", value=text[:5000], height=250)
                if len(text) > 5000:
                    st.caption("Showing the first 5,000 characters of extracted text.")
                with st.expander("Preview first text chunk"):
                    st.text(chunks[0])
            else:
                st.warning("No selectable text found. Scanned PDFs may require OCR.")
else:
    st.info("Upload a PDF to begin.")
