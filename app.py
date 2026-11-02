"""Day 1: A simple PDF upload interface for our document assistant.

Based on the learning goals of saksham-giri/notebooklm-lite.
We will add PDF reading, retrieval, and AI answers in later lessons.
"""
import streamlit as st

st.set_page_config(page_title="Notebook LLM", page_icon="📚")
st.title("📚 Notebook LLM")
st.write("Learn to build a document question-answering assistant step by step.")

uploaded_file = st.file_uploader("Choose a PDF", type=["pdf"])

if uploaded_file is not None:
    st.success("PDF selected successfully!")
    st.write(f"File name: {uploaded_file.name}")
    st.write(f"File size: {uploaded_file.size / 1024:.1f} KB")
    st.info("Next lesson: extract text from the PDF using PyPDF.")
else:
    st.info("Upload a PDF to begin.")
