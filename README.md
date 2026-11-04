# Notebook LLM

A Python application for uploading PDF documents and preparing their contents for document-based question answering.

## Tech stack

- Python
- Streamlit
- PyPDF
- LangChain
- FAISS
- HuggingFace embeddings
- Groq

## Current features

- PDF upload interface with file metadata
- PDF text extraction and preview
- Overlapping text chunks for downstream retrieval
- Input validation and automated chunking tests

Semantic search and question answering are under development.

## Setup

```bash
python -m venv .venv
pip install -r requirements.txt
streamlit run app.py
```

## Run tests

```bash
python -m unittest discover -s tests -v
```

Never commit API keys or local secrets.
