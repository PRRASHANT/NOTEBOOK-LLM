# Notebook LLM

A Python application for uploading PDF documents and searching their contents by semantic similarity.

## Tech stack

- Python
- Streamlit
- PyPDF
- LangChain
- FAISS
- HuggingFace embeddings

## Current features

- PDF upload interface with file metadata
- PDF text extraction and preview
- Overlapping text chunks for downstream retrieval
- Text embeddings generated with the all-MiniLM-L6-v2 model
- In-memory FAISS similarity search for the top three relevant passages
- Input validation and automated tests for text processing and retrieval

Generated answers using an LLM are not yet implemented. Scanned PDFs without selectable text require OCR.

## Setup

```bash
python -m venv .venv
pip install -r requirements.txt
streamlit run app.py
```

The first document processing run may download the embedding model. An internet connection is required for that initial download.

## Run tests

```bash
python -m unittest discover -s tests -v
```

The FAISS retrieval unit tests use a lightweight index double; verify the installed native FAISS integration when deploying.

Never commit API keys or local secrets.
