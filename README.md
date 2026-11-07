# Notebook LLM

A Python application for uploading PDF documents, searching their contents, and generating document-grounded answers.

## Tech stack

- Python and Streamlit
- PyPDF
- LangChain and HuggingFace embeddings
- FAISS
- Groq

## Features

- PDF upload with file metadata and text extraction
- Overlapping text chunks and MiniLM text embeddings
- In-memory FAISS similarity search over document passages
- Question answering grounded in the three most relevant retrieved passages
- Numbered source passages shown alongside generated answers
- Search-only mode when no Groq API key is configured
- Automated tests for text preparation, retrieval, and answer generation

Scanned PDFs without selectable text require OCR. Generated answers should be checked against the displayed source passages.

## Setup

```bash
python -m venv .venv
pip install -r requirements.txt
streamlit run app.py
```

Set `GROQ_API_KEY` as an environment variable, or in a local `.streamlit/secrets.toml` file (ignored by Git):

```toml
GROQ_API_KEY = "your_api_key_here"
```

The first document processing run may download the embedding model. An internet connection is required for that initial download and for Groq API calls.

## Run tests

```bash
python -m unittest discover -s tests -v
```

The FAISS retrieval unit tests use a lightweight index double; verify the installed native FAISS integration when deploying. Never commit API keys or local secrets.
