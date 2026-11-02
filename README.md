# Notebook LLM

A Python application for uploading PDF documents and asking questions about their content.

## Tech stack

- Python
- Streamlit
- PyPDF
- LangChain
- FAISS
- HuggingFace embeddings
- Groq

## Current features

- PDF upload interface
- File name and size display

Document parsing, semantic search, and question answering are under development.

## Setup

```bash
python -m venv .venv
pip install -r requirements.txt
streamlit run app.py
```

Never commit API keys or local secrets.
