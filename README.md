# Notebook LLM — Learning Build

A beginner-friendly, step-by-step rebuild of a PDF question-answering assistant using Python, Streamlit, LangChain, embeddings, FAISS, and Groq.

> Learning reference: [saksham-giri/notebooklm-lite](https://github.com/saksham-giri/notebooklm-lite). This project develops an educational implementation incrementally and acknowledges the original inspiration.

## Current progress

**Day 1:** Create a Streamlit interface and select PDF files. PDF text extraction and AI question-answering are not implemented yet.

Read the walkthrough: [Day 1 learning notes](learning/DAY_01.md).

## Run locally

```bash
python -m venv .venv
# Activate your environment, then:
pip install -r requirements.txt
streamlit run app.py
```

## Learning roadmap

| Day | Goal |
| --- | --- |
| 1 | Streamlit setup and PDF upload |
| 2 | Read and extract PDF text with PyPDF |
| 3 | Split long text into chunks |
| 4 | Generate text embeddings |
| 5 | Store and search document chunks with FAISS |
| 6 | Connect Groq and generate grounded answers |
| 7 | Build a chat experience and display sources |
| 8+ | Add error handling, tests, and improvements |

Each day's changes should be reviewed, tested, and explained before proceeding. Never commit API keys or uploaded private documents.
