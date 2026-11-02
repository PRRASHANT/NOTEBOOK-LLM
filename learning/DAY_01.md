# Day 1 — First Streamlit app

## What we added
- A Streamlit page with a title and instructions
- A PDF-only file upload control
- A success message showing the uploaded file name and size
- Python dependencies and a .gitignore that excludes local secrets

## Learn the code
- `import streamlit as st`: load the Streamlit library with a short alias.
- `st.title(...)`: show a heading.
- `st.file_uploader(..., type=["pdf"])`: display a picker that accepts PDF files.
- `if uploaded_file is not None`: execute the next block only after selecting a file.
- `uploaded_file.name` and `uploaded_file.size`: inspect metadata about the uploaded file.

## Run locally
1. Install Python 3.10+.
2. Create a virtual environment: `python -m venv .venv`.
3. Activate it and run `pip install -r requirements.txt`.
4. Start the app with `streamlit run app.py`.
5. Upload a PDF and check the displayed name and file size.

## Practice
Change the page title and display the file size in MB instead of KB.

## Next lesson
Use PyPDF to extract text from a selected PDF file.

## Source acknowledgment
This educational rebuild is based on ideas from [saksham-giri/notebooklm-lite](https://github.com/saksham-giri/notebooklm-lite). We are implementing the application incrementally rather than claiming the original code as our own.
