"""Utilities for reading text from uploaded PDF documents."""

from io import BytesIO

from pypdf import PdfReader


def extract_pdf_text(pdf_bytes: bytes) -> tuple[int, str]:
    """Return the page count and extractable text from a PDF byte string."""
    reader = PdfReader(BytesIO(pdf_bytes))
    if reader.is_encrypted:
        raise ValueError("Password-protected PDFs are not supported.")

    pages = [page.extract_text() or "" for page in reader.pages]
    return len(pages), "\n\n".join(text.strip() for text in pages if text.strip())
