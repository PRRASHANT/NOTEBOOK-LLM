"""Utilities for reading text from uploaded PDF documents."""

from io import BytesIO

from pypdf import PdfReader

MAX_PDF_BYTES = 20 * 1024 * 1024


def validate_pdf_size(size: int) -> None:
    """Reject empty or oversized PDF uploads before loading their contents."""
    if size <= 0:
        raise ValueError("The PDF is empty.")
    if size > MAX_PDF_BYTES:
        raise ValueError("PDF exceeds the 20 MB upload limit.")


def extract_pdf_text(pdf_bytes: bytes) -> tuple[int, str]:
    """Return the page count and extractable text from a PDF byte string."""
    validate_pdf_size(len(pdf_bytes))
    reader = PdfReader(BytesIO(pdf_bytes))
    if reader.is_encrypted:
        raise ValueError("Password-protected PDFs are not supported.")

    pages = [page.extract_text() or "" for page in reader.pages]
    return len(pages), "\n\n".join(text.strip() for text in pages if text.strip())
