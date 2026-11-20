"""Utilities for reading text from uploaded PDF documents."""

from io import BytesIO

from pypdf import PdfReader

MAX_PDF_BYTES = 20 * 1024 * 1024
MAX_PDF_PAGES = 100
MAX_EXTRACTED_CHARACTERS = 1_000_000


def validate_pdf_size(size: int) -> None:
    """Reject empty or oversized PDF uploads before loading their contents."""
    if size <= 0:
        raise ValueError("The PDF is empty.")
    if size > MAX_PDF_BYTES:
        raise ValueError("PDF exceeds the 20 MB upload limit.")


def validate_pdf_header(pdf_bytes: bytes) -> None:
    """Reject files without a PDF header in the first 1024 bytes."""
    if b"%PDF-" not in pdf_bytes[:1024]:
        raise ValueError("The uploaded file does not have a valid PDF header.")


def extract_pdf_text(pdf_bytes: bytes) -> tuple[int, str]:
    """Return page count and selectable text from a PDF within resource limits."""
    validate_pdf_size(len(pdf_bytes))
    validate_pdf_header(pdf_bytes)
    reader = PdfReader(BytesIO(pdf_bytes))
    if reader.is_encrypted:
        raise ValueError("Password-protected PDFs are not supported.")

    page_count = len(reader.pages)
    if page_count > MAX_PDF_PAGES:
        raise ValueError(f"PDF exceeds the {MAX_PDF_PAGES}-page limit.")

    pages = []
    extracted_characters = 0
    for page in reader.pages:
        text = (page.extract_text() or "").strip()
        if not text:
            continue
        # Account for separators so the final returned string stays bounded.
        next_length = extracted_characters + len(text) + (2 if pages else 0)
        if next_length > MAX_EXTRACTED_CHARACTERS:
            raise ValueError("PDF exceeds the extracted-text character limit.")
        pages.append(text)
        extracted_characters = next_length
    return page_count, "\n\n".join(pages)
