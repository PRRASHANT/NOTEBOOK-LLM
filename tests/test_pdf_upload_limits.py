"""Upload size validation tests."""

import unittest
from io import BytesIO

from pypdf import PdfWriter

from pdf_utils import MAX_PDF_BYTES, extract_pdf_text, validate_pdf_size


class PdfUploadLimitTests(unittest.TestCase):
    def test_accepts_sizes_up_to_limit(self):
        validate_pdf_size(1)
        validate_pdf_size(MAX_PDF_BYTES)

    def test_rejects_empty_and_oversized_uploads(self):
        for size, message in [(0, "empty"), (MAX_PDF_BYTES + 1, "20 MB")]:
            with self.subTest(size=size), self.assertRaisesRegex(ValueError, message):
                validate_pdf_size(size)

    def test_extraction_rejects_empty_bytes(self):
        with self.assertRaisesRegex(ValueError, "empty"):
            extract_pdf_text(b"")

    def test_extraction_rejects_oversized_bytes(self):
        with self.assertRaisesRegex(ValueError, "20 MB"):
            extract_pdf_text(b"x" * (MAX_PDF_BYTES + 1))

    def test_valid_pdf_still_extracts(self):
        writer = PdfWriter()
        writer.add_blank_page(width=200, height=200)
        buffer = BytesIO()
        writer.write(buffer)
        self.assertEqual(extract_pdf_text(buffer.getvalue()), (1, ""))


if __name__ == "__main__":
    unittest.main()
