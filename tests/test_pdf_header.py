"""PDF signature validation tests."""

import unittest
from io import BytesIO

from pypdf import PdfWriter

from pdf_utils import extract_pdf_text, validate_pdf_header


class PdfHeaderTests(unittest.TestCase):
    def test_rejects_non_pdf_data(self):
        for content in (b"not a pdf", b"%PDA-1.7\n", b"x" * 1024 + b"%PDF-1.7"):
            with self.subTest(content=content[:20]), self.assertRaisesRegex(ValueError, "PDF header"):
                extract_pdf_text(content)

    def test_accepts_header_after_short_prefix(self):
        validate_pdf_header(b"prefix\n%PDF-1.7\n")

    def test_accepts_real_pdf(self):
        writer = PdfWriter()
        writer.add_blank_page(width=200, height=200)
        output = BytesIO()
        writer.write(output)
        self.assertEqual(extract_pdf_text(output.getvalue()), (1, ""))


if __name__ == "__main__":
    unittest.main()
