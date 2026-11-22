"""Regression tests for PDFs containing no pages."""

import unittest
from io import BytesIO

from pypdf import PdfWriter

from pdf_utils import extract_pdf_text


def pdf_bytes(page_count):
    writer = PdfWriter()
    for _ in range(page_count):
        writer.add_blank_page(width=300, height=300)
    output = BytesIO()
    writer.write(output)
    return output.getvalue()


class EmptyPdfTests(unittest.TestCase):
    def test_rejects_zero_page_pdf(self):
        with self.assertRaisesRegex(ValueError, "PDF contains no pages"):
            extract_pdf_text(pdf_bytes(0))

    def test_accepts_one_blank_page(self):
        self.assertEqual(extract_pdf_text(pdf_bytes(1)), (1, ""))


if __name__ == "__main__":
    unittest.main()
