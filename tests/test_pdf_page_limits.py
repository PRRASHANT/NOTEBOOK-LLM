"""PDF page-count resource limit tests."""

import unittest
from io import BytesIO
from unittest.mock import patch

from pypdf import PdfWriter

from pdf_utils import MAX_PDF_PAGES, extract_pdf_text


class PdfPageLimitTests(unittest.TestCase):
    @staticmethod
    def make_pdf(pages):
        writer = PdfWriter()
        for _ in range(pages):
            writer.add_blank_page(width=200, height=200)
        output = BytesIO()
        writer.write(output)
        return output.getvalue()

    def test_small_pdf_extracts(self):
        self.assertEqual(extract_pdf_text(self.make_pdf(2)), (2, ""))

    def test_exact_page_limit_is_accepted(self):
        self.assertEqual(extract_pdf_text(self.make_pdf(MAX_PDF_PAGES)), (MAX_PDF_PAGES, ""))

    def test_over_page_limit_is_rejected_before_text_extraction(self):
        pdf = self.make_pdf(MAX_PDF_PAGES + 1)
        with patch("pypdf._page.PageObject.extract_text", side_effect=AssertionError("extracted")):
            with self.assertRaisesRegex(ValueError, "100-page limit"):
                extract_pdf_text(pdf)


if __name__ == "__main__":
    unittest.main()
