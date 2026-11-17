"""Tests for bounded PDF text extraction."""

import unittest
from io import BytesIO
from unittest.mock import patch

from pypdf import PdfWriter

from pdf_utils import MAX_EXTRACTED_CHARACTERS, extract_pdf_text


class PdfTextLimitTests(unittest.TestCase):
    @staticmethod
    def make_pdf(pages=1):
        writer = PdfWriter()
        for _ in range(pages):
            writer.add_blank_page(width=200, height=200)
        output = BytesIO()
        writer.write(output)
        return output.getvalue()

    def test_text_at_limit_is_accepted(self):
        text = "x" * MAX_EXTRACTED_CHARACTERS
        with patch("pypdf._page.PageObject.extract_text", return_value=text):
            self.assertEqual(extract_pdf_text(self.make_pdf()), (1, text))

    def test_text_over_limit_is_rejected(self):
        with patch("pypdf._page.PageObject.extract_text", return_value="x" * (MAX_EXTRACTED_CHARACTERS + 1)):
            with self.assertRaisesRegex(ValueError, "extracted-text character limit"):
                extract_pdf_text(self.make_pdf())

    def test_separators_count_toward_limit(self):
        with patch("pypdf._page.PageObject.extract_text", side_effect=["x" * (MAX_EXTRACTED_CHARACTERS - 1), "y"]):
            with self.assertRaisesRegex(ValueError, "extracted-text character limit"):
                extract_pdf_text(self.make_pdf(2))

    def test_blank_pages_do_not_consume_separator_budget(self):
        with patch("pypdf._page.PageObject.extract_text", side_effect=["", " hello "]):
            self.assertEqual(extract_pdf_text(self.make_pdf(2)), (2, "hello"))


if __name__ == "__main__":
    unittest.main()
