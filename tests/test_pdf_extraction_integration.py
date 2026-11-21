"""Integration tests for extracting real selectable text from PDF bytes."""

import unittest
from io import BytesIO

from pypdf import PdfWriter
from pypdf.generic import DecodedStreamObject, DictionaryObject, NameObject

from pdf_utils import extract_pdf_text


def make_pdf_with_text(*page_texts):
    """Create a small PDF with actual text drawing operations, without fixtures."""
    writer = PdfWriter()
    font = DictionaryObject({
        NameObject("/Type"): NameObject("/Font"),
        NameObject("/Subtype"): NameObject("/Type1"),
        NameObject("/BaseFont"): NameObject("/Helvetica"),
    })
    font_ref = writer._add_object(font)
    for text in page_texts:
        page = writer.add_blank_page(width=300, height=300)
        page[NameObject("/Resources")] = DictionaryObject({
            NameObject("/Font"): DictionaryObject({NameObject("/F1"): font_ref}),
        })
        stream = DecodedStreamObject()
        stream.set_data(f"BT /F1 12 Tf 40 200 Td ({text}) Tj ET".encode("ascii"))
        page[NameObject("/Contents")] = writer._add_object(stream)
    output = BytesIO()
    writer.write(output)
    return output.getvalue()


class PdfExtractionIntegrationTests(unittest.TestCase):
    def test_extracts_selectable_text_from_real_pdf(self):
        count, text = extract_pdf_text(make_pdf_with_text("Alpha passage"))
        self.assertEqual(count, 1)
        self.assertEqual(text, "Alpha passage")

    def test_preserves_page_order_and_separators(self):
        count, text = extract_pdf_text(make_pdf_with_text("First page", "Second page"))
        self.assertEqual(count, 2)
        self.assertEqual(text, "First page\n\nSecond page")

    def test_skips_blank_pages_without_losing_page_count(self):
        writer = PdfWriter()
        writer.add_blank_page(width=300, height=300)
        output = BytesIO()
        writer.write(output)
        count, text = extract_pdf_text(output.getvalue())
        self.assertEqual((count, text), (1, ""))


if __name__ == "__main__":
    unittest.main()
