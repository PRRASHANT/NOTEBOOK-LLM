"""Validate recognizable PDF version signatures before parsing."""

import unittest
from io import BytesIO

from pypdf import PdfWriter

from pdf_utils import extract_pdf_text, validate_pdf_header


class PdfVersionHeaderTests(unittest.TestCase):
    def test_accepts_supported_version_headers_with_prefix(self):
        for version in (b"1.4", b"1.7", b"2.0"):
            with self.subTest(version=version):
                validate_pdf_header(b"prefix\n%PDF-" + version + b"\n")

    def test_rejects_incomplete_or_malformed_versions(self):
        for header in (b"%PDF-", b"%PDF-x.y", b"%PDF-1.70", b"%PDF-3.0"):
            with self.subTest(header=header):
                with self.assertRaisesRegex(ValueError, "valid PDF header"):
                    validate_pdf_header(header)

    def test_rejects_signature_outside_header_window(self):
        with self.assertRaisesRegex(ValueError, "valid PDF header"):
            validate_pdf_header(b"x" * 1024 + b"%PDF-1.7")

    def test_accepts_generated_pdf(self):
        writer = PdfWriter()
        writer.add_blank_page(width=200, height=200)
        buffer = BytesIO()
        writer.write(buffer)
        self.assertEqual(extract_pdf_text(buffer.getvalue()), (1, ""))


if __name__ == "__main__":
    unittest.main()
