"""Exercise PDF extraction through grounded answering without external services."""

from io import BytesIO
import importlib.util
from types import SimpleNamespace
import unittest

from pypdf import PdfWriter
from pypdf.generic import DecodedStreamObject, DictionaryObject, NameObject

from answer_utils import answer_question
from document_utils import index_document
from pdf_utils import extract_pdf_text
from retrieval_utils import build_retrieval_query
from search_utils import search_chunks
from text_utils import chunk_text


class DeterministicEmbedder:
    """Use stable semantic categories without downloading an embedding model."""

    def embed_documents(self, texts):
        return [self.embed_query(text) for text in texts]

    def embed_query(self, text):
        return [1.0, 0.0] if "RAG" in text else [0.0, 1.0]


class RecordingChatModel:
    def __init__(self):
        self.messages = None

    def invoke(self, messages):
        self.messages = messages
        return SimpleNamespace(content="RAG retrieves document passages. [1]")


def make_pdf(*page_texts):
    """Generate a small selectable-text PDF in memory using PyPDF."""
    writer = PdfWriter()
    for text in page_texts:
        page = writer.add_blank_page(width=400, height=400)
        font = writer._add_object(DictionaryObject({
            NameObject("/Type"): NameObject("/Font"),
            NameObject("/Subtype"): NameObject("/Type1"),
            NameObject("/BaseFont"): NameObject("/Helvetica"),
        }))
        page[NameObject("/Resources")] = DictionaryObject({
            NameObject("/Font"): DictionaryObject({NameObject("/F1"): font}),
        })
        stream = DecodedStreamObject()
        stream.set_data(b"BT /F1 12 Tf 25 300 Td (" + text.encode("ascii") + b") Tj ET")
        page[NameObject("/Contents")] = writer._add_object(stream)
    buffer = BytesIO()
    writer.write(buffer)
    return buffer.getvalue()


class PdfQuestionAnsweringIntegrationTests(unittest.TestCase):
    def setUp(self):
        self.pdf = make_pdf(
            "RAG retrieves relevant passages from a document.",
            "Gardening needs soil and water for healthy plants.",
        )

    def test_extracts_selectable_text_from_multiple_pages(self):
        pages, text = extract_pdf_text(self.pdf)
        self.assertEqual(pages, 2)
        self.assertIn("RAG retrieves relevant passages", text)
        self.assertIn("Gardening needs soil", text)
        self.assertEqual(len(chunk_text(text, chunk_size=51, chunk_overlap=0)), 2)

    def test_rejects_empty_document_before_embedding(self):
        with self.assertRaisesRegex(ValueError, "without selectable text"):
            index_document("  ", DeterministicEmbedder())

    @unittest.skipUnless(importlib.util.find_spec("faiss") is not None,
                         "faiss-cpu is not installed")
    def test_pdf_to_grounded_answer_with_native_faiss(self):
        pages, text = extract_pdf_text(self.pdf)
        self.assertEqual(pages, 2)
        embedder = DeterministicEmbedder()
        chunks, index, dimensions = index_document(
            text, embedder, chunk_size=51, chunk_overlap=0,
        )
        self.assertEqual((len(chunks), dimensions, index.ntotal), (2, 2, 2))

        question = "What does it retrieve?"
        query = build_retrieval_query(question, ["What is RAG?"])
        self.assertIn("What is RAG?", query)
        results = search_chunks(index, chunks, query, embedder, k=1)
        self.assertIn("RAG retrieves relevant passages", results[0][0])
        self.assertAlmostEqual(results[0][1], 1.0, places=6)

        model = RecordingChatModel()
        answer = answer_question(
            question, results, model,
            history=[("What is RAG?", "It combines retrieval and generation.")],
        )
        self.assertEqual(answer, "RAG retrieves document passages. [1]")
        self.assertIn("[1] RAG retrieves relevant passages", model.messages[1][1])
        self.assertIn("Previous question: What is RAG?", model.messages[1][1])
        self.assertIn("NOT evidence", model.messages[0][1])


if __name__ == "__main__":
    unittest.main()
