from unittest.mock import MagicMock

import pymupdf
import pytest

from app.models.rag import DocumentChunk
from app.services import rag_service


def create_test_pdf_bytes(text: str) -> bytes:
    doc = pymupdf.open()
    page = doc.new_page()
    page.insert_text((50, 50), text)
    data = doc.tobytes()
    doc.close()
    return data


def test_process_pdf_valid():
    sample_text = "Acme Corp Financial Summary\nFY2024 Revenue was 120M USD.\nNet profit margin 18.5%."
    pdf_bytes = create_test_pdf_bytes(sample_text)
    extracted = rag_service.process_pdf(pdf_bytes)
    assert "Acme Corp Financial Summary" in extracted
    assert "120M USD" in extracted


def test_process_pdf_empty_error():
    doc = pymupdf.open()
    doc.new_page()  # blank page without text
    empty_pdf = doc.tobytes()
    doc.close()
    with pytest.raises(ValueError, match="does not contain extractable text"):
        rag_service.process_pdf(empty_pdf)


def test_chunk_text_basic():
    text = (
        "Q1 Revenue grew by 12% year-over-year reaching $4.2B.\n\n"
        "Operating expenses decreased by 3% due to optimized procurement.\n\n"
        "Free cash flow conversion remained robust at 110% of net income."
    )
    chunks = rag_service.chunk_text(text, chunk_size=30, overlap=5)
    assert len(chunks) >= 1
    assert any("Revenue grew" in c for c in chunks)


def test_chunk_text_empty():
    assert rag_service.chunk_text("") == []
    assert rag_service.chunk_text("   ") == []


def test_generate_embeddings_mocked():
    mock_client = MagicMock()
    mock_item_1 = MagicMock()
    mock_item_1.index = 0
    mock_item_1.embedding = [0.1] * 1536

    mock_item_2 = MagicMock()
    mock_item_2.index = 1
    mock_item_2.embedding = [0.2] * 1536

    mock_client.embeddings.create.return_value.data = [mock_item_1, mock_item_2]

    chunks = ["chunk one", "chunk two"]
    embeddings = rag_service.generate_embeddings(chunks, client=mock_client)
    assert len(embeddings) == 2
    assert len(embeddings[0]) == 1536
    assert embeddings[0][0] == 0.1
    assert embeddings[1][0] == 0.2


def test_generate_answer_mocked():
    mock_client = MagicMock()
    mock_choice = MagicMock()
    mock_choice.message.content = "Apple Q3 revenue was $85.8B."
    mock_client.chat.completions.create.return_value.choices = [mock_choice]

    chunk = DocumentChunk(
        id=1,
        document_id=1,
        chunk_text="In Q3, Apple posted revenue of $85.8 billion.",
        embedding=[0.0] * 1536,
    )
    answer, citations = rag_service.generate_answer(
        query="What was Apple's Q3 revenue?",
        context_chunks=[chunk],
        client=mock_client,
    )

    assert answer == "Apple Q3 revenue was $85.8B."
    assert len(citations) == 1
    assert citations[0] == "In Q3, Apple posted revenue of $85.8 billion."
