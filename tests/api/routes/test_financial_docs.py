from unittest.mock import MagicMock, patch

import pymupdf
from fastapi.testclient import TestClient

from app.core.config import settings
from app.models.rag import DocumentChunk


def create_sample_pdf_bytes(content: str = "Alphabet Inc. Q4 Net Income reached $20.7 Billion.") -> bytes:
    doc = pymupdf.open()
    page = doc.new_page()
    page.insert_text((50, 50), content)
    data = doc.tobytes()
    doc.close()
    return data


def test_upload_financial_doc_unauthorized(client: TestClient) -> None:
    response = client.post(
        f"{settings.API_V1_STR}/financial-docs/upload",
        files={"file": ("test.pdf", b"%PDF-dummy", "application/pdf")},
    )
    assert response.status_code == 401


def test_query_financial_doc_unauthorized(client: TestClient) -> None:
    response = client.post(
        f"{settings.API_V1_STR}/financial-docs/query",
        json={"query": "What is the net income?"},
    )
    assert response.status_code == 401


def test_upload_invalid_file_extension(
    client: TestClient, superuser_token_headers: dict[str, str]
) -> None:
    response = client.post(
        f"{settings.API_V1_STR}/financial-docs/upload",
        headers=superuser_token_headers,
        files={"file": ("test.txt", b"plain text", "text/plain")},
    )
    assert response.status_code == 400
    assert "Invalid file format" in response.json()["detail"]


def test_upload_empty_pdf(
    client: TestClient, superuser_token_headers: dict[str, str]
) -> None:
    response = client.post(
        f"{settings.API_V1_STR}/financial-docs/upload",
        headers=superuser_token_headers,
        files={"file": ("test.pdf", b"", "application/pdf")},
    )
    assert response.status_code == 400
    assert "The uploaded file is empty" in response.json()["detail"]


@patch("app.api.routes.financial_docs.rag_service.generate_embeddings")
def test_upload_financial_doc_success(
    mock_embeddings: MagicMock,
    client: TestClient,
    superuser_token_headers: dict[str, str],
) -> None:
    mock_embeddings.return_value = [[0.05] * 1536]
    pdf_bytes = create_sample_pdf_bytes("Tesla reported $25.17B in total revenues.")

    response = client.post(
        f"{settings.API_V1_STR}/financial-docs/upload",
        headers=superuser_token_headers,
        files={"file": ("tesla_q4.pdf", pdf_bytes, "application/pdf")},
    )
    assert response.status_code == 201
    data = response.json()
    assert "document_id" in data
    assert data["filename"] == "tesla_q4.pdf"
    assert data["status"] == "completed"
    assert data["chunks_count"] >= 1


@patch("app.api.routes.financial_docs.rag_service.generate_answer")
@patch("app.api.routes.financial_docs.rag_service.hybrid_search")
def test_query_financial_docs_success(
    mock_search: MagicMock,
    mock_answer: MagicMock,
    client: TestClient,
    superuser_token_headers: dict[str, str],
) -> None:
    fake_chunk = DocumentChunk(
        id=10,
        document_id=1,
        chunk_text="Tesla reported $25.17B in total revenues.",
        embedding=[0.05] * 1536,
    )
    mock_search.return_value = [fake_chunk]
    mock_answer.return_value = (
        "Tesla's total revenue was $25.17B.",
        ["Tesla reported $25.17B in total revenues."],
    )

    response = client.post(
        f"{settings.API_V1_STR}/financial-docs/query",
        headers=superuser_token_headers,
        json={"query": "What was Tesla revenue?"},
    )
    assert response.status_code == 200
    data = response.json()
    assert data["query"] == "What was Tesla revenue?"
    assert data["answer"] == "Tesla's total revenue was $25.17B."
    assert len(data["citations"]) == 1
    assert data["citations"][0] == "Tesla reported $25.17B in total revenues."


def test_query_empty_string(
    client: TestClient, superuser_token_headers: dict[str, str]
) -> None:
    response = client.post(
        f"{settings.API_V1_STR}/financial-docs/query",
        headers=superuser_token_headers,
        json={"query": "   "},
    )
    assert response.status_code == 400
    assert "Query string cannot be empty" in response.json()["detail"]


def test_query_nonexistent_document_id(
    client: TestClient, superuser_token_headers: dict[str, str]
) -> None:
    response = client.post(
        f"{settings.API_V1_STR}/financial-docs/query",
        headers=superuser_token_headers,
        json={"query": "What are the numbers?", "document_id": 99999999},
    )
    assert response.status_code == 404
    assert "not found" in response.json()["detail"]
