import logging
from typing import Any

from fastapi import APIRouter, File, HTTPException, UploadFile, status
from sqlmodel import col, select

from app.api.deps import CurrentUser, SessionDep
from app.models.rag import (
    DocumentUploadResponse,
    FinancialDocument,
    FinancialDocumentPublic,
    QueryRequest,
    QueryResponse,
)
from app.services import rag_service

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/financial-docs", tags=["financial-docs"])


@router.post(
    "/upload",
    response_model=DocumentUploadResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Upload and process a Financial PDF Document",
)
async def upload_financial_document(
    session: SessionDep,
    _current_user: CurrentUser,
    file: UploadFile = File(..., description="PDF file to ingest and index"),
) -> Any:
    """Accepts a PDF file upload, calls the RAG service to process, chunk, embed,

    and store it into PostgreSQL with pgvector. Returns the document ID.
    """
    if not file.filename or not file.filename.lower().endswith(".pdf"):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid file format. Only PDF files are supported.",
        )

    try:
        content = await file.read()
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Could not read uploaded file: {exc}",
        )

    if not content:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="The uploaded file is empty.",
        )

    # 1. Create document entry with 'processing' status
    doc = FinancialDocument(
        filename=file.filename,
        status="processing",
    )
    session.add(doc)
    session.commit()
    session.refresh(doc)
    assert doc.id is not None

    try:
        # 2. Extract text from PDF
        text = rag_service.process_pdf(content)

        # 3. Chunk text into semantic chunks
        chunks = rag_service.chunk_text(text)
        if not chunks:
            raise ValueError(
                "No extractable text chunks could be generated from the document."
            )

        # 4. Generate embeddings via OpenAI
        embeddings = rag_service.generate_embeddings(chunks)

        # 5. Store chunks in database
        rag_service.store_chunks_in_db(
            chunks=chunks,
            embeddings=embeddings,
            doc_id=doc.id,
            session=session,
        )

        # 6. Update document status
        doc.status = "completed"
        session.add(doc)
        session.commit()
        session.refresh(doc)

        return DocumentUploadResponse(
            document_id=doc.id,
            filename=doc.filename,
            status=doc.status,
            chunks_count=len(chunks),
            message="Document successfully processed, embedded, and indexed for search.",
        )

    except Exception as exc:
        logger.exception(f"Error processing financial document {doc.id}: {exc}")
        doc.status = "failed"
        session.add(doc)
        session.commit()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to process document: {exc}",
        )


@router.post(
    "/query",
    response_model=QueryResponse,
    status_code=status.HTTP_200_OK,
    summary="Query Financial Documents using RAG",
)
def query_financial_documents(
    session: SessionDep,
    _current_user: CurrentUser,
    payload: QueryRequest,
) -> Any:
    """Accepts a JSON query payload with an optional document_id filter.

    Performs vector cosine similarity search, synthesizes an answer using
    gpt-4o-mini, and returns the grounded answer along with exact citation
    chunks.
    """
    clean_query = payload.query.strip()
    if not clean_query:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Query string cannot be empty.",
        )

    # If document_id is specified, verify it exists
    if payload.document_id is not None:
        doc = session.get(FinancialDocument, payload.document_id)
        if not doc:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Financial document with id {payload.document_id} not found.",
            )

    try:
        # Retrieve top 5 most relevant chunks using hybrid vector search
        chunks = rag_service.hybrid_search(
            query_text=clean_query,
            document_id=payload.document_id,
            top_k=5,
            session=session,
        )

        if not chunks:
            return QueryResponse(
                query=clean_query,
                answer="No relevant financial document context was found to answer this query. Please ensure relevant documents have been uploaded and processed.",
                citations=[],
            )

        # Generate response grounded in retrieved chunks
        answer, citations = rag_service.generate_answer(
            query=clean_query,
            context_chunks=chunks,
        )

        return QueryResponse(
            query=clean_query,
            answer=answer,
            citations=citations,
        )

    except HTTPException:
        raise
    except Exception as exc:
        logger.exception(f"Error executing RAG query: {exc}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"RAG query execution failed: {exc}",
        )


@router.get(
    "/",
    response_model=list[FinancialDocumentPublic],
    summary="List all uploaded financial documents",
)
def list_financial_documents(
    session: SessionDep,
    _current_user: CurrentUser,
    skip: int = 0,
    limit: int = 50,
) -> Any:
    """Retrieve all indexed financial documents."""
    statement = (
        select(FinancialDocument)
        .order_by(col(FinancialDocument.upload_date).desc())
        .offset(skip)
        .limit(limit)
    )
    docs = session.exec(statement).all()
    return docs


@router.get(
    "/{document_id}",
    response_model=FinancialDocumentPublic,
    summary="Get details of a financial document",
)
def get_financial_document(
    document_id: int,
    session: SessionDep,
    _current_user: CurrentUser,
) -> Any:
    """Retrieve a single document by its ID."""
    doc = session.get(FinancialDocument, document_id)
    if not doc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Document with id {document_id} not found.",
        )
    return doc


@router.delete(
    "/{document_id}",
    summary="Delete a financial document and its chunks",
)
def delete_financial_document(
    document_id: int,
    session: SessionDep,
    _current_user: CurrentUser,
) -> Any:
    """Delete a financial document along with all its associated vector chunks."""
    doc = session.get(FinancialDocument, document_id)
    if not doc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Document with id {document_id} not found.",
        )
    session.delete(doc)
    session.commit()
    return {"message": f"Document {document_id} and all related chunks successfully deleted."}
