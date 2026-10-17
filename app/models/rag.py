from datetime import UTC, datetime
from typing import TYPE_CHECKING

from pgvector.sqlalchemy import Vector
from sqlalchemy import Column, Text
from sqlmodel import Field, Relationship, SQLModel

if TYPE_CHECKING:
    pass


def get_datetime_utc() -> datetime:
    return datetime.now(UTC)


class FinancialDocumentBase(SQLModel):
    filename: str = Field(max_length=255)
    upload_date: datetime = Field(default_factory=get_datetime_utc)
    status: str = Field(default="completed", max_length=50)


class FinancialDocument(FinancialDocumentBase, table=True):
    id: int | None = Field(default=None, primary_key=True)

    chunks: list[DocumentChunk] = Relationship(
        back_populates="document", cascade_delete=True
    )


class FinancialDocumentPublic(FinancialDocumentBase):
    id: int


class DocumentChunkBase(SQLModel):
    document_id: int = Field(
        foreign_key="financialdocument.id", nullable=False, ondelete="CASCADE"
    )
    chunk_text: str = Field(sa_column=Column(Text, nullable=False))


class DocumentChunk(DocumentChunkBase, table=True):
    id: int | None = Field(default=None, primary_key=True)
    embedding: list[float] | None = Field(
        default=None, sa_column=Column(Vector(1536))
    )

    document: FinancialDocument | None = Relationship(back_populates="chunks")


class DocumentChunkPublic(DocumentChunkBase):
    id: int


# API Schemas
class DocumentUploadResponse(SQLModel):
    document_id: int
    filename: str
    status: str
    chunks_count: int
    message: str = "Document processed and indexed successfully"


class QueryRequest(SQLModel):
    query: str
    document_id: int | None = None


class QueryResponse(SQLModel):
    query: str
    answer: str
    citations: list[str]
