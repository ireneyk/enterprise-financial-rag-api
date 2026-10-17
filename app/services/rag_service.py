import logging
from io import BytesIO

import tiktoken
from openai import OpenAI
from sqlmodel import Session, select

from app.core.config import settings
from app.core.db import engine
from app.models.rag import DocumentChunk

logger = logging.getLogger(__name__)


def process_pdf(file_bytes: bytes) -> str:
    """Extracts text from the uploaded PDF using PyMuPDF (with pypdf fallback)."""
    text_content: list[str] = []

    # Attempt extraction with PyMuPDF
    try:
        import pymupdf  # type: ignore

        with pymupdf.open(stream=file_bytes, filetype="pdf") as doc:
            for page in doc:
                page_text = page.get_text()
                if page_text and page_text.strip():
                    text_content.append(page_text.strip())
    except Exception as exc:
        logger.warning(
            f"PyMuPDF extraction failed or encountered error: {exc}. Trying pypdf fallback..."
        )
        try:
            import pypdf

            reader = pypdf.PdfReader(BytesIO(file_bytes))
            for page in reader.pages:
                extracted = page.extract_text()
                if extracted and extracted.strip():
                    text_content.append(extracted.strip())
        except Exception as e2:
            raise ValueError(f"Failed to extract text from PDF: {e2}") from e2

    # If PyMuPDF returned no text, also try pypdf fallback before erroring
    if not text_content:
        try:
            import pypdf

            reader = pypdf.PdfReader(BytesIO(file_bytes))
            for page in reader.pages:
                extracted = page.extract_text()
                if extracted and extracted.strip():
                    text_content.append(extracted.strip())
        except Exception:
            pass

    extracted_text = "\n\n".join(text_content).strip()
    if not extracted_text:
        raise ValueError("The provided PDF does not contain extractable text.")
    return extracted_text


def chunk_text(text: str, chunk_size: int = 500, overlap: int = 50) -> list[str]:
    """Splits the text into semantic chunks of ~500 tokens with 50-token overlap."""
    if not text or not text.strip():
        return []

    try:
        tokenizer = tiktoken.get_encoding("cl100k_base")
    except Exception:
        tokenizer = tiktoken.encoding_for_model("gpt-4o-mini")

    clean_text = text.strip()
    paragraphs = [p.strip() for p in clean_text.split("\n\n") if p.strip()]
    if not paragraphs:
        paragraphs = [clean_text]

    chunks: list[str] = []
    current_tokens: list[int] = []

    for paragraph in paragraphs:
        p_tokens = tokenizer.encode(paragraph)

        # Handle paragraph longer than chunk size
        if len(p_tokens) > chunk_size:
            if current_tokens:
                chunks.append(tokenizer.decode(current_tokens).strip())
                current_tokens = (
                    current_tokens[-overlap:] if overlap < len(current_tokens) else []
                )

            step = max(1, chunk_size - overlap)
            for i in range(0, len(p_tokens), step):
                chunk_slice = p_tokens[i : i + chunk_size]
                chunk_str = tokenizer.decode(chunk_slice).strip()
                if chunk_str:
                    chunks.append(chunk_str)
            current_tokens = []
            continue

        if len(current_tokens) + len(p_tokens) <= chunk_size:
            current_tokens.extend(p_tokens)
        else:
            if current_tokens:
                chunks.append(tokenizer.decode(current_tokens).strip())
                current_tokens = (
                    current_tokens[-overlap:] if overlap < len(current_tokens) else []
                )
            current_tokens.extend(p_tokens)

    if current_tokens:
        last_chunk = tokenizer.decode(current_tokens).strip()
        if last_chunk and (not chunks or last_chunk != chunks[-1]):
            chunks.append(last_chunk)

    if not chunks:
        all_tokens = tokenizer.encode(clean_text)
        step = max(1, chunk_size - overlap)
        for i in range(0, len(all_tokens), step):
            chunk_slice = all_tokens[i : i + chunk_size]
            chunk_str = tokenizer.decode(chunk_slice).strip()
            if chunk_str:
                chunks.append(chunk_str)

    return chunks


def generate_embeddings(
    chunks: list[str], client: OpenAI | None = None
) -> list[list[float]]:
    """Calls the OpenAI Embeddings API to vectorize the chunks."""
    if not chunks:
        return []

    if client is None:
        if not settings.OPENAI_API_KEY:
            raise ValueError(
                "OPENAI_API_KEY is not configured. Please set OPENAI_API_KEY in your environment or .env file."
            )
        client = OpenAI(api_key=settings.OPENAI_API_KEY)

    batch_size = 128
    all_embeddings: list[list[float]] = []

    for i in range(0, len(chunks), batch_size):
        batch = chunks[i : i + batch_size]
        response = client.embeddings.create(
            input=batch,
            model=settings.OPENAI_EMBEDDING_MODEL,
        )
        sorted_data = sorted(response.data, key=lambda x: x.index)
        for item in sorted_data:
            all_embeddings.append(item.embedding)

    return all_embeddings


def store_chunks_in_db(
    chunks: list[str],
    embeddings: list[list[float]],
    doc_id: int,
    session: Session | None = None,
) -> list[DocumentChunk]:
    """Saves chunks and embeddings to PostgreSQL."""
    if len(chunks) != len(embeddings):
        raise ValueError(
            f"Mismatch between chunks count ({len(chunks)}) and embeddings count ({len(embeddings)})"
        )

    def _store(s: Session) -> list[DocumentChunk]:
        created_chunks: list[DocumentChunk] = []
        for text_chunk, emb in zip(chunks, embeddings, strict=True):
            chunk = DocumentChunk(
                document_id=doc_id,
                chunk_text=text_chunk,
                embedding=emb,
            )
            s.add(chunk)
            created_chunks.append(chunk)
        s.commit()
        for chunk in created_chunks:
            s.refresh(chunk)
        return created_chunks

    if session is not None:
        return _store(session)
    else:
        with Session(engine) as new_session:
            return _store(new_session)


def hybrid_search(
    query_text: str,
    document_id: int | None = None,
    top_k: int = 5,
    session: Session | None = None,
    client: OpenAI | None = None,
) -> list[DocumentChunk]:
    """Vectorizes the query and performs a cosine similarity search against the DocumentChunk table to return the top 5 most relevant chunks."""
    if not query_text or not query_text.strip():
        return []

    query_embeddings = generate_embeddings([query_text], client=client)
    if not query_embeddings:
        return []
    query_vector = query_embeddings[0]

    def _search(s: Session) -> list[DocumentChunk]:
        cosine_dist = DocumentChunk.embedding.cosine_distance(query_vector)
        stmt = select(DocumentChunk, cosine_dist.label("distance"))

        if document_id is not None:
            stmt = stmt.where(DocumentChunk.document_id == document_id)

        stmt = stmt.order_by(cosine_dist).limit(top_k)
        results = s.exec(stmt).all()

        chunks: list[DocumentChunk] = []
        for row in results:
            if isinstance(row, tuple):
                chunk = row[0]
                dist = row[1]
                chunk._score = 1.0 - float(dist) if dist is not None else 0.0
                chunks.append(chunk)
            else:
                chunks.append(row)
        return chunks

    if session is not None:
        return _search(session)
    else:
        with Session(engine) as new_session:
            return _search(new_session)


def generate_answer(
    query: str,
    context_chunks: list[DocumentChunk],
    client: OpenAI | None = None,
) -> tuple[str, list[str]]:
    """Constructs a prompt injecting retrieved context, calls OpenAI ChatCompletion (gpt-4o-mini),
    and returns the generated response alongside exact citation chunks."""
    if client is None:
        if not settings.OPENAI_API_KEY:
            raise ValueError(
                "OPENAI_API_KEY is not configured. Please set OPENAI_API_KEY in your environment or .env file."
            )
        client = OpenAI(api_key=settings.OPENAI_API_KEY)

    citations: list[str] = [c.chunk_text for c in context_chunks]

    formatted_context = "\n\n---\n\n".join(
        [
            f"[Citation {i + 1}]:\n{chunk.chunk_text}"
            for i, chunk in enumerate(context_chunks)
        ]
    )

    system_prompt = (
        "You are an expert financial analyst and RAG assistant specializing in financial document analysis. "
        "Answer the user's query truthfully and accurately, strictly based on the provided context excerpts below. "
        "If the answer cannot be determined from the provided context, state that clearly without speculating or hallucinating. "
        "Cite specific figures, dates, percentages, and metrics from the context whenever available."
    )

    user_prompt = (
        f"Context excerpts from financial documents:\n{formatted_context}\n\n"
        f"User Query: {query}\n\n"
        "Provide a detailed, grounded response based only on the excerpts above."
    )

    response = client.chat.completions.create(
        model=settings.OPENAI_CHAT_MODEL,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0.1,
    )

    answer = response.choices[0].message.content or ""
    return answer, citations
