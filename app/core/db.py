from sqlalchemy import text
from sqlmodel import Session, SQLModel, create_engine, select

from app import crud
from app.core.config import settings
from app.models import DocumentChunk, FinancialDocument, User, UserCreate  # noqa: F401

engine = create_engine(str(settings.DATABASE_URL), pool_pre_ping=True)


# make sure all SQLModel models are imported (app.models) before initializing DB
# otherwise, SQLModel might fail to initialize relationships properly
# for more details: https://github.com/fastapi/full-stack-fastapi-template/issues/28


def init_db(session: Session) -> None:
    # Ensure pgvector extension is enabled on PostgreSQL
    try:
        session.exec(text("CREATE EXTENSION IF NOT EXISTS vector"))
        session.commit()
    except Exception:
        session.rollback()

    # Create tables if not already created
    bind_engine = session.bind if session.bind is not None else engine
    SQLModel.metadata.create_all(bind_engine)

    user = session.exec(
        select(User).where(User.email == settings.FIRST_SUPERUSER)
    ).first()
    if not user:
        user_in = UserCreate(
            email=settings.FIRST_SUPERUSER,
            password=settings.FIRST_SUPERUSER_PASSWORD,
            is_superuser=True,
        )
        user = crud.create_user(session=session, user_create=user_in)
