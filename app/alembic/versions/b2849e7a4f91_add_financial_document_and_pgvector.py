"""Add pgvector extension and financial document tables

Revision ID: b2849e7a4f91
Revises: 1a31ce608336
Create Date: 2026-10-17 18:30:00.000000

"""
from alembic import op
import sqlalchemy as sa
from pgvector.sqlalchemy import Vector
import sqlmodel


# revision identifiers, used by Alembic.
revision = 'b2849e7a4f91'
down_revision = 'fe56fa70289e'
branch_labels = None
depends_on = None


def upgrade():
    # Enable pgvector extension
    op.execute('CREATE EXTENSION IF NOT EXISTS vector;')

    # Create financialdocument table
    op.create_table(
        'financialdocument',
        sa.Column('id', sa.Integer(), nullable=False),
        sa.Column('filename', sqlmodel.sql.sqltypes.AutoString(length=255), nullable=False),
        sa.Column('upload_date', sa.DateTime(timezone=True), nullable=False),
        sa.Column('status', sqlmodel.sql.sqltypes.AutoString(length=50), nullable=False),
        sa.PrimaryKeyConstraint('id')
    )

    # Create documentchunk table
    op.create_table(
        'documentchunk',
        sa.Column('id', sa.Integer(), nullable=False),
        sa.Column('document_id', sa.Integer(), nullable=False),
        sa.Column('chunk_text', sa.Text(), nullable=False),
        sa.Column('embedding', Vector(1536), nullable=True),
        sa.ForeignKeyConstraint(['document_id'], ['financialdocument.id'], ondelete='CASCADE'),
        sa.PrimaryKeyConstraint('id')
    )


def downgrade():
    op.drop_table('documentchunk')
    op.drop_table('financialdocument')
    op.execute('DROP EXTENSION IF EXISTS vector;')
