"""phase10_reports

Revision ID: 0003_phase10_reports
Revises: 0002_phase8_experiments
Create Date: 2026-08-16 00:00:00.000000

"""
from typing import Sequence, Union
from alembic import op
import sqlalchemy as sa


revision: str = '0003_phase10_reports'
down_revision: Union[str, None] = '0002_phase8_experiments'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        'publication_reports',
        sa.Column('id', sa.String(length=36), nullable=False),
        sa.Column('experiment_id', sa.String(length=36), nullable=False),
        sa.Column('name', sa.String(length=255), nullable=False),
        sa.Column('status', sa.Enum('PENDING', 'RUNNING', 'SUCCESS', 'FAILED', 'CANCELLED', name='jobstatus'), nullable=False),
        sa.Column('configuration', sa.JSON(), nullable=False),
        sa.Column('generated_artifact_path', sa.String(length=512), nullable=True),
        sa.Column('report_metadata', sa.JSON(), nullable=True),
        sa.Column('user_id', sa.Integer(), nullable=True),
        sa.Column('created_at', sa.DateTime(timezone=True), nullable=False),
        sa.Column('updated_at', sa.DateTime(timezone=True), nullable=False),
        sa.ForeignKeyConstraint(['experiment_id'], ['experiments.id'], ondelete='CASCADE'),
        sa.ForeignKeyConstraint(['user_id'], ['users.id'], ondelete='SET NULL'),
        sa.PrimaryKeyConstraint('id')
    )
    op.create_index(op.f('ix_publication_reports_id'), 'publication_reports', ['id'], unique=False)
    op.create_index(op.f('ix_publication_reports_experiment_id'), 'publication_reports', ['experiment_id'], unique=False)


def downgrade() -> None:
    op.drop_index(op.f('ix_publication_reports_experiment_id'), table_name='publication_reports')
    op.drop_index(op.f('ix_publication_reports_id'), table_name='publication_reports')
    op.drop_table('publication_reports')
