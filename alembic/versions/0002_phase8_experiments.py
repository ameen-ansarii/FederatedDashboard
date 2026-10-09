"""phase8_experiments

Revision ID: 0002_phase8_experiments
Revises: 0001_initial_schema
Create Date: 2026-08-15 00:00:00.000000

"""
from typing import Sequence, Union
from alembic import op
import sqlalchemy as sa


revision: str = '0002_phase8_experiments'
down_revision: Union[str, None] = '0001_initial_schema'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # 1. Create experiments table
    op.create_table(
        'experiments',
        sa.Column('id', sa.String(length=36), nullable=False),
        sa.Column('name', sa.String(length=255), nullable=False),
        sa.Column('description', sa.Text(), nullable=True),
        sa.Column('status', sa.Enum('PENDING', 'RUNNING', 'SUCCESS', 'FAILED', 'CANCELLED', name='jobstatus'), nullable=False),
        sa.Column('artifact_id', sa.Integer(), nullable=False),
        sa.Column('config', sa.JSON(), nullable=False),
        sa.Column('results', sa.JSON(), nullable=True),
        sa.Column('user_id', sa.Integer(), nullable=True),
        sa.Column('created_at', sa.DateTime(timezone=True), nullable=False),
        sa.Column('updated_at', sa.DateTime(timezone=True), nullable=False),
        sa.ForeignKeyConstraint(['artifact_id'], ['preprocessing_artifacts.id'], ondelete='CASCADE'),
        sa.ForeignKeyConstraint(['user_id'], ['users.id'], ondelete='SET NULL'),
        sa.PrimaryKeyConstraint('id')
    )
    op.create_index(op.f('ix_experiments_id'), 'experiments', ['id'], unique=False)

    # 2. Add experiment_id column to training_runs
    op.add_column('training_runs', sa.Column('experiment_id', sa.String(length=36), nullable=True))
    op.create_index(op.f('ix_training_runs_experiment_id'), 'training_runs', ['experiment_id'], unique=False)
    op.create_foreign_key('fk_training_runs_experiment_id', 'training_runs', 'experiments', ['experiment_id'], ['id'], ondelete='SET NULL')


def downgrade() -> None:
    op.drop_constraint('fk_training_runs_experiment_id', 'training_runs', type_='foreignkey')
    op.drop_index(op.f('ix_training_runs_experiment_id'), table_name='training_runs')
    op.drop_column('training_runs', 'experiment_id')
    op.drop_index(op.f('ix_experiments_id'), table_name='experiments')
    op.drop_table('experiments')
