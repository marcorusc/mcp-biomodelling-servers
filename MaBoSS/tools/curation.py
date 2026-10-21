"""Explicitly authorized, bounded edits to stochastic summary readouts."""
from typing import Annotated

from pydantic import Field

from ..app import tool
from ..contracts import IDEMPOTENT, NonEmptyString
from ..locking import session_locked
from ..services.model import ReadoutEdit, ReadoutEditResult
from ..services.model import edit_readouts as apply_readouts
from ..session_manager import ensure_session


@tool(title="Edit MaBoSS summary readouts", annotations=IDEMPOTENT, structured_output=True)
@session_locked
def edit_maboss_readouts(
    edits: Annotated[list[ReadoutEdit], Field(min_length=1, max_length=20, description="Explicitly approved readout additions or rule changes.")],
    expected_revision: Annotated[NonEmptyString, Field(description="Exact model_revision returned by inspection.")],
    session_id: Annotated[NonEmptyString | None, Field(description="Session to edit; omit for the active session.")] = None,
) -> ReadoutEditResult:
    """Add/update explicitly approved summary nodes; regulators must be core nodes.

    Inspect get_maboss_nodes first for model_revision. Creation requires explicit
    rates and initial probability. This records changes, not scientific approval.
    """
    session = ensure_session(session_id)
    if session.sim is None:
        raise ValueError("Load a model first")
    return ReadoutEditResult(**apply_readouts(session, edits, expected_revision))
