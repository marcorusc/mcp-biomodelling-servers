"""Bounded, state-checked exposure of NeKo's validated interaction editor."""

from typing import Annotated, Literal

from pydantic import Field, model_validator

from mcp_biomodelling_servers.structured_outputs import StructuredOutputModel

from ..app import tool
from ..contracts import DESTRUCTIVE_NON_IDEMPOTENT_CLOSED, NonEmptyString
from ..src.helpers import _invalidate, requires_network

Text = Annotated[str, Field(min_length=1, max_length=2000, pattern=r".*\S.*")]
Identifier = Annotated[str, Field(min_length=1, max_length=256, pattern=r".*\S.*")]
Effect = Literal["stimulation", "inhibition", "bimodal", "form complex", "undefined"]


class InteractionEvidence(StructuredOutputModel):
    summary: Text
    pmids: list[Annotated[str, Field(pattern=r"^[1-9][0-9]{0,15}$")]] = Field(
        default_factory=list, max_length=20,
    )
    citations: list[Text] = Field(default_factory=list, max_length=10)

    @model_validator(mode="after")
    def require_support(self):
        if not self.pmids and not self.citations:
            raise ValueError("Supply a PMID or supporting publication citation.")
        return self


class InteractionEdit(StructuredOutputModel):
    operation: Literal["add", "set_effect"]
    source: Identifier
    target: Identifier
    effect: Effect
    rationale: Text
    evidence: InteractionEvidence


class InteractionChange(StructuredOutputModel):
    operation: Literal["add", "set_effect"]
    source: str
    target: str
    before_effect: str | None
    after_effect: str
    origin: str
    edit_origin: Literal["manual"] = "manual"


class InteractionEditResult(StructuredOutputModel):
    server: Literal["NeKo"] = "NeKo"
    session_id: str
    network_id: str
    changed: bool
    previous_state_id: int
    state_id: int
    unchanged_count: int
    changes: list[InteractionChange]
    details_uri: str


@tool(
    title="Edit interactions",
    annotations=DESTRUCTIVE_NON_IDEMPOTENT_CLOSED,
    structured_output=True,
)
@requires_network
def edit_interactions(
    edits: Annotated[list[InteractionEdit], Field(min_length=1, max_length=20, description="Authorized edits; the whole batch succeeds or fails.")],
    expected_network_id: Annotated[NonEmptyString, Field(description="Network identity returned by status.")],
    expected_state_id: Annotated[int, Field(ge=0, strict=True, description="Exact current state returned by status or the previous edit.")],
    session_id: Annotated[NonEmptyString, Field(description="Session owning the network.")],
    *, sess=None, network=None,
) -> InteractionEditResult:
    """Apply authorized add/set_effect edits atomically; no implicit nodes or export.

    Get network/state IDs from status. Evidence is required, not approval.
    Full before/after evidence is available through the returned history URI.
    """
    if expected_network_id != sess.network_id:
        raise ValueError("stale_network: the session's network has been replaced.")
    if not callable(getattr(network, "edit_interactions", None)):
        raise RuntimeError("This operation requires the NeKo Wave 2 curation API.")
    if expected_state_id != network.current_state_id:
        raise ValueError("stale_state: inspect the current network before editing.")
    try:
        result = network.edit_interactions(
            [edit.model_dump() for edit in edits], expected_state_id=expected_state_id,
        )
    except ValueError as exc:
        code = getattr(exc, "code", "invalid_request")
        index = getattr(exc, "edit_index", None)
        raise ValueError(f"{code} (edit {index}): {exc}") from exc
    if result["changed"]:
        _invalidate(sess)
    return InteractionEditResult(
        session_id=sess.session_id, network_id=sess.network_id,
        changed=result["changed"], previous_state_id=result["previous_state_id"],
        state_id=result["state_id"], unchanged_count=len(edits) - len(result["changes"]),
        changes=[InteractionChange(
            operation=change["operation"], source=change["source"], target=change["target"],
            before_effect=change["before"]["Effect"] if change["before"] else None,
            after_effect=change["after"]["Effect"],
            origin=change["after"]["Provenance"]["origin"],
        ) for change in result["changes"]],
        details_uri=f"neko://session/{sess.session_id}/history/{sess.network_id}/{result['state_id']}",
    )
