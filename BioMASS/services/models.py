"""Authoring state transitions, coverage, and reference validation."""

from __future__ import annotations

from ..outputs import Coverage, ModelDocument
from .authoring import render_records, validate_text


def coverage(document: ModelDocument) -> Coverage:
    all_edges = (
        {e.edge_id for e in document.network.edges} if document.network else set()
    )
    supported = {
        e for r in document.reactions if r.status == "supported" for e in r.edge_ids
    }
    assumed = {
        e for r in document.reactions if r.status == "assumed" for e in r.edge_ids
    }
    return Coverage(
        total_edges=len(all_edges),
        supported_edges=sorted(supported),
        assumed_edges=sorted(assumed),
        unresolved_edges=sorted(all_edges - supported - assumed),
        assumed_reactions=[
            r.reaction_id for r in document.reactions if r.status == "assumed"
        ],
        conflicting_evidence=[
            e.evidence_id
            for e in document.evidence.values()
            if e.stance == "contradicts"
        ],
    )


def validate_links(document: ModelDocument) -> None:
    edges = {e.edge_id for e in document.network.edges} if document.network else set()
    for evidence in document.evidence.values():
        if set(evidence.edge_ids) - edges:
            raise ValueError("Evidence refers to unknown network edges.")
    for record in document.reactions:
        if (
            set(record.edge_ids) - edges
            or set(record.evidence_ids) - document.evidence.keys()
        ):
            raise ValueError("Reaction refers to unknown edges or evidence.")
        if record.status == "supported" and not any(
            document.evidence[e].stance == "supports" for e in record.evidence_ids
        ):
            raise ValueError(
                "Supported reactions need at least one supporting evidence record."
            )
    for link in document.line_evidence:
        if (
            link.line_number > len((document.text or "").splitlines())
            or set(link.evidence_ids) - document.evidence.keys()
        ):
            raise ValueError("Document provenance refers to unknown lines or evidence.")
    quantities = [
        *document.configuration.parameters.values(),
        *document.configuration.initials.values(),
    ]
    for condition in document.configuration.conditions:
        quantities.extend(condition.parameters.values())
        quantities.extend(condition.initials.values())
    if any(set(q.evidence_ids) - document.evidence.keys() for q in quantities):
        raise ValueError("Quantity refers to unknown evidence.")


def render(document: ModelDocument) -> tuple[str, dict[str, int]]:
    validate_links(document)
    if document.mode == "records":
        return render_records(document.reactions, document.configuration)
    if document.mode == "document" and document.text is not None:
        validate_text(document.text)
        return document.text, {
            f"line_{n}": n
            for n, line in enumerate(document.text.splitlines(), 1)
            if line.strip()
            and not line.lstrip().startswith(("#", "@obs", "@sim", "@add"))
        }
    raise ValueError("Import a network or Text2Model description first.")


def edited(document: ModelDocument) -> ModelDocument:
    validate_links(document)
    document.version += 1
    document.current_revision = None
    return document
