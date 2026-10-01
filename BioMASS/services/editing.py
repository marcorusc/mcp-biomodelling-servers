"""Mechanical, versioned model edits without scientific inference."""

from __future__ import annotations

import re
import shutil
from pathlib import Path

from ..contracts import (
    DocumentLineEdit,
    ModelConfiguration,
    ReactionEdit,
    ReactionRecord,
)
from ..outputs import ModelDocument, ReactionInventoryItem
from . import artifacts
from .authoring import identifier, validate_text
from .models import render, validate_links
from .templates import template_statement


def is_reaction(raw: str) -> bool:
    line = raw.split("#", 1)[0].strip()
    return bool(line) and (not line.startswith("@") or line.startswith("@rxn "))


def text_and_inventory(
    document: ModelDocument,
) -> tuple[str, list[ReactionInventoryItem]]:
    if document.mode == "empty":
        return "", []
    if document.mode == "records" and not document.reactions:
        # Empty drafts are editable, but not executable models.
        text = "# Empty reaction model\n"
        return text, []
    text = document.text if document.mode == "document" else render(document)[0]
    by_line = {line: rid for rid, line in document.reaction_lines.items()}
    if document.mode == "records" and not by_line:
        by_line = {n: r.reaction_id for n, r in enumerate(document.reactions, 1)}
    inventory = [
        ReactionInventoryItem(
            reaction_id=by_line.get(n, f"line_{n}"),
            line_number=n,
            statement=raw.rstrip("\r\n"),
        )
        for n, raw in enumerate((text or "").splitlines(keepends=True), 1)
        if is_reaction(raw)
    ]
    return text or "", inventory


def compile_structure(directory: Path, text: str, timeout: float) -> dict:
    """Discover reaction symbols without executing observables or conditions.

    Keep line numbers so the resulting parameter names match the real model.
    """
    structure = "".join(
        "# Directive omitted for symbol inspection\n"
        if raw.lstrip().startswith(("@obs ", "@sim "))
        else raw
        for raw in text.splitlines(keepends=True)
    )
    if not any(is_reaction(line) for line in structure.splitlines()):
        return {
            "species": re.findall(r"(?m)^\s*@add species (\w+)", structure),
            "parameters": dict.fromkeys(
                re.findall(r"(?m)^\s*@add param (\w+)", structure)
            ),
            "reactions": [],
        }
    path, summary = artifacts.job(
        directory,
        {"operation": "generate", "text": structure, "configuration": {}},
        timeout,
    )
    shutil.rmtree(path)
    return summary


def annotate_inventory(items: list[ReactionInventoryItem], summary: dict) -> None:
    parameters = {}
    for flux in summary.get("reactions", []):
        match = re.match(r"v\[(\d+)\]", flux)
        if match:
            parameters[int(match[1])] = sorted(set(re.findall(r"x\[C\.(\w+)\]", flux)))
    for item in items:
        item.parameters = parameters.get(item.line_number, [])


def dependencies(document: ModelDocument, text: str, summary: dict) -> list[str]:
    species, parameters = (
        set(summary.get("species", [])),
        set(summary.get("parameters", [])),
    )
    issues = []
    for number, raw in enumerate(text.splitlines(), 1):
        for root, name in re.findall(r"\b(p|u|init)\[(\w+)\]", raw.split("#", 1)[0]):
            if name not in (parameters if root == "p" else species):
                issues.append(
                    f"Line {number}: unknown {root}[{name}]. Edit or remove the dependent expression in the same batch."
                )
    cfg = document.configuration
    for name, expr in cfg.observables.items():
        for root, symbol in re.findall(r"\b(p|u)\[(\w+)\]", expr):
            if symbol not in (parameters if root == "p" else species):
                issues.append(
                    f"Observable {name}: unknown {root}[{symbol}]; update the configuration."
                )
    for location, values, known in (
        ("parameters", cfg.parameters, parameters),
        ("initials", cfg.initials, species),
    ):
        for name in set(values) - known:
            issues.append(
                f"configuration.{location}.{name} no longer exists; remove it explicitly or use prune_unused_values."
            )
    for condition in cfg.conditions:
        for kind, values, known in (
            ("parameters", condition.parameters, parameters),
            ("initials", condition.initials, species),
        ):
            for name in set(values) - known:
                issues.append(
                    f"Condition {condition.name}: unknown {kind} {name}; update the configuration."
                )
    return sorted(set(issues))


def species_mapping(
    document: ModelDocument, supplied: dict[str, list[str]] | None
) -> dict[str, list[str]]:
    mapping = {k: list(v) for k, v in document.species_mapping.items()}
    nodes = {n.node_id: n for n in document.network.nodes} if document.network else {}
    if supplied:
        if set(supplied) - nodes.keys():
            raise ValueError("Species mapping refers to unknown network nodes.")
        mapping.update(supplied)
    # Never guess scientific states: these defaults are identifier aliases only.
    for nid, node in nodes.items():
        if nid not in mapping:
            name = re.sub(r"[^A-Za-z0-9]+", "_", node.gene_symbol or nid).strip("_")
            if not name or not name[0].isalpha():
                name = "S_" + name
            mapping[nid] = [name]
    owners = {}
    for nid, names in mapping.items():
        if not names or len(names) != len(set(names)):
            raise ValueError(
                "Each node needs one or more distinct mapped species names."
            )
        for name in names:
            identifier(name)
            if name in owners and owners[name] != nid:
                raise ValueError(
                    f"Species mapping collision for {name}; supply distinct names for the original nodes."
                )
            owners[name] = nid
    for nid, old in document.species_mapping.items():
        if not set(old).issubset(mapping.get(nid, [])):
            raise ValueError(
                "Existing mapped species cannot be removed or renamed; extend the mapping with additional states or use a new session for a remapping."
            )
    return mapping


def candidate(
    original: ModelDocument,
    edits: list[ReactionEdit],
    line_edits: list[DocumentLineEdit],
    mapping: dict[str, list[str]] | None,
    configuration: ModelConfiguration | None,
    append_lines: list[str],
) -> tuple[ModelDocument, list[str], list[str]]:
    document = original.model_copy(deep=True)
    if document.mode == "empty":
        document.mode = "records"
    old_text, inventory = text_and_inventory(document)
    existing = {item.reaction_id: item for item in inventory}
    if len({edit.reaction_id for edit in edits}) != len(edits):
        raise ValueError("Each reaction can be edited only once per batch.")
    document.species_mapping = species_mapping(document, mapping)
    document.reaction_lines = {item.reaction_id: item.line_number for item in inventory}
    if document.mode == "records":
        document.record_line_count = max(
            document.record_line_count, max(document.reaction_lines.values(), default=0)
        )
    if configuration is not None:
        if document.mode == "document" and (
            configuration.observables
            or configuration.conditions
            or configuration.time_span
        ):
            raise ValueError(
                "Document structural directives belong in line_edits or append_lines."
            )
        document.configuration = configuration.model_copy(deep=True)
    records = {r.reaction_id: r for r in document.reactions}
    lines = old_text.splitlines(keepends=True) if document.mode == "document" else []
    notes, removed = [], []
    newline = "\r\n" if "\r\n" in old_text else "\n"
    for edit in edits:
        present = edit.reaction_id in existing
        if (edit.action == "add") == present:
            raise ValueError(
                f"{edit.action}: reaction {edit.reaction_id} {'already exists' if present else 'does not exist'}."
            )
        if edit.action == "remove":
            number = document.reaction_lines.pop(edit.reaction_id)
            records.pop(edit.reaction_id, None)
            if document.mode == "document":
                lines[number - 1] = f"# Removed reaction {edit.reaction_id}" + (
                    newline if lines[number - 1].endswith(("\n", "\r")) else ""
                )
                document.line_evidence = [
                    e for e in document.line_evidence if e.line_number != number
                ]
            removed.append(edit.reaction_id)
            continue
        if present:
            number = existing[edit.reaction_id].line_number
        elif document.mode == "document":
            if lines and not lines[-1].endswith(("\n", "\r")):
                lines[-1] += newline
            number = len(lines) + 1
            lines.append(newline)
        else:
            document.record_line_count += 1
            number = document.record_line_count
        statement = template_statement(edit, document.species_mapping)
        if not is_reaction(statement):
            raise ValueError(
                "Reaction edits require a reaction, not a comment or directive."
            )
        document.reaction_lines[edit.reaction_id] = number
        old = records.get(edit.reaction_id)
        metadata = (
            edit.metadata.model_dump()
            if edit.metadata is not None
            else (
                {
                    k: getattr(old, k)
                    for k in ("status", "assumption", "evidence_ids", "edge_ids")
                }
                if old
                else {}
            )
        )
        record = ReactionRecord(
            reaction_id=edit.reaction_id,
            statement=statement,
            share_parameters_with=edit.share_parameters_with,
            **metadata,
        )
        # Existing record sharing remains effective unless the replacement supplies
        # a new parameter section or explicitly selects a different source.
        if (
            old
            and old.share_parameters_with
            and edit.share_parameters_with is None
            and "|" not in statement
            and not edit.parameters
        ):
            record.share_parameters_with = old.share_parameters_with
        records[edit.reaction_id] = record
        for key, value in edit.parameters.items():
            generated = (
                key if statement.lstrip().startswith("@rxn ") else key + str(number)
            )
            document.configuration.parameters[generated] = value.model_copy(deep=True)
        document.configuration.initials.update(
            {k: v.model_copy(deep=True) for k, v in edit.initials.items()}
        )
        if document.mode == "document":
            if record.share_parameters_with:
                source = document.reaction_lines.get(record.share_parameters_with)
                if source is None or source >= number:
                    raise ValueError(
                        "Parameter sharing must target an earlier reaction."
                    )
                body, marker, comment = statement.partition("#")
                parts = body.split("|")
                if len(parts) > 1 and parts[1].strip():
                    raise ValueError(
                        "Parameter values and sharing are mutually exclusive."
                    )
                parts += [""] * (2 - len(parts))
                parts[1] = str(source)
                statement = " | ".join(parts)
                if marker:
                    statement += " #" + comment
            ending = (
                newline
                if not present or lines[number - 1].endswith(("\n", "\r"))
                else ""
            )
            lines[number - 1] = statement + ending
            if present and edit.metadata is not None:
                document.line_evidence = [
                    e for e in document.line_evidence if e.line_number != number
                ]
    document.reactions = sorted(
        records.values(), key=lambda r: document.reaction_lines[r.reaction_id]
    )
    if document.mode != "document" and (line_edits or append_lines):
        raise ValueError(
            "Use configuration for record-mode directives; line edits apply to imported documents."
        )
    if len({e.line_number for e in line_edits}) != len(line_edits):
        raise ValueError("Each document line can be edited only once per batch.")
    for edit in line_edits:
        if not 1 <= edit.line_number <= len(old_text.splitlines()):
            raise ValueError("Document line number is outside the original document.")
        if is_reaction(old_text.splitlines()[edit.line_number - 1]) or (
            edit.text and is_reaction(edit.text)
        ):
            raise ValueError(
                "Use reaction IDs to edit reactions; line_edits are for directives and comments."
            )
        ending = newline if lines[edit.line_number - 1].endswith(("\n", "\r")) else ""
        lines[edit.line_number - 1] = (
            edit.text if edit.text is not None else "# Removed directive"
        ) + ending
        document.line_evidence = [
            e for e in document.line_evidence if e.line_number != edit.line_number
        ]
    for line in append_lines:
        if "\n" in line or "\r" in line or is_reaction(line):
            raise ValueError(
                "append_lines takes individual directives/comments; use reaction edits for reactions."
            )
        if lines and not lines[-1].endswith(("\n", "\r")):
            lines[-1] += newline
        lines.append(line + newline)
    if document.mode == "document":
        document.text = "".join(lines)
    validate_links(document)
    text, items = text_and_inventory(document)
    if items:
        validate_text(text)
    if removed:
        notes.append(
            "Removed reaction lines remain comments; later parameter names and sharing references retain their line positions."
        )
    return document, removed, notes
