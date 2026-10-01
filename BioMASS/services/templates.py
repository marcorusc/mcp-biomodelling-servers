"""Canonical construction templates shared by the tool and its resource."""

from __future__ import annotations

from dataclasses import dataclass

from ..contracts import ReactionEdit
from .authoring import identifier


@dataclass(frozen=True)
class Template:
    required: tuple[str, ...]
    optional: tuple[str, ...]
    sentence: str
    parameters: tuple[str, ...]
    reversible: bool = False


TEMPLATES = {
    "binding": Template(
        ("left", "right", "complex"),
        (),
        "{left} binds {right} {arrow} {complex}",
        ("kf",),
        True,
    ),
    "dissociation": Template(
        ("complex", "left", "right"),
        (),
        "{complex} {arrow} {left} + {right}",
        ("kf",),
        True,
    ),
    "dimerization": Template(
        ("monomer", "dimer"), (), "{monomer} dimerizes {arrow} {dimer}", ("kf",), True
    ),
    "conversion": Template(
        ("substrate", "product"), (), "{substrate} {arrow} {product}", ("kf",), True
    ),
    "phosphorylation": Template(
        ("enzyme", "substrate", "product"),
        (),
        "{enzyme} phosphorylates {substrate} --> {product}",
        ("V", "K"),
    ),
    "dephosphorylation": Template(
        ("enzyme", "substrate", "product"),
        (),
        "{enzyme} dephosphorylates {substrate} --> {product}",
        ("V", "K"),
    ),
    "transcription": Template(
        ("regulator", "product"),
        (),
        "{regulator} transcribes {product}",
        ("V", "K", "n"),
    ),
    "synthesis": Template(
        ("product",), ("regulator",), "{product} is synthesized", ("kf",)
    ),
    "degradation": Template(
        ("substrate",), ("regulator",), "{substrate} is degraded", ("kf",)
    ),
    "transport": Template(
        ("source", "destination"),
        (),
        "{source} translocates {arrow} {destination}",
        ("kf",),
        True,
    ),
}


def template_statement(edit: ReactionEdit, mapping: dict[str, list[str]]) -> str:
    if edit.statement is not None:
        return edit.statement
    spec = TEMPLATES[edit.template]
    roles = set(edit.participants)
    missing, extra = (
        set(spec.required) - roles,
        roles - set(spec.required + spec.optional),
    )
    if missing or extra:
        raise ValueError(
            f"{edit.template}: missing participants {sorted(missing)}; unexpected participants {sorted(extra)}."
        )
    if edit.reversible and not spec.reversible:
        raise ValueError(
            f"{edit.template} does not have a reversible form; supply an explicit statement."
        )
    names = {}
    for role, value in edit.participants.items():
        if value in mapping:
            if len(mapping[value]) != 1:
                raise ValueError(
                    f"Node {value} maps to multiple states; specify a species name explicitly."
                )
            value = mapping[value][0]
        identifier(value)
        names[role] = value
    parameters = set(spec.parameters) | ({"kr"} if edit.reversible else set())
    if set(edit.parameters) - parameters:
        raise ValueError(
            f"{edit.template} parameters are {sorted(parameters)} (local names, without line suffixes)."
        )
    sentence = spec.sentence
    if "regulator" in names:
        if edit.template == "synthesis":
            sentence = "{regulator} synthesizes {product}"
        elif edit.template == "degradation":
            sentence = "{regulator} degrades {substrate}"
    return sentence.format(**names, arrow="<-->" if edit.reversible else "-->")


def template_reference() -> str:
    lines = [
        "| Template | Required participants | Optional participants | Local parameters |",
        "|---|---|---|---|",
    ]
    for name, spec in TEMPLATES.items():
        params = ", ".join(spec.parameters) + (
            "; kr when reversible" if spec.reversible else ""
        )
        lines.append(
            f"| `{name}` | {', '.join(spec.required)} | {', '.join(spec.optional) or 'none'} | {params} |"
        )
    return "\n".join(lines)
