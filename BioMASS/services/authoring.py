"""Deterministic rendering and validation of the accepted Text2Model language."""

from __future__ import annotations

import ast
import keyword
import math
import re

from ..contracts import ModelConfiguration, ReactionRecord

NAME = re.compile(r"^[A-Za-z][A-Za-z0-9_]*$")
NUMBER = re.compile(r"^[+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?$")


def identifier(value: str) -> None:
    if not NAME.fullmatch(value) or keyword.iskeyword(value) or "__" in value:
        raise ValueError(
            f"Invalid model identifier: {value!r}. Use letters, digits, and single underscores."
        )


def expression(value: str, roots: tuple[str, ...] = ("p", "u")) -> None:
    """Accept only arithmetic, finite numbers, and model-symbol subscripts."""
    if len(value) > 4000:
        raise ValueError("Expression exceeds 4000 characters.")
    tree = ast.parse(value.replace("^", "**").strip(), mode="eval")

    def visit(node: ast.AST) -> None:
        if isinstance(node, ast.Expression):
            visit(node.body)
        elif isinstance(node, ast.Constant) and type(node.value) in (int, float):
            if not math.isfinite(node.value):
                raise ValueError("Expression constants must be finite.")
        elif isinstance(node, ast.BinOp) and isinstance(
            node.op, (ast.Add, ast.Sub, ast.Mult, ast.Div, ast.Pow)
        ):
            visit(node.left)
            visit(node.right)
        elif isinstance(node, ast.UnaryOp) and isinstance(
            node.op, (ast.UAdd, ast.USub)
        ):
            visit(node.operand)
        elif (
            isinstance(node, ast.Subscript)
            and isinstance(node.value, ast.Name)
            and node.value.id in roots
            and isinstance(node.slice, ast.Name)
        ):
            identifier(node.slice.id)
        else:
            raise ValueError(
                "Expressions allow only arithmetic and p[name]/u[name]/init[name] references; Python calls and attributes are not supported."
            )

    visit(tree)


def assignments(value: str) -> None:
    for item in value.split(";"):
        match = re.fullmatch(r"\s*(p|init)\[([A-Za-z][A-Za-z0-9_]*)\]\s*=\s*(.+)", item)
        if not match:
            raise ValueError("Conditions require p[name] or init[name] assignments.")
        identifier(match[2])
        expression(match[3], ("p", "init"))


def validate_text(text: str) -> None:
    if not text.strip() or len(text.encode()) > 1024 * 1024:
        raise ValueError("Provide a nonempty Text2Model description of at most 1 MiB.")
    lines = text.splitlines()
    if len(lines) > 2000:
        raise ValueError("Descriptions are limited to 2000 lines.")
    condition_names: set[str] = set()
    observables: set[str] = set()
    reaction_lines: set[int] = set()
    times = 0
    for number, raw in enumerate(lines, 1):
        line = raw.split("#", 1)[0].strip()
        if not line:
            continue
        try:
            if line.startswith("@obs "):
                name, expr = line[5:].split(":", 1)
                identifier(name.strip())
                if name.strip() in observables:
                    raise ValueError("Duplicate observable name.")
                observables.add(name.strip())
                expression(expr)
            elif line.startswith("@sim "):
                name, expr = line[5:].split(":", 1)
                if name == "tspan":
                    times += 1
                    span = ast.literal_eval(expr.strip())
                    if (
                        not isinstance(span, list)
                        or len(span) != 2
                        or any(type(v) is not int for v in span)
                    ):
                        raise ValueError("Time span requires two integers.")
                    ModelConfiguration(time_span=tuple(span))
                    if times > 1:
                        raise ValueError("Only one time span is allowed.")
                elif name == "unperturbed":
                    assignments(expr)
                elif name.startswith("condition "):
                    label = name[10:].strip()
                    identifier(label)
                    if label in condition_names or len(condition_names) >= 20:
                        raise ValueError("Conditions must be unique (maximum 20).")
                    condition_names.add(label)
                    assignments(expr)
                else:
                    raise ValueError("Unknown simulation directive.")
            elif line.startswith("@add "):
                kind, name = line[5:].split(" ", 1)
                if kind not in ("param", "species"):
                    raise ValueError("Unknown @add directive.")
                identifier(name.strip())
            else:
                sections = line.split("|")
                if len(sections) > 3:
                    raise ValueError(
                        "A reaction has at most three pipe-separated sections."
                    )
                reaction = sections[0]
                if reaction.startswith("@rxn "):
                    reaction, expr = reaction[5:].split(":", 1)
                    expression(expr)
                elif reaction.startswith("@"):
                    raise ValueError("Unknown directive.")
                if not re.fullmatch(r"[A-Za-z0-9_ +<>=\-(),.&∅⇄→↔\t]+", reaction):
                    raise ValueError("Unsupported characters in reaction description.")
                if len(sections) > 1 and sections[1].strip():
                    params = sections[1].strip()
                    if params.isdecimal():
                        if int(params) not in reaction_lines:
                            raise ValueError(
                                "Parameter sharing must reference an earlier reaction line."
                            )
                    else:
                        numeric_assignments(params, "const ")
                if len(sections) > 2 and sections[2].strip():
                    numeric_assignments(sections[2], "fixed ")
                reaction_lines.add(number)
        except (ValueError, SyntaxError, TypeError) as exc:
            raise ValueError(f"Line {number}: {exc}") from exc
    if not reaction_lines:
        raise ValueError("Description contains no reactions.")


def numeric_assignments(text: str, prefix: str) -> None:
    for item in text.split(","):
        name, value = item.strip().removeprefix(prefix).split("=", 1)
        identifier(name.strip())
        if not NUMBER.fullmatch(value.strip()) or not math.isfinite(float(value)):
            raise ValueError(
                "Reaction parameter/initial values must be finite nonnegative numbers."
            )


def render_records(
    records: list[ReactionRecord], config: ModelConfiguration
) -> tuple[str, dict[str, int]]:
    lines: list[str] = []
    mapping: dict[str, int] = {}
    for record in records:
        if record.reaction_id in mapping:
            raise ValueError("Reaction IDs must be unique.")
        statement, separator, comment = record.statement.partition("#")
        statement = statement.strip()
        if not statement:
            raise ValueError("A reaction record cannot be blank or a comment.")
        if statement.startswith("@") and not statement.startswith("@rxn "):
            raise ValueError(
                "Reaction records cannot contain observable or simulation directives."
            )
        parts = statement.split("|")
        if len(parts) > 1 and parts[1].strip().isdecimal():
            raise ValueError(
                "Use share_parameters_with instead of numeric line references in reaction records."
            )
        if record.share_parameters_with:
            if record.share_parameters_with not in mapping:
                raise ValueError(
                    "Shared parameters must refer to an earlier reaction ID."
                )
            if len(parts) > 1 and parts[1].strip():
                raise ValueError(
                    "Parameter values and parameter sharing are mutually exclusive."
                )
            parts += [""] * (2 - len(parts))
            parts[1] = str(mapping[record.share_parameters_with])
            statement = " | ".join(parts)
        mapping[record.reaction_id] = len(lines) + 1
        lines.append(statement + (" #" + comment if separator else ""))
    for name, expr in config.observables.items():
        lines.append(f"@obs {name}: {expr}")
    if config.time_span:
        lines.append(f"@sim tspan: {list(config.time_span)}")
    for condition in config.conditions:
        changes = [f"p[{k}] = {v.value!r}" for k, v in condition.parameters.items()]
        changes += [f"init[{k}] = {v.value!r}" for k, v in condition.initials.items()]
        if not changes:
            raise ValueError(
                "Each condition needs at least one parameter or initial-value assignment."
            )
        lines.append(f"@sim condition {condition.name}: " + "; ".join(changes))
    text = "\n".join(lines) + "\n"
    validate_text(text)
    return text, mapping
