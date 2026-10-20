"""Bounded Boolean readout validation shared by curation and handoff consumers."""
import re
from pathlib import Path

NAME = r"[A-Za-z][A-Za-z_0-9]*"


def validate_boolean_rule(rule, core):
    """A deliberately small Boolean grammar, validated without engine execution."""
    tokens = re.findall(NAME + r"|[01]|[!&|()]", rule)
    if re.sub(r"\s+", "", rule) != "".join(tokens):
        raise ValueError("Use existing core names, 0/1, !, &, | and parentheses")
    position = 0

    def atom():
        nonlocal position
        if position >= len(tokens):
            raise ValueError("Incomplete Boolean rule")
        token = tokens[position]
        position += 1
        if token == "!":
            atom()
        elif token == "(":
            expression()
            if position >= len(tokens) or tokens[position] != ")":
                raise ValueError("Unbalanced parentheses")
            position += 1
        elif token not in ("0", "1") and token not in core:
            raise ValueError(f"Readout regulators must be existing core nodes: {token}")

    def expression():
        nonlocal position
        atom()
        while position < len(tokens) and tokens[position] in ("&", "|"):
            position += 1
            atom()

    expression()
    if position != len(tokens):
        raise ValueError("Invalid Boolean rule syntax")


def validate_definitions(definitions, nodes):
    core = set(nodes) - set(definitions)
    for name, definition in definitions.items():
        if not re.fullmatch(NAME, name) or name.lower() in {"true", "false", "not", "or", "and", "xor", "node"}:
            raise ValueError("Invalid readout name")
        if set(definition) != {"origin", "rule", "explanation"} or definition["origin"] != "readout":
            raise ValueError("Readouts require explicit origin, exact rule and explanation")
        for field, maximum in (("rule", 2048), ("explanation", 2000)):
            value = definition[field]
            if not isinstance(value, str) or not value.strip() or len(value) > maximum:
                raise ValueError("Invalid readout " + field)
        validate_boolean_rule(definition["rule"], core)


def verify_bnd_readouts(path, definitions, nodes):
    if not definitions:
        return
    validate_definitions(definitions, nodes)
    text = re.sub(r"/\*.*?\*/", "", Path(path).read_text(), flags=re.S)
    text = "\n".join(line.split("//", 1)[0].split("#", 1)[0] for line in text.splitlines())
    bodies = dict(re.findall(r"(?i:node)\s+(" + NAME + r")\s*\{([^{}]*)\}", text))
    if set(bodies) != set(nodes):
        raise ValueError("Cannot verify readout isolation in these BND declarations")
    for name, definition in definitions.items():
        logic = re.search(r"\blogic\s*=\s*([^;]+);", bodies[name])
        if logic is None or re.sub(r"\s", "", logic[1]) != re.sub(r"\s", "", definition["rule"]):
            raise ValueError("Readout definition differs from BND logic: " + name)
    for name in set(nodes) - set(definitions):
        refs = set(re.findall(r"(?<![$@A-Za-z_0-9])" + NAME, bodies[name]))
        if refs & set(definitions):
            raise ValueError("Readout feedback into core BND node: " + name)
