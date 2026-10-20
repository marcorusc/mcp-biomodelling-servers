"""Bounded model inspection and transactional readout/rate editing.

Only public pyMaBoSS model APIs are used. Revisions fingerprint effective BND,
CFG and readout provenance; they identify models, not scientific approval.
"""
from __future__ import annotations

import hashlib
import io
import json
import re
from copy import copy, deepcopy
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from mcp_biomodelling_servers.boolean_readouts import NAME, validate_boolean_rule
from mcp_biomodelling_servers.structured_outputs import StructuredOutputModel

PARAMETER = re.compile(r"\$" + NAME)
COEFFICIENT = r"(?:\$[A-Za-z_][A-Za-z_0-9]*|(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?)"


class RateEdit(BaseModel):
    model_config = ConfigDict(extra="forbid")
    node: str = Field(min_length=1, max_length=128)
    up: float | None = Field(default=None, ge=0, allow_inf_nan=False)
    down: float | None = Field(default=None, ge=0, allow_inf_nan=False)

    @model_validator(mode="after")
    def require_change(self):
        if self.up is None and self.down is None:
            raise ValueError("Supply up and/or down coefficient")
        return self


class ReadoutEdit(BaseModel):
    model_config = ConfigDict(extra="forbid")
    operation: Literal["add", "set_rule"]
    node: str = Field(pattern="^" + NAME + "$", max_length=128)
    rule: str = Field(min_length=1, max_length=2048)
    explanation: str = Field(min_length=1, max_length=2000)
    rate_up: float | None = Field(default=None, ge=0, allow_inf_nan=False)
    rate_down: float | None = Field(default=None, ge=0, allow_inf_nan=False)
    initial_probability_on: float | None = Field(default=None, ge=0, le=1, allow_inf_nan=False)

    @model_validator(mode="after")
    def explicit_creation_settings(self):
        if self.node.lower() in {"true", "false", "not", "or", "and", "xor", "node"}:
            raise ValueError("Readout name is reserved by MaBoSS")
        if not self.explanation.strip():
            raise ValueError("A readout explanation is required")
        values = (self.rate_up, self.rate_down, self.initial_probability_on)
        if self.operation == "add" and any(v is None for v in values):
            raise ValueError("Creation requires explicit rate_up, rate_down and initial_probability_on")
        if self.operation == "set_rule" and any(v is not None for v in values):
            raise ValueError("Use the rate and initial-state configuration interfaces to change those settings")
        return self


def model_revision(sim, readouts=None):
    bnd = io.StringIO()
    sim.print_bnd(out=bnd)
    cfg = io.StringIO()
    sim.print_cfg(out=cfg)
    raw = json.dumps([bnd.getvalue(), cfg.getvalue(), readouts or {}], sort_keys=True)
    return hashlib.sha256(raw.encode()).hexdigest()


def require_revision(session, expected):
    current = model_revision(session.sim, session.readouts)
    if expected != current:
        raise ValueError("stale_model_revision: inspect the active model before editing")
    return current


def inspect_nodes(sim, names, readouts=None):
    """Report expressions and referenced values, never instantaneous rates."""
    records = []
    for name in names:
        if name not in sim.network:
            raise ValueError(f"Unknown node: {name}")
        node = sim.network[name]
        expressions = {"up": str(node.rt_up), "down": str(node.rt_down)}
        refs = sorted(set(PARAMETER.findall(" ".join(expressions.values()))))
        records.append({"node": name, "rule": node.logExp,
                        "rule_limitation": None if node.logExp else "No explicit Boolean logic; custom transition rates determine behavior",
                        "rate_expressions": expressions,
                        "parameters": {ref: sim.param[ref] for ref in refs if ref in sim.param},
                        "unresolved_parameters": [ref for ref in refs if ref not in sim.param],
                        "output": name in sim.network.get_output(),
                        "origin": "readout" if name in (readouts or {}) else "core",
                        "readout": (readouts or {}).get(name)})
    return records


def replace_coefficient(expression, value, direction):
    """Accept the standard conditional form; never guess arbitrary expressions."""
    text = str(expression).strip()
    if re.fullmatch(COEFFICIENT, text):
        return repr(float(value))  # Already unconditional; no logical guard is removed.
    pattern = (rf"(@logic\s*\?\s*)({COEFFICIENT})(\s*:\s*0(?:\.0*)?)"
               if direction == "up" else
               rf"(@logic\s*\?\s*0(?:\.0*)?\s*:\s*)({COEFFICIENT})()")
    match = re.fullmatch(pattern, text)
    if not match:
        raise ValueError(f"Cannot unambiguously edit {direction} coefficient in {text!r}; expected standard @logic conditional")
    # Inline the coefficient rather than modifying a potentially shared parameter.
    return match[1] + repr(float(value)) + match[3]


def edit_rates(session, edits, expected_revision):
    before_revision = require_revision(session, expected_revision)
    names = [edit.node for edit in edits]
    if len(set(names)) != len(names):
        raise ValueError("Each node may appear only once")
    before = inspect_nodes(session.sim, names, session.readouts)
    candidate = copy(session.sim)
    candidate.network = session.sim.network.copy()
    for edit in edits:
        node = candidate.network[edit.node]
        for direction, value, attr in (("up", edit.up, "rt_up"), ("down", edit.down, "rt_down")):
            if value is not None:
                setattr(node, attr, replace_coefficient(getattr(node, attr), value, direction))
    result = {"previous_revision": before_revision, "model_revision": model_revision(candidate, session.readouts),
              "before": before, "after": inspect_nodes(candidate, names, session.readouts)}
    session.sim = candidate
    return result



def edit_readouts(session, edits, expected_revision):
    before = require_revision(session, expected_revision)
    names = [edit.node for edit in edits]
    if len(set(names)) != len(names):
        raise ValueError("Each readout may appear only once")
    core = set(session.sim.network) - set(session.readouts)
    candidate = copy(session.sim)
    candidate.network = session.sim.network.copy()
    definitions = deepcopy(session.readouts)
    for edit in edits:
        validate_boolean_rule(edit.rule, core)
        if edit.operation == "add":
            if edit.node in candidate.network:
                raise ValueError(f"Node already exists: {edit.node}")
            # Avoid turning a previously unresolved core reference into feedback.
            if any(edit.node in re.findall(NAME, str(node)) for node in session.sim.network.values()):
                raise ValueError("A core expression already references this name; readout feedback is prohibited")
            candidate.network.add_node(edit.node)
            node = candidate.network[edit.node]
            node.rt_up = f"@logic ? {edit.rate_up} : 0"
            node.rt_down = f"@logic ? 0 : {edit.rate_down}"
            candidate.network.set_istate(edit.node, [1 - edit.initial_probability_on, edit.initial_probability_on])
            node.is_internal = True  # preserve the explicitly selected output set
        elif edit.node not in definitions:
            raise ValueError("set_rule can only edit registered readouts")
        candidate.network[edit.node].set_logic(edit.rule)
        definitions[edit.node] = {"origin": "readout", "rule": edit.rule, "explanation": edit.explanation}
    result = {"session_id": session.session_id, "previous_revision": before,
            "model_revision": model_revision(candidate, definitions),
            "changed_nodes": inspect_nodes(candidate, names, definitions),
            "scientific_approval": "not_granted", "timing": "stochastic simulated nodes, not instantaneous observables"}
    session.sim = candidate
    session.readouts = definitions
    return result


PROVENANCE_PREFIX = "// MCP_READOUTS_V1 "


def print_bnd_with_readouts(session, out):
    """Keep definitions inside the hash-verified BND across portable handoffs."""
    if session.readouts:
        print(PROVENANCE_PREFIX + json.dumps({"definitions": session.readouts, "source_model_revision": model_revision(session.sim, session.readouts)}, sort_keys=True), file=out)
    session.sim.print_bnd(out=out)


def read_readouts(path, sim):
    from pathlib import Path
    records = {}
    for line in Path(path).read_text().splitlines():
        if line.startswith(PROVENANCE_PREFIX):
            if records:
                raise ValueError("Multiple readout provenance records")
            payload = json.loads(line[len(PROVENANCE_PREFIX):])
            records = payload.get("definitions", payload)
    if not isinstance(records, dict):
        raise ValueError("Invalid readout provenance")
    if not records:
        return {}
    core = set(sim.network) - set(records)
    for name, record in records.items():
        if name not in sim.network or record.get("origin") != "readout":
            raise ValueError("Readout provenance does not match the loaded model")
        if record.get("rule") != sim.network[name].logExp:
            raise ValueError("Readout rule differs from its provenance")
        validate_boolean_rule(record["rule"], core)
        if any(name in re.findall(NAME, str(sim.network[n])) for n in core):
            raise ValueError("Readout feedback into the core is prohibited")
    return records




class ReadoutEditResult(StructuredOutputModel):
    server: Literal["MaBoSS"] = "MaBoSS"
    session_id: str
    previous_revision: str
    model_revision: str
    changed_nodes: list[dict]
    scientific_approval: Literal["not_granted"]
    timing: str
