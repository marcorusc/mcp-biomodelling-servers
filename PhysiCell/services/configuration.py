"""Typed, explicit configuration patches and compact stored readback."""
import hashlib
import json
from copy import deepcopy
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from mcp_biomodelling_servers.structured_outputs import StructuredOutputModel


class CycleTransition(BaseModel):
    model_config = ConfigDict(extra="forbid")
    from_phase: int = Field(ge=0)
    to_phase: int = Field(ge=0)
    rate: float = Field(ge=0, allow_inf_nan=False)
    fixed_duration: bool | None = None


class UserParameter(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    name: str = Field(pattern=r"^[A-Za-z_][A-Za-z_0-9]*$", max_length=128)
    type: Literal["int", "double", "bool", "string"]
    value: int | float | bool | str
    units: str = Field(min_length=1, max_length=128)
    description: str | None = Field(default=None, max_length=2000)

    @model_validator(mode="after")
    def value_matches_type(self):
        import math
        valid = {"int": type(self.value) is int, "double": type(self.value) in (int, float),
                 "bool": type(self.value) is bool, "string": type(self.value) is str}
        if not valid[self.type] or (type(self.value) is float and not math.isfinite(self.value)):
            raise ValueError("value must match the declared type and be finite")
        return self


def revision(config):
    # Serialize a copy: the pinned library fills derived defaults during serialization.
    xml = deepcopy(config).generate_xml()
    rules = config.cell_rules.get_rules() if hasattr(config, "cell_rules") else []
    return hashlib.sha256(json.dumps({"xml": xml, "cell_rules": rules}, sort_keys=True).encode()).hexdigest()


def cell_values(config, cell_type, sections):
    cells = config.cell_types.get_cell_types()
    if cell_type not in cells:
        raise ValueError(f"Unknown cell type: {cell_type}")
    phenotype = cells[cell_type]["phenotype"]
    return {section: deepcopy(phenotype.get(section)) for section in sections}


def input_values(config, names):
    missing = [name for name in names if name not in config.user_parameters]
    if missing:
        raise ValueError("Unknown user parameters: " + ", ".join(missing))
    return {"parameters": {name: deepcopy(config.user_parameters[name]) for name in names},
            "initial_cells": deepcopy(config.initial_conditions.csv_config),
            "number_of_cells": deepcopy(config.user_parameters.get("number_of_cells")),
            "interpretation": "Stored settings; the target executable determines placement/count/seed semantics. CSV may be additive."}




class CellConfigurationResult(StructuredOutputModel):
    server: Literal["PhysiCell"] = "PhysiCell"
    session_id: str
    model_revision: str
    cell_type: str
    changed: bool | None = Field(default=None, exclude_if=lambda value: value is None)
    stored: dict


class SimulationInputsResult(StructuredOutputModel):
    server: Literal["PhysiCell"] = "PhysiCell"
    session_id: str
    previous_revision: str
    model_revision: str
    changed: bool
    parameters: dict
    initial_cells: dict
    number_of_cells: dict | None
    interpretation: str
    csv_artifact: dict | None


class ConfigurationBundleResult(StructuredOutputModel):
    server: Literal["PhysiCell"] = "PhysiCell"
    session_id: str
    model_revision: str
    bundle_id: str
    bundle: str
    manifest: str
    status: Literal["configuration_complete"]
    requirements: list[str]
