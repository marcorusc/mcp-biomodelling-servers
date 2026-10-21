"""Typed user parameters and preserved, researcher-supplied cell positions."""
import csv
import hashlib
import math
import uuid
from copy import deepcopy
from pathlib import Path
from typing import Annotated

from pydantic import Field

from mcp_biomodelling_servers.artifact_manager import (
    get_artifact_dir,
    safe_artifact_path,
)

from ..app import tool
from ..contracts import NON_IDEMPOTENT, NonEmptyString
from ..locking import session_locked
from ..services.configuration import (
    SimulationInputsResult,
    UserParameter,
    input_values,
    revision,
)
from ..session_manager import WorkflowStep, get_current_session

ROOT = Path(__file__).resolve().parents[1]


def validate_cells_csv(path):
    if not path.is_absolute() or path.is_symlink() or not path.is_file() or path.suffix.lower() != ".csv":
        raise ValueError("Supply an existing absolute regular CSV path")
    if path.stat().st_size > 16 * 1024 * 1024:
        raise ValueError("Initial cells CSV exceeds the 16 MiB import bound")
    with path.open("rb") as raw:
        if raw.read(3) == b"\xef\xbb\xbf":
            raise ValueError("UTF-8 BOM is unsupported by the target engine CSV version detector; supply a BOM-free CSV")
    count = 0
    header = False
    with path.open(newline="", encoding="utf-8-sig") as stream:
        for index, line in enumerate(stream):
            if '"' in line:
                raise ValueError("Quoted CSV fields are unsupported by the target engine reader")
            row = next(csv.reader([line]))
            if index == 0 and [v.lower() for v in row] == ["x", "y", "z", "type"]:
                header = True  # PhysiCell v2 interprets column four as a cell-type name.
                continue
            if len(row) != 4:
                raise ValueError("Supported CSV format is x,y,z,type (optional matching header), four columns")
            try:
                if not all(math.isfinite(float(value)) for value in row[:3]) or not row[3].strip():
                    raise ValueError()
            except ValueError as exc:
                raise ValueError(f"Invalid cell coordinates/type at CSV row {index + 1}") from exc
            if any('"' in value or "," in value for value in row):
                raise ValueError("Quoted/embedded-comma fields are unsupported by the target engine CSV reader")
            if not header:
                try:
                    identifier = float(row[3])
                    if not math.isfinite(identifier) or identifier < 0 or not identifier.is_integer():
                        raise ValueError()
                except ValueError as exc:
                    raise ValueError("Headerless PhysiCell CSV requires a nonnegative integer cell type ID in column four") from exc
            count += 1
    if not count:
        raise ValueError("CSV contains no cell records")
    return {"cell_records": count, "format": "PhysiCell CSV v2" if header else "PhysiCell CSV v1",
            "type_semantics": "cell-type names" if header else "numeric cell-type IDs"}


@tool(title="Configure PhysiCell simulation inputs", annotations=NON_IDEMPOTENT, structured_output=True)
@session_locked
def configure_simulation_inputs(
    session_id: Annotated[NonEmptyString | None, Field(description="Session to inspect or update.")] = None,
    parameters: Annotated[list[UserParameter] | None, Field(min_length=1, max_length=25, description="Explicit typed user parameter updates.")] = None,
    inspect_parameters: Annotated[list[NonEmptyString] | None, Field(max_length=25, description="Named parameters to inspect; updates are returned when omitted.")] = None,
    cells_csv: Annotated[NonEmptyString | None, Field(description="Absolute path to an existing x,y,z,type CSV to preserve.")] = None,
    csv_enabled: Annotated[bool | None, Field(description="Explicitly enable or disable CSV loading; required when registering a file.")] = None,
) -> SimulationInputsResult:
    """Store explicitly typed parameters or register an existing x,y,z,type CSV.

    No edits means inspection. CSV registration requires explicit csv_enabled.
    Counts and random_seed are stored, not proof the target executable uses them.
    """
    session = get_current_session(session_id)
    if session is None or session.config is None:
        raise ValueError("Create or load a configuration first")
    candidate = deepcopy(session.config)
    names = inspect_parameters or [item.name for item in parameters or []]
    if len({item.name for item in parameters or []}) != len(parameters or []):
        raise ValueError("Parameter names must be unique")
    for parameter in parameters or []:
        old = candidate.user_parameters.get(parameter.name, {})
        candidate.add_user_parameter(parameter.name, parameter.value, units=parameter.units,
                                     parameter_type=parameter.type,
                                     description=parameter.description if parameter.description is not None else old.get("description", ""))
    artifact = None
    try:
        if cells_csv is not None:
            if csv_enabled is None:
                raise ValueError("Registering a CSV requires explicit csv_enabled")
            source = Path(cells_csv)
            count = validate_cells_csv(source)
            directory = get_artifact_dir(ROOT, session.session_id)
            artifact = safe_artifact_path(directory, "initial_cells_" + uuid.uuid4().hex + ".csv")
            data = source.read_bytes()
            with artifact.open("xb") as out:
                out.write(data)
            count = validate_cells_csv(artifact)  # Validate the captured bytes, not a possibly changed source.
            candidate.initial_conditions.add_csv_file(artifact.name, folder=str(directory), enabled=csv_enabled)
            record = {"path": str(artifact), "sha256": hashlib.sha256(data).hexdigest(),
                      "size_bytes": len(data), **count}
        elif csv_enabled is not None:
            current = candidate.initial_conditions.csv_config
            if csv_enabled:
                validate_cells_csv(Path(current["folder"]) / current["filename"])
            candidate.initial_conditions.add_csv_file(current["filename"], folder=current["folder"], enabled=csv_enabled)
        values = input_values(candidate, names)
        before = revision(session.config)
        after = revision(candidate)
        if before != after:
            session.publish_config_update(config=candidate, completed_step=WorkflowStep.DOMAIN_SETUP,
                                          configuration_changed=True)
        if artifact is not None:
            session.initial_cells_artifact = record
        return SimulationInputsResult(**{"session_id": session.session_id, "previous_revision": before, "model_revision": after,
                "changed": before != after, **values, "csv_artifact": session.initial_cells_artifact})
    except BaseException:
        if artifact is not None:
            artifact.unlink(missing_ok=True)
        raise
