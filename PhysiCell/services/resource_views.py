"""Read-only formatting helpers for PhysiCell resources and configuration data."""

import math
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from mcp_biomodelling_servers.artifact_manager import (
    METADATA_FILENAME,
    list_artifacts,
)

from ..session_manager import SessionState


def mapping_from_method(
    owner: object,
    method_name: str,
) -> Mapping[str, Any]:
    """Return a backend mapping while retaining useful backend failures."""
    method = getattr(owner, method_name, None)
    if method is not None:
        value = method()
        if not isinstance(value, Mapping):
            raise TypeError(
                f"{type(owner).__name__}.{method_name}() did not return a mapping."
            )
        return value

    data = getattr(owner, "data", None)
    if isinstance(data, Mapping):
        return data
    return {}


def mapping_at(value: object, *keys: str) -> Mapping[str, Any]:
    """Read a nested mapping, returning an empty mapping when absent."""
    current = value
    for key in keys:
        if not isinstance(current, Mapping):
            return {}
        current = current.get(key)
    return current if isinstance(current, Mapping) else {}


def configuration_cell_type(
    config: object,
    cell_type: str,
) -> Mapping[str, Any]:
    """Return one configured cell type or raise an actionable error."""
    try:
        cell_types = config.cell_types.get_cell_types()
    except Exception as exc:
        raise RuntimeError(
            f"Could not inspect PhysiCell cell types: {exc}"
        ) from exc
    if not isinstance(cell_types, Mapping):
        raise TypeError(
            "PhysiCell cell_types.get_cell_types() did not return a mapping."
        )
    if cell_type not in cell_types:
        available = ", ".join(str(name) for name in cell_types) or "none"
        raise ValueError(
            f"Cell type {cell_type!r} is not configured. "
            f"Available cell types: {available}."
        )
    cell_data = cell_types[cell_type]
    if not isinstance(cell_data, Mapping):
        raise TypeError(
            f"PhysiCell cell type {cell_type!r} did not return a mapping."
        )
    return cell_data


def configuration_substrates(config: object) -> Mapping[str, Any]:
    """Return configured substrates or raise when the backend is malformed."""
    try:
        substrates = config.substrates.get_substrates()
    except Exception as exc:
        raise RuntimeError(
            f"Could not inspect PhysiCell substrates: {exc}"
        ) from exc
    if not isinstance(substrates, Mapping):
        raise TypeError(
            "PhysiCell substrates.get_substrates() did not return a mapping."
        )
    return substrates


def preserved_interaction_value(
    interaction: Mapping[str, Any],
    key: str,
    default: float,
) -> float:
    """Read one existing secretion value without accepting malformed data."""
    value = interaction.get(key, default)
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
    ):
        raise ValueError(
            f"Existing substrate interaction field {key!r} must be a finite "
            "number."
        )
    return float(value)


def display_value(value: object) -> str:
    """Format one compact, human-readable resource value."""
    if value is None or value == "":
        return "unavailable"
    if isinstance(value, bool):
        return "yes" if value else "no"
    if isinstance(value, float):
        return f"{value:g}" if math.isfinite(value) else str(value)
    return str(value)


def attribute_or_mapping_value(
    owner: object,
    mapping: Mapping[str, Any],
    key: str,
) -> Any:
    """Read a mapping value with an attribute fallback for test/backends."""
    if key in mapping:
        return mapping[key]
    return getattr(owner, key, None)


def display_extent(lower: object, upper: object) -> str:
    """Format an axis extent when both configured bounds are numeric."""
    try:
        extent = float(upper) - float(lower)
    except (TypeError, ValueError):
        return "unavailable"
    return display_value(extent)


def format_domain_resource(session: SessionState) -> str:
    """Render the spatial and temporal configuration."""
    config = session.config
    domain = config.domain
    options = config.options
    domain_info = mapping_from_method(domain, "get_info")
    options_info = mapping_from_method(options, "get_options")

    def domain_value(key: str) -> Any:
        return attribute_or_mapping_value(domain, domain_info, key)

    def option_value(key: str) -> Any:
        return attribute_or_mapping_value(options, options_info, key)

    units = display_value(option_value("space_units"))
    time_units = display_value(option_value("time_units"))
    mode = "2D" if bool(domain_value("use_2D")) else "3D"
    lines = [
        "# PhysiCell Domain",
        "",
        f"- Session: `{session.session_id}`",
        f"- Mode: {mode}",
        (
            "- Bounds: "
            f"x=[{display_value(domain_value('x_min'))}, "
            f"{display_value(domain_value('x_max'))}], "
            f"y=[{display_value(domain_value('y_min'))}, "
            f"{display_value(domain_value('y_max'))}], "
            f"z=[{display_value(domain_value('z_min'))}, "
            f"{display_value(domain_value('z_max'))}] {units}"
        ),
        (
            "- Extent: "
            f"x={display_extent(domain_value('x_min'), domain_value('x_max'))}, "
            f"y={display_extent(domain_value('y_min'), domain_value('y_max'))}, "
            f"z={display_extent(domain_value('z_min'), domain_value('z_max'))} "
            f"{units}"
        ),
        (
            "- Mesh spacing: "
            f"dx={display_value(domain_value('dx'))}, "
            f"dy={display_value(domain_value('dy'))}, "
            f"dz={display_value(domain_value('dz'))} {units}"
        ),
        (
            "- Maximum time: "
            f"{display_value(option_value('max_time'))} {time_units}"
        ),
        (
            "- Time steps: "
            f"diffusion={display_value(option_value('dt_diffusion'))}, "
            f"mechanics={display_value(option_value('dt_mechanics'))}, "
            f"phenotype={display_value(option_value('dt_phenotype'))} "
            f"{time_units}"
        ),
    ]
    return "\n".join(lines)


def format_substrates_resource(session: SessionState) -> str:
    """Render configured diffusible substrates."""
    substrates = mapping_from_method(
        session.config.substrates,
        "get_substrates",
    )
    lines = [
        "# PhysiCell Substrates",
        "",
        f"- Session: `{session.session_id}`",
        f"- Count: {len(substrates)}",
        "",
    ]
    if not substrates:
        lines.append("No substrates configured.")
        return "\n".join(lines)

    for name, raw_data in sorted(
        substrates.items(),
        key=lambda item: str(item[0]),
    ):
        data = raw_data if isinstance(raw_data, Mapping) else {}
        dirichlet = (
            f"enabled at {display_value(data.get('dirichlet_value'))}"
            if data.get("dirichlet_enabled")
            else "disabled"
        )
        lines.append(
            f"- **{name}** — "
            f"diffusion={display_value(data.get('diffusion_coefficient'))}; "
            f"decay={display_value(data.get('decay_rate'))}; "
            f"initial={display_value(data.get('initial_condition'))} "
            f"{display_value(data.get('units'))}; "
            f"Dirichlet={dirichlet}"
        )
    return "\n".join(lines)


def format_cell_types_resource(session: SessionState) -> str:
    """Render the principal phenotype values for each cell type."""
    cell_types = mapping_from_method(
        session.config.cell_types,
        "get_cell_types",
    )
    lines = [
        "# PhysiCell Cell Types",
        "",
        f"- Session: `{session.session_id}`",
        f"- Count: {len(cell_types)}",
        "",
    ]
    if not cell_types:
        lines.append("No cell types configured.")
        return "\n".join(lines)

    for name, raw_data in sorted(
        cell_types.items(),
        key=lambda item: str(item[0]),
    ):
        data = raw_data if isinstance(raw_data, Mapping) else {}
        phenotype = mapping_at(data, "phenotype")
        cycle = mapping_at(phenotype, "cycle")
        volume = mapping_at(phenotype, "volume")
        motility = mapping_at(phenotype, "motility")
        death = mapping_at(phenotype, "death")
        apoptosis = mapping_at(death, "apoptosis")
        necrosis = mapping_at(death, "necrosis")
        lines.append(
            f"- **{name}** — "
            f"cycle={display_value(cycle.get('model'))}; "
            "volume("
            f"total={display_value(volume.get('total'))}, "
            f"nuclear={display_value(volume.get('nuclear'))}); "
            "motility("
            f"speed={display_value(motility.get('speed'))}, "
            "persistence="
            f"{display_value(motility.get('persistence_time'))}); "
            "death("
            f"apoptosis={display_value(apoptosis.get('default_rate'))}, "
            f"necrosis={display_value(necrosis.get('default_rate'))}); "
            f"PhysiBoSS={display_value('intracellular' in phenotype)}"
        )
    return "\n".join(lines)


def format_cell_rules_resource(session: SessionState) -> str:
    """Render cell rules and external ruleset declarations."""
    rules_module = session.config.cell_rules
    rules = rules_module.get_rules()
    if not isinstance(rules, list):
        raise TypeError(
            f"{type(rules_module).__name__}.get_rules() did not return a list."
        )
    rulesets = mapping_from_method(rules_module, "get_rulesets")
    lines = [
        "# PhysiCell Cell Rules",
        "",
        f"- Session: `{session.session_id}`",
        f"- Rule count: {len(rules)}",
        f"- Ruleset count: {len(rulesets)}",
        "",
    ]
    if not rules:
        lines.append("No cell rules configured.")
    else:
        for raw_rule in rules:
            rule = raw_rule if isinstance(raw_rule, Mapping) else {}
            lines.append(
                "- "
                f"**{display_value(rule.get('cell_type'))}**: "
                f"{display_value(rule.get('signal'))} "
                f"{display_value(rule.get('direction'))} "
                f"{display_value(rule.get('behavior'))}; "
                "saturation="
                f"{display_value(rule.get('saturation_value'))}; "
                f"half-max={display_value(rule.get('half_max'))}; "
                f"Hill={display_value(rule.get('hill_power'))}; "
                "apply-to-dead="
                f"{display_value(bool(rule.get('apply_to_dead', False)))}"
            )

    if rulesets:
        lines.extend(["", "## Rulesets", ""])
        for name, raw_data in sorted(
            rulesets.items(),
            key=lambda item: str(item[0]),
        ):
            data = raw_data if isinstance(raw_data, Mapping) else {}
            folder = data.get("folder")
            filename = data.get("filename")
            path = (
                str(Path(str(folder)) / str(filename))
                if folder and filename
                else "unavailable"
            )
            lines.append(
                f"- **{name}** — enabled="
                f"{display_value(data.get('enabled'))}; file=`{path}`"
            )
    return "\n".join(lines)


def format_physiboss_resource(session: SessionState) -> str:
    """Render upstream MaBoSS context and configured intracellular models."""
    lines = [
        "# PhysiBoSS Integration",
        "",
        f"- Session: `{session.session_id}`",
    ]
    contexts = list(session.maboss_contexts.values())
    if contexts:
        lines.extend(
            [
                "",
                "## MaBoSS contexts",
                "",
                f"- Context count: {len(contexts)}",
            ]
        )
        for context in contexts:
            lines.extend(
                [
                    "",
                    (
                        f"### {display_value(context.target_cell_type)} — "
                        f"{display_value(context.model_name)}"
                    ),
                    "",
                    f"- Model: {display_value(context.model_name)}",
                    f"- BND file: `{context.bnd_file_path}`",
                    f"- CFG file: `{context.cfg_file_path}`",
                    f"- Available nodes: {len(context.available_nodes)}",
                    f"- Output nodes: {len(context.output_nodes)}",
                    (
                        "- Source manifest: "
                        f"`{context.source_manifest_path or 'not recorded'}`"
                    ),
                ]
            )

    intracellular_models: dict[str, Mapping[str, Any]] = {}
    try:
        cell_types = mapping_from_method(
            session.config.cell_types,
            "get_cell_types",
        )
    except Exception:  # noqa: BLE001
        cell_types = {}
    for name, raw_data in cell_types.items():
        data = raw_data if isinstance(raw_data, Mapping) else {}
        intracellular = mapping_at(data, "phenotype", "intracellular")
        if intracellular:
            intracellular_models[str(name)] = intracellular

    known_names = sorted(
        set(intracellular_models) | set(session.loaded_physiboss_models)
    )
    lines.extend(
        [
            "",
            "## Intracellular models",
            "",
            f"- Model count: {len(known_names)}",
            (
                "- Tracked operations: "
                f"settings={session.physiboss_settings_count}; "
                f"inputs={session.physiboss_input_links_count}; "
                f"outputs={session.physiboss_output_links_count}; "
                f"mutations={session.physiboss_mutations_count}"
            ),
        ]
    )
    if not known_names:
        lines.extend(["", "No PhysiBoSS integration configured."])
        return "\n".join(lines)

    for name in known_names:
        intracellular = intracellular_models.get(name, {})
        settings = mapping_at(intracellular, "settings")
        mapping = mapping_at(intracellular, "mapping")
        inputs = mapping.get("inputs")
        outputs = mapping.get("outputs")
        mutations = settings.get("mutations")
        initial_values = intracellular.get("initial_values")
        setting_values = "; ".join(
            f"{key}={display_value(value)}"
            for key, value in settings.items()
            if key != "mutations"
        )
        lines.append(
            f"- **{name}** — "
            f"type={display_value(intracellular.get('type'))}; "
            f"BND=`{display_value(intracellular.get('bnd_filename'))}`; "
            f"CFG=`{display_value(intracellular.get('cfg_filename'))}`; "
            f"settings=[{setting_values or 'none'}]; "
            f"initial-values={len(initial_values) if isinstance(initial_values, list) else 0}; "
            f"inputs={len(inputs) if isinstance(inputs, list) else 0}; "
            f"outputs={len(outputs) if isinstance(outputs, list) else 0}; "
            f"mutations={len(mutations) if isinstance(mutations, list) else 0}"
        )
    return "\n".join(lines)


def format_files_resource(session: SessionState, server_root: Path) -> str:
    """Render generated files without creating the artifact directory."""
    files = [
        path
        for path in list_artifacts(
            server_root,
            session_id=session.session_id,
        )
        if path.name != METADATA_FILENAME
    ]
    lines = [
        "# PhysiCell Artifact Files",
        "",
        f"- Session: `{session.session_id}`",
        f"- Count: {len(files)}",
        "",
    ]
    if not files:
        lines.append("No artifact files found for this session.")
    else:
        lines.extend(f"- `{path}`" for path in files)
    return "\n".join(lines)
