"""One bounded BioMASS operation in a fresh interpreter and working directory.

Only the server's validated text and JSON inputs cross this boundary. Generated
Python is always rebuilt here; caller-provided Python packages are never loaded.
"""

from __future__ import annotations

import ast
import importlib
import json
import os
import re
import sys
import traceback
from importlib.metadata import version
from pathlib import Path

if __package__ in (None, ""):
    server_directory = Path(__file__).resolve().parent
    if server_directory.parent.name == "mcp_biomodelling_servers":
        # The wheel remaps this server into the package namespace. Adding that
        # namespace directory to sys.path would shadow upstream biomass with
        # our console entrypoint biomass.py.
        sys.path.insert(0, str(server_directory.parent.parent))
        __package__ = "mcp_biomodelling_servers.BioMASS"
    else:
        sys.path.insert(0, str(server_directory.parent))
        __package__ = "BioMASS"

from .contracts import (
    GRAPH_LIMITATIONS,
    GraphOptions,
    ModelConfiguration,
    SimulationScenario,
)
from .services.authoring import identifier, validate_text


def write_json(name: str, value: object) -> None:
    Path(name).write_text(
        json.dumps(value, indent=2, allow_nan=False), encoding="utf-8"
    )


def apply_values(
    path: Path, parameters: dict, initials: dict, known_parameters: list, species: list
) -> None:
    """Change only generated numeric defaults, preserving upstream constraints."""
    for mapping, known in ((parameters, known_parameters), (initials, species)):
        unknown = set(mapping) - set(known)
        if unknown:
            raise ValueError(f"Unknown numerical symbols: {sorted(unknown)}")
    tree = ast.parse(path.read_text())
    for node in tree.body:
        if not isinstance(node, ast.FunctionDef):
            continue
        if node.name == "param_values":
            mapping, target, names = parameters, "x", "C"
        elif node.name == "initial_values":
            mapping, target, names = initials, "y0", "V"
        else:
            continue
        # BioMASS's initial-value function returns y0 (verified in the generated template).
        return_node = next(n for n in node.body if isinstance(n, ast.Return))
        if isinstance(return_node.value, ast.Name):
            target = return_node.value.id
        for name, quantity in mapping.items():
            identifier(name)
            assignment = ast.parse(
                f"{target}[{names}.{name}] = {quantity['value']!r}"
            ).body[0]
            node.body.insert(node.body.index(return_node), assignment)
    path.write_text(
        ast.unparse(ast.fix_missing_locations(tree)) + "\n", encoding="utf-8"
    )


def supplied_names(text: str) -> tuple[set[str], set[str], dict[str, str]]:
    params: set[str] = set()
    initials: set[str] = set()
    sharing: dict[str, str] = {}
    for n, line in enumerate(text.splitlines(), 1):
        line = line.split("#", 1)[0].strip()
        sections = line.split("|")
        if len(sections) > 1 and not sections[1].strip().isdecimal():
            for match in re.finditer(
                r"(?:const\s+)?([A-Za-z][A-Za-z0-9_]*)\s*=", sections[1]
            ):
                params.add(match[1] if line.startswith("@rxn ") else match[1] + str(n))
        if len(sections) > 2:
            initials.update(
                re.findall(r"(?:fixed\s+)?([A-Za-z][A-Za-z0-9_]*)\s*=", sections[2])
            )
    return params, initials, sharing


def quantity_details(
    names: list[str], values: list[float], supplied: set[str], overrides: dict
) -> dict:
    return {
        name: {
            "value": float(value),
            "units": overrides.get(name, {}).get("units"),
            "origin": overrides.get(name, {}).get(
                "origin", "supplied" if name in supplied else "placeholder"
            ),
            "evidence_ids": overrides.get(name, {}).get("evidence_ids", []),
        }
        for name, value in zip(names, values, strict=True)
    }


def generate(request: dict):
    from biomass import Text2Model, create_model

    text = request["text"]
    validate_text(text)
    config = ModelConfiguration.model_validate(request["configuration"])
    Path("model.txt").write_text(text, encoding="utf-8")
    # Fixed, server-owned package basename; independent of user labels or paths.
    Path("generated_model.txt").write_text(text, encoding="utf-8")
    description = Text2Model("generated_model.txt")
    description.convert()
    if len(description.species) > 500 or len(description.reactions) > 1000:
        raise ValueError(
            "v1 supports at most 500 species and 1000 reactions per model."
        )
    for name in [*description.species, *description.parameters]:
        identifier(name)
    # Reject dangling symbols in custom rates before importing generated code.
    for root, name in re.findall(r"\b(p|u)\[([A-Za-z][A-Za-z0-9_]*)\]", text):
        if name not in (description.parameters if root == "p" else description.species):
            raise ValueError(f"Unknown {root} symbol: {name}")
    params = {k: v.model_dump() for k, v in config.parameters.items()}
    initials = {k: v.model_dump() for k, v in config.initials.items()}
    apply_values(
        Path("generated_model/ode.py"),
        params,
        initials,
        description.parameters,
        description.species,
    )
    # Keep generated parameter-sharing equalities effective after default overrides.
    tree = ast.parse(Path("generated_model/ode.py").read_text())
    constraints = [ast.parse(s).body[0] for s in description.param_constraints]
    targets = {re.search(r"x\[C\.(\w+)\]", s)[1] for s in description.param_constraints}
    if targets & params.keys():
        raise ValueError(
            "Override the source parameter of a sharing constraint, not its dependent parameter."
        )
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name == "param_values":
            pos = next(
                i for i, child in enumerate(node.body) if isinstance(child, ast.Return)
            )
            node.body[pos:pos] = constraints
    Path("generated_model/ode.py").write_text(
        ast.unparse(ast.fix_missing_locations(tree)) + "\n"
    )
    # A model without observables is valid structurally; remove upstream's empty-name placeholder.
    if not description.obs_desc:
        observable = Path("generated_model/observable.py")
        source = observable.read_text().replace("            '',\n", "")
        observable.write_text(source)
    # Reapply parameter sharing after condition/steady-state assignments as well.
    # Otherwise an overridden source rate would leave its dependent rates stale.
    if targets:
        assigned = set(re.findall(r"p\[([A-Za-z][A-Za-z0-9_]*)\]\s*=", text))
        if targets & assigned:
            raise ValueError(
                "Conditions must override the source of shared parameters, not dependent parameters."
            )

        class SharedParameters(ast.NodeTransformer):
            def visit_Assign(self, node):
                if (
                    isinstance(node.value, ast.Call)
                    and isinstance(node.value.func, ast.Name)
                    and node.value.func.id in ("solve_ode", "get_steady_state")
                ):
                    return [*constraints, node]
                return node

        observable = Path("generated_model/observable.py")
        transformed = SharedParameters().visit(ast.parse(observable.read_text()))
        observable.write_text(
            ast.unparse(ast.fix_missing_locations(transformed)) + "\n"
        )
    sys.path.insert(0, str(Path.cwd()))
    model = create_model("generated_model")
    pnames, inames, _ = supplied_names(text)
    pdetails = quantity_details(model.parameters, model.pval(), pnames, params)
    idetails = quantity_details(model.species, model.ival(), inames, initials)
    for constraint in description.param_constraints:
        destination, source = re.findall(r"x\[C\.(\w+)\]", constraint)
        pdetails[destination] = {**pdetails[source], "shared_with": source}
    summary = {
        "biomass_version": version("biomass"),
        "species": model.species,
        "parameters": pdetails,
        "initials": idetails,
        "reactions": description.reactions,
        "equations": description.differential_equations,
        "observables": model.observables,
        "conditions": model.problem.conditions,
        "time_span": [model.problem.t[0], model.problem.t[-1]],
        "time_units": config.time_units,
        "explicit_time_span": bool(description.sim_tspan),
        "parameter_constraints": description.param_constraints,
        "line_mapping": request.get("line_mapping", {}),
    }
    write_json("model_summary.json", summary)
    return description, model, summary


def simulate(request: dict, model, summary: dict) -> dict:
    import numpy as np
    import pandas as pd
    from biomass.dynamics.solver import solve_ode

    scenario = SimulationScenario.model_validate(request["scenario"])
    if not summary["explicit_time_span"]:
        raise ValueError("Set an explicit @sim tspan before simulation.")
    if not scenario.allow_placeholders and any(
        q["origin"] == "placeholder"
        for group in (summary["parameters"], summary["initials"])
        for q in group.values()
    ):
        raise ValueError(
            "Scenario contains placeholder values. Supply all missing values or explicitly set allow_placeholders=true."
        )
    observable_module = importlib.import_module("generated_model.observable")
    condition_records = []
    species_frames = []
    observable_frames = []
    condition_names = list(model.problem.conditions)
    for name in condition_names:
        # Fresh defaults for every condition: upstream reuses mutable arrays between conditions.
        model.problem.conditions = [name]
        model.problem.simulations = np.full(
            (len(model.observables), 1, len(model.problem.t)), np.nan
        )
        captures = []

        def capture(diffeq, y0, times, parameters, _captures=captures, **kwargs):
            sol = solve_ode(diffeq, y0, times, parameters, **kwargs)
            _captures.append((sol, list(y0), list(parameters)))
            return sol

        observable_module.solve_ode = capture
        success = model.problem.simulate(model.pval(), model.ival())
        if success is False or len(captures) != 1 or captures[0][0] is None:
            raise RuntimeError(f"BioMASS solver failed for condition {name}.")
        sol, actual_initials, actual_parameters = captures[0]
        if (
            not np.isfinite(sol.y).all()
            or not np.isfinite(model.problem.simulations).all()
        ):
            raise RuntimeError(f"Nonfinite numerical output in condition {name}.")
        frame = pd.DataFrame(sol.y.T, columns=model.species)
        frame.insert(0, "time", sol.t)
        frame.insert(0, "condition", name)
        species_frames.append(frame)
        obs = pd.DataFrame(
            model.problem.simulations[:, 0, :].T, columns=model.observables
        )
        obs.insert(0, "time", sol.t)
        obs.insert(0, "condition", name)
        observable_frames.append(obs)
        condition_records.append(
            {
                "name": name,
                "parameters": dict(
                    zip(model.parameters, map(float, actual_parameters), strict=True)
                ),
                "initials": dict(
                    zip(model.species, map(float, actual_initials), strict=True)
                ),
            }
        )
    pd.concat(species_frames).to_csv("species.csv", index=False)
    pd.concat(observable_frames).to_csv("observables.csv", index=False)
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots()
    for frame in species_frames:
        for name in model.species[:20]:
            ax.plot(
                frame["time"],
                frame[name],
                label=f"{frame['condition'].iloc[0]}: {name}",
            )
    ax.set_xlabel(f"Time ({summary['time_units'] or 'unspecified units'})")
    ax.set_ylabel("Species amount (see numerical configuration for units)")
    if len(model.species) * len(species_frames) <= 20:
        ax.legend(fontsize="small")
    fig.savefig("trajectories.png", dpi=120, bbox_inches="tight")
    plt.close(fig)
    result = {
        "scenario": scenario.model_dump(),
        "numerical_valid": True,
        "exploratory": True,
        "conditions": condition_records,
        "parameters": summary["parameters"],
        "initials": summary["initials"],
        "time_span": summary["time_span"],
        "time_units": summary["time_units"],
        "plotted_species": model.species[:20],
    }
    write_json("simulation.json", result)
    return result


def graph(request: dict, description) -> dict:
    import importlib.util
    import shutil

    if importlib.util.find_spec("pygraphviz") is None or shutil.which("dot") is None:
        raise RuntimeError(
            "Graph rendering requires mcp-biomodelling-servers[biomass-graph] and the Graphviz system runtime (dot). Install Graphviz headers/compiler if building PyGraphviz from source."
        )
    options = GraphOptions.model_validate(request["graph_options"])
    fmt = request["format"]
    if fmt not in ("png", "svg", "html", "dot"):
        raise ValueError("Unsupported graph format.")
    filename = f"graph.{fmt}"
    if fmt == "dot":
        description.graph.write(filename)
    elif fmt == "html":
        if importlib.util.find_spec("pyvis") is None:
            raise RuntimeError(
                "Interactive HTML requires mcp-biomodelling-servers[biomass-graph] (pyvis)."
            )
        description.dynamic_plot(
            save_dir=".",
            file_name=filename,
            show=False,
            show_controls=options.show_controls,
            which_controls=["physics", "layout"],
        )

    else:
        description.static_plot(
            save_dir=".", file_name=filename, gviz_prog=options.layout
        )
    return {
        "format": fmt,
        "species": description.species,
        "edges": [[str(a), str(b)] for a, b in description.graph.edges()],
        "layout": options.layout if fmt in ("png", "svg") else None,
        "interpretation_limits": GRAPH_LIMITATIONS,
    }


def main() -> None:
    os.environ["MPLBACKEND"] = "Agg"
    request = json.loads(Path("request.json").read_text())
    try:
        description, model, summary = generate(request)
        if request["operation"] == "simulate":
            result = simulate(request, model, summary)
        elif request["operation"] == "graph":
            result = graph(request, description)
        elif request["operation"] == "generate":
            result = summary
        else:
            raise ValueError("Unknown worker operation.")
        write_json("result.json", result)
    except Exception as exc:  # noqa: BLE001 -- worker boundary returns typed diagnostics
        write_json("error.json", {"type": type(exc).__name__, "message": str(exc)})
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
