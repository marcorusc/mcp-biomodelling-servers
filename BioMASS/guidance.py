"""Workflow guidance published to every MCP client."""

BIOMASS_SERVER_INSTRUCTIONS = (
    "Use one session per ODE model. Import a NeKo-to-BioMASS handoff or standalone "
    "Text2Model text. Read literature with the calling agent's tools, record "
    "evidence, and distinguish supported reactions from explicit assumptions. "
    "Inspect unresolved edges before generation. Visualize generated revisions "
    "before interpreting simulations; graph edges do not distinguish inhibition "
    "or modifiers. Simulation requires an explicit scenario and time span; "
    "placeholder values require explicit opt-in. Read docs://biomass/agent_manual "
    "or use biomass_workflow_prompt."
    " Before authoring, read docs://biomass/reaction_syntax and "
    "docs://biomass/authoring_examples; for imported networks also read "
    "docs://biomass/network_to_reactions. A signed edge or PMID alone does not "
    "establish a mechanism or kinetic law. set_reactions replaces the whole list."
)

BIOMASS_AGENT_MANUAL = """# BioMASS Agent Operations Manual

## Authoring references
Read these MCP resources before writing reactions; no source-code search or
internet access is required to learn the supported grammar:
- docs://biomass/reaction_syntax: BioMASS 0.14 statements, kinetic laws, parameter
  names, directives, sharing, and the subset accepted by this server.
- docs://biomass/authoring_examples: tested complete tool arguments for evidence,
  reaction records, numerical configuration, and standalone line provenance.
- docs://biomass/network_to_reactions: how to justify mechanisms from literature,
  name molecular states, record assumptions, and preserve unresolved edges.

## Workflow
1. create_session; pass session_id explicitly when several sessions are active.
2. For NeKo, call export_biomass_handoff on NeKo, then import_neko_handoff here.
   For standalone work, use import_text with a complete Text2Model document.
3. Retrieve papers with the calling agent's literature tools. Preserve DOI/PMID
   identifiers, passage locations, context, limitations, and contradictory evidence.
   set_evidence upserts records; it does not retrieve or interpret papers.
4. For a network model, set_reactions replaces the ordered reaction list. Each
   supported reaction links supporting evidence; each assumed reaction needs a
   rationale. Keep stable reaction IDs. Several edges may support one reaction,
   and an edge may need several reactions. Missing mechanisms remain unresolved.
   A signed edge or a PMID alone does not establish a mechanism or kinetic law.
   Do not force one reaction per edge. set_reactions replaces the COMPLETE list;
   retain existing records when extending a model with another batch.
5. Use share_parameters_with to refer to an earlier reaction ID. Never use raw
   line-number sharing in reaction records. Standalone documents preserve their
   exact text, including line-number references; import_text replaces the whole
   document and its line provenance together.
6. configure_model replaces the configuration. Set observables and time_span for
   record mode. In document mode, put observables, conditions, and @sim tspan in
   the text; configuration supports numerical defaults and unit metadata.
7. inspect_model and validate_model report coverage and syntax independently.
   validate_model(check_generation=true) also compiles in a disposable worker.
   Neither establishes biological validity or numerical validity.
8. generate_model creates an immutable revision, with source text, configuration,
   evidence, assumptions, equations, parameter origins, and integrity hashes.
   Authoring changes clear current_revision while retaining previous revisions.
9. visualize_model supports PNG, SVG, and interactive HTML; export_model_graph
   writes DOT. These use BioMASS's species projection: reactants and modifiers,
   and positive and negative modulation, are not distinguished. Interactive
   graphs are not time animations. Graphs do not need calibrated parameters.
10. run_simulation takes a named scenario, requires an explicit time span, and
    rejects placeholders unless allow_placeholders=true. Initial zero defaults
    also count as placeholders. Exact numerical values used in each condition
    are saved; results are exploratory. Scenario overrides replace model defaults
    before Text2Model condition assignments are applied. Sharing constraints
    remain effective; override the source parameter of a shared parameter.
11. export_model_bundle includes the selected revision, its graphs and runs,
    provenance, coverage, and a script reproducing saved numerical scenarios.

## Supported text
Use BioMASS 0.14 reaction rules and @rxn arithmetic, @obs, @add species/param,
and @sim directives. Expressions support finite numeric constants, arithmetic,
and p[name], u[name], init[name] references where applicable. Python calls,
attributes, imports, and arbitrary executable packages are not accepted. Model
identifiers use letters, digits, and single underscores, starting with a letter.
Numeric reaction values are finite and nonnegative. Text input is limited to
1 MiB / 2000 lines, generated models to 500 species / 1000 reactions, time spans
to 10001 integer samples, and conditions to 20. Numerical configuration names
must exactly match generated symbols shown by inspect_model/generate_model.

## Sessions, artifacts, and failures
Workers run with a 60-second timeout (configurable from 1 to 300 seconds), in
isolated working directories. A failed/timed-out job keeps only diagnostics,
never a completed revision. Closing a session preserves disk artifacts;
restore_session reloads the durable authoring state. list_generated_files lists
nested artifacts. clean_generated_files deletes generated revisions and runs,
but preserves authoring state and source lineage. Session resources are read-only.

Install mcp-biomodelling-servers[biomass-graph] plus Graphviz for graph rendering.
PyGraphviz may require Graphviz headers and a compiler when no wheel is available.
Missing graph dependencies do not prevent construction or simulation.

Calibration, sensitivity analysis, literature retrieval, and a companion agent
skill are outside this server's first-release scope.
"""
