# From a referenced network to biochemical reactions

Read docs://biomass/reaction_syntax for accepted statements and generated
kinetics, and docs://biomass/authoring_examples for complete tool calls.

## Decide what the network supports

A signed edge describes an observed or curated relationship. It does not by
itself specify a molecular mechanism, kinetic law, or reaction stoichiometry.
A network with 92 edges need not become 92 reactions. An edge may need several
reactions, several edges may support one reaction, and some edges should remain
unresolved. Do not optimize authoring for a zero-unresolved coverage report.

Start with a small coherent subsystem. Inspect each edge's original effect,
mechanism metadata, reference identifiers, and biological context. Keep the
handoff's stable IDs in `edge_ids`; use separate readable names for model species.
Genes, proteins, complexes, modified forms, and compartment-specific pools may
need distinct species. One imported node can correspond to several ODE species.
State naming is an explicit modelling decision, not an automatic network mapping.

## Review evidence before choosing kinetics

Use the calling agent's literature tools to retrieve the edge's sources. Keep
DOI, PMID, database IDs, and unresolved identifiers. A PMID is a citation locator,
not proof of a proposed mechanism. Record what was actually accessible: full
text, abstract, database annotation, or unavailable source. Never represent a
database summary as a passage you read in the paper.

Store supporting passages or summaries, figure/page/section locations, organism,
cell type, stimulus, compartment, and relevant access limitations. Use `stance`
to distinguish `supports`, `contradicts`, and `context`. A context-only or
contradictory source cannot by itself justify a supported reaction. Keep conflicting
findings visible rather than discarding them to make the mechanism appear settled.

Assess separate questions: does the evidence support a direct interaction, the
specific molecular transformation, and the chosen kinetic approximation? Evidence
for phosphorylation may not justify Michaelis-Menten kinetics or a numerical V/K.
Do not silently promote mechanistic evidence into kinetic or parameter evidence.

## Choose a representation

| Available evidence | Modelling decision |
|---|---|
| Physical binding with a defined complex | Consider explicit association/dissociation; justify reversibility and complex composition. |
| Kinase/substrate transformation | Represent modified and unmodified substrate pools; choose and justify a saturating rule or explicit enzyme reactions. |
| Transcriptional regulation | Distinguish transcript from protein and justify any Hill approximation, synthesis, and degradation terms. |
| Inhibitory signed edge only | Investigate repression, sequestration, dephosphorylation, degradation, or another mechanism; keep unresolved if none is supported or explicitly assumed. |
| Indirect pathway effect | Avoid inventing direct molecular contact; introduce justified intermediate steps or record a coarse-grained assumption. |

`status="supported"` requires supporting evidence links. If the representation
requires an unsupported mechanistic or kinetic choice, use `status="assumed"`
and explain that choice in `assumption`, while retaining relevant evidence links.
The status is a coarse label for the whole reaction record; qualify which parts
are supported and which are assumed. Default numerical values are not evidence.
Quantity provenance and units are recorded separately in configuration/scenarios.

When no reaction can be justified, leave the edge without a reaction link. It
remains in `coverage.unresolved_edges`. Evidence records can explain the gap,
including inaccessible papers, ambiguous direction, and missing state information.
Do not submit a dummy reaction or comment to claim coverage.

## Author and revise without losing work

1. Store evidence with `set_evidence`; this upserts by evidence ID.
2. Read the current model with `inspect_model`. Assemble the complete ordered
   reaction list with stable IDs, edge links, evidence links, and assumptions.
3. Call `set_reactions` with that entire list. It REPLACES the list; it does not
   append a batch. Preserve earlier records when adding another subsystem.
4. Use `share_parameters_with` only for justified sharing with an earlier record.
   Reaction edits clear numerical overrides and conditions because generated
   parameter names depend on line numbers. Reapply configuration after inspection.
5. `configure_model` also replaces its complete configuration. Document units,
   provenance, observables, conditions, and explicit time bounds.
6. Use `validate_model(check_generation=true)`, generate, inspect equations and
   species, then visualize. Compare the generated transformations with the intended
   mechanisms; a valid syntax or successful conversion does not establish biology.
7. Simulate only an explicit numerical scenario; opt into placeholders only when
   deliberately exploring hypothetical values. Check relevant conservation laws.

Coverage tracks authored links, not the fraction of biological truth established.
Graphs project species connections and do not distinguish activating/inhibiting
modifiers. Review evidence and unresolved edges alongside a selected revision's
graph. Export the revision, its assumptions, evidence, and numerical provenance
when sharing results. Calibration and sensitivity analysis remain outside scope.
