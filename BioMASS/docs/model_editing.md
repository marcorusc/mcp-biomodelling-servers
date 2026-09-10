# Conversational construction and model editing

Use `build_reactions` to translate an agent's explicit reaction choices into
Text2Model, starting from an empty session, an imported NeKo network, or an
existing Text2Model document. The server validates and applies edits; the calling
agent interprets the conversation and chooses mechanisms, kinetics, and any
scientific annotations. There is no automatic model-reduction operation.

## Inspect, preview, apply

1. `create_session`, or restore the session containing the model to edit.
2. Optionally use `import_neko_handoff`, `import_text`, or `import_text_file`.
3. `inspect_reactions` returns `document_version`, reaction IDs, text line numbers,
   generated parameter names, and species. It compiles in disposable workers by
   default; `check_generation=false` returns the textual inventory quickly.
4. Call `build_reactions` with that `expected_version` and explicit edits. The
   default `preview=true` validates the whole candidate without changing state.
5. Inspect `can_apply`, `issues`, `changes`, and added/removed symbols. To apply,
   repeat with `preview=false` and the same version. A stale version is rejected;
   inspect again before preparing a new edit.
6. `generate_model` publishes a revision for visualization, simulation, and export.

Each edit has `action` (`add`, `update`, or `remove`) and a stable `reaction_id`.
Add/update takes exactly one of `template` or `statement`. Update replaces the
reaction statement; it preserves existing metadata when `metadata` is omitted.
Removal takes only the action and ID. A batch can edit each ID once.

Preview compilation proves executable construction, not biological correctness or
numerical behavior. Failures leave authoring state and completed revisions intact.
Applied edits invalidate the current revision while retaining previous revisions.
Inspection and each edit batch have a total worker budget of 60 seconds by default,
configurable through `timeout_seconds` up to 300 seconds. Large models may need a
larger budget: preview compiles both structures and then the full candidate.

## Construction templates

The table below is checked against the implementation in tests. The tool schema
lists template names; no separate template-listing tool is needed.
See `docs://biomass/reaction_syntax` for the corresponding equations and kinetics.

| Template | Required participants | Optional participants | Local parameters |
|---|---|---|---|
| `binding` | left, right, complex | none | kf; kr when reversible |
| `dissociation` | complex, left, right | none | kf; kr when reversible |
| `dimerization` | monomer, dimer | none | kf; kr when reversible |
| `conversion` | substrate, product | none | kf; kr when reversible |
| `phosphorylation` | enzyme, substrate, product | none | V, K |
| `dephosphorylation` | enzyme, substrate, product | none | V, K |
| `transcription` | regulator, product | none | V, K, n |
| `synthesis` | product | regulator | kf |
| `degradation` | substrate | regulator | kf |
| `transport` | source, destination | none | kf; kr when reversible |

Pass required participant roles as a dictionary. `reversible=true` adds the
reverse reaction for binding, dissociation, dimerization, conversion, and transport.
Transport templates use Text2Model's equal-volume default; use a raw statement for
explicit compartment volumes. Synthesis/degradation with a `regulator` use the
regulated forms; without it they use basal forms. More complex regulation and
custom reaction shapes belong in `statement`.

Example `build_reactions` arguments for an empty session (replace the session ID
and use its actual version):

```json
{
  "session_id": "SESSION_ID",
  "expected_version": 0,
  "preview": true,
  "edits": [
    {
      "action": "add",
      "reaction_id": "binding",
      "template": "binding",
      "participants": {"left": "E", "right": "S", "complex": "ES"},
      "reversible": true,
      "parameters": {"kf": {"value": 0.003}, "kr": {"value": 0.001}},
      "initials": {"E": {"value": 100}, "S": {"value": 50}, "ES": {"value": 0}}
    }
  ],
  "configuration": {"observables": {"TotalEnzyme": "u[E] + u[ES]"}, "time_span": [0, 5]}
}
```

Template parameters use local names such as `kf`, `V`, or `K`; the server maps
them to generated names such as `kf1`. Quantities accept units and provenance as
described in `docs://biomass/authoring_examples`. Unspecified units remain null.
Values in this example are illustrative; the server does not infer values.

For raw custom kinetics, use `statement` and exact parameter names:

```json
{
  "action": "add",
  "reaction_id": "custom_conversion",
  "statement": "@rxn A --> B: p[rate] * u[A] / (p[half] + u[A])",
  "parameters": {"rate": {"value": 0.2}, "half": {"value": 1}},
  "initials": {"A": {"value": 10}, "B": {"value": 0}}
}
```

This is one item for `edits`, not a whole tool call. Expressions are validated
before conversion. Python imports, function calls, attributes, and arbitrary
Python model packages are not accepted. For raw built-in statements, numeric
overrides still use local parameter names; `@rxn` overrides use exact names.

## Editing existing files

`import_text_file(path="/absolute/path/model.txt", session_id=...)` reads at most
1 MiB of UTF-8 text, preserving comments, whitespace, directives, and CRLF/LF
line endings. It copies the contents into session state, records the source path
and SHA-256, and never writes back to the original file. Like `import_text`, this
replaces a standalone document and resets its configuration and line provenance;
provide `line_evidence` with the import if needed. Use a fresh session when
starting from a file while another model is open.

Imported reactions initially have IDs such as `line_2`. `inspect_reactions`
discovers these, including custom `@rxn` lines. Added reactions receive the IDs
you supply. Replacements retain their line positions; additions append. Removed
reactions become comment lines, preserving later generated parameter names and
numeric sharing references. Deleting a sharing source requires repairing or
removing the dependent reaction in the same batch.

Use `line_edits=[{"line_number": 4, "text": "@obs Total: u[A] + u[B]"}]` to
replace an existing directive/comment by its original line number. `text=null`
leaves a removal comment. Use `append_lines` for new directives/comments. These
interfaces cannot insert reactions; use `edits` for reactions. Changed directive
lines lose their old line-evidence links. Reaction metadata can be explicitly
replaced through the edit's `metadata` field.

If removing a reaction also removes species C, an observable containing `u[C]`
must be repaired in the same batch. The result reports dangling expressions,
initial values, parameters, and conditions. `configuration`, when supplied,
replaces the complete configuration atomically with the edits. In document mode,
observables, conditions, and time span belong in directives; numerical defaults
and units can be configured separately.

Compatible numerical settings are retained. `prune_unused_values=true` explicitly
removes parameter and initial-value defaults for symbols absent after the edit.
It does not rewrite observables or simulation conditions. For parameter sharing,
use `share_parameters_with` to name an earlier reaction ID. Assign numerical
values to the source reaction; sharing and local parameter values are exclusive.

`set_reactions` and `import_text` remain complete replacement interfaces.
`set_reactions` can renumber generated parameters and clears numerical overrides
when the reaction list changes. Prefer `build_reactions` for incremental edits.

## Network names and optional annotations

For NeKo models, `species_mapping` maps node IDs to one or more species names.
Default names are sanitized gene symbols, with collisions rejected. These are
identifier aliases, not inferred molecular states. Supply explicit mappings when
needed, for example `{"node_id": ["Kinase", "Kinase_active"]}`. Existing mappings
can be extended with additional states; renaming/removing existing mapped names
requires a new session. A template participant may use a node ID only when its
mapping has exactly one species; otherwise choose the species name explicitly.
Raw statements and initial-value keys use species names, without alias expansion.

Optional `metadata` contains `status`, `edge_ids`, `evidence_ids`, and `assumption`.
New reactions default to `unreviewed`; their linked edges remain unresolved in
evidence coverage. Explicit `supported` status requires supporting evidence links.
`assumed` status does not require assumption text. The agent decides what context
to collect and supply; the server preserves supplied metadata and validates links.
It never invents scientific assumptions or checks whether they came from a user.
Many-to-many edge/reaction links are supported and exported with the snapshot.
