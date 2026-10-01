# Tested authoring examples

These tool calls use synthetic enzyme chemistry to demonstrate the interfaces;
they make no claim about a real SIGNOR or OmniPath interaction. Read
docs://biomass/reaction_syntax and docs://biomass/network_to_reactions first.

## NeKo-derived reaction records

First `create_session`, then `import_neko_handoff` with a manifest returned by
NeKo's `export_biomass_handoff`. Inspect the imported network. In the JSON below,
replace `SESSION_ID` with the returned BioMASS session UUID and `EDGE_ID` with
an imported edge ID. These calls are appropriate for an illustrative E/S edge;
do not apply the chemistry to unrelated network edges.

The example source is deliberately labelled synthetic. For real work, replace
it with the actual source identifiers, reviewed content, and access limitations.
Support for chemistry does not establish the kinetic approximation; both
reaction records below therefore retain explicit assumptions.

```json
{
  "tool": "set_evidence",
  "arguments": {
    "session_id": "SESSION_ID",
    "records": [{
      "evidence_id": "enzyme_example",
      "source_identifiers": ["example:enzyme-chemistry"],
      "summary": "Synthetic example: E binds S to form ES, which releases E and P.",
      "source_location": "This resource, NeKo-derived reaction records example",
      "biological_context": "Hypothetical enzyme system; no organism or cell type specified.",
      "access_limitations": "No literature was retrieved for this example.",
      "edge_ids": ["EDGE_ID"],
      "stance": "supports"
    }]
  }
}
```

This is the complete ordered replacement list. Two reactions link to one edge;
other imported edges remain unresolved. When extending an existing model, include
all existing records as well. Numbers here are illustrative supplied values.

```json
{
  "tool": "set_reactions",
  "arguments": {
    "session_id": "SESSION_ID",
    "reactions": [
      {
        "reaction_id": "binding",
        "statement": "E + S <--> ES | kf=0.003, kr=0.001 | E=100, S=50, ES=0",
        "evidence_ids": ["enzyme_example"],
        "edge_ids": ["EDGE_ID"],
        "status": "assumed",
        "assumption": "Assume reversible mass-action binding with a single 1:1 complex; numerical values are illustrative."
      },
      {
        "reaction_id": "catalysis",
        "statement": "ES --> E + P | kf=0.002 | P=0",
        "evidence_ids": ["enzyme_example"],
        "edge_ids": ["EDGE_ID"],
        "status": "assumed",
        "assumption": "Assume irreversible first-order product release that regenerates E; numerical values are illustrative."
      }
    ]
  }
}
```

`configure_model` replaces the complete configuration. This example records
assumed units and numerical provenance explicitly. The forward bimolecular rate
constant has different units from the two first-order constants. These assumed
units are labels preserved by the server, not a dimensional-consistency proof.

```json
{
  "tool": "configure_model",
  "arguments": {
    "session_id": "SESSION_ID",
    "configuration": {
      "observables": {"Total_enzyme": "u[E] + u[ES]", "Product": "u[P]"},
      "time_span": [0, 100],
      "time_units": "s",
      "parameters": {
        "kf1": {"value": 0.003, "units": "1/(nM*s)", "origin": "assumed"},
        "kr1": {"value": 0.001, "units": "1/s", "origin": "assumed"},
        "kf2": {"value": 0.002, "units": "1/s", "origin": "assumed"}
      },
      "initials": {
        "E": {"value": 100, "units": "nM", "origin": "assumed"},
        "S": {"value": 50, "units": "nM", "origin": "assumed"},
        "ES": {"value": 0, "units": "nM", "origin": "assumed"},
        "P": {"value": 0, "units": "nM", "origin": "assumed"}
      },
      "conditions": [{"name": "control", "initials": {"S": {"value": 50, "units": "nM", "origin": "assumed"}}}]
    }
  }
}
```

Call `validate_model(check_generation=true)` and `generate_model`, passing the
session ID. Inspect equations and coverage; `Total_enzyme` should stay 100 in
this hypothetical model. Render with `visualize_model` before simulation.

```json
{
  "tool": "run_simulation",
  "arguments": {
    "session_id": "SESSION_ID",
    "scenario": {"name": "enzyme_example", "allow_placeholders": false}
  }
}
```

Export with `export_model_bundle`. Save the returned revision and artifact paths.
The bundle retains the synthetic evidence label, assumptions, and unresolved edges.

## Standalone document and line provenance

Create another session, then pass this argument object to `import_text`. It
preserves the document exactly. Line provenance attaches assumptions to the two
reaction lines. Put structural directives in the text in this mode.

```json
{
  "tool": "import_text",
  "arguments": {
    "session_id": "SESSION_ID",
    "biological_context": "Hypothetical enzyme example with illustrative values and unspecified units.",
    "text": "E + S <--> ES | kf=0.003, kr=0.001 | E=100, S=50, ES=0\nES --> E + P | kf=0.002 | P=0\n@obs Total_enzyme: u[E] + u[ES]\n@obs Product: u[P]\n@sim tspan: [0, 100]\n",
    "line_evidence": [
      {"line_number": 1, "assumption": "Assume reversible mass-action binding; supplied numbers are illustrative."},
      {"line_number": 2, "assumption": "Assume irreversible product release; supplied numbers are illustrative."}
    ]
  }
}
```

Validate, generate, visualize, simulate an explicit named scenario, and export as
above. Unspecified units remain null. To add literature provenance, first upsert
the actual evidence with `set_evidence`, then provide those IDs in `line_evidence`.
Replacing a document clears its old configuration and replaces the line mapping.
