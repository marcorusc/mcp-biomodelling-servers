# Reaction syntax accepted by this MCP server

Target: BioMASS 0.14.x; examples checked against 0.14.0 and the server's
expression validator. Read this before calling `set_reactions` or `import_text`.
This is a reference for the server's supported subset, not arbitrary Python
or unrestricted natural language. Examples describe hypothetical mechanisms.

## Reaction lines

A line contains `statement | parameters | initial conditions`. The two value
sections are optional. Use spaces around arrows and between words. Prefer the
canonical ASCII arrows `-->` and `<-->`. Species names are case-sensitive:
start with a letter, then letters, digits, or single underscores; no Python
keywords or double underscores. Names denote species/states, not gene IDs that
are automatically resolved. Choose separate names for modified forms.

In the table, brackets denote species concentrations; the rate expressions
explain the generated kinetics and are not text to paste into a reaction.
Parameters listed here are the local names used after the first pipe.

| Rule | Statement | Parameters | Net rate |
|---|---|---|---|
| Conversion | `A --> B` | kf | kf[A] |
| Association | `A + B <--> AB` | kf, kr | kf[A][B] - kr[AB] |
| Binding | `A binds B <--> AB` | kf, kr | kf[A][B] - kr[AB] |
| Dimerization | `A dimerizes <--> AA` | kf, kr | kf[A]^2 - kr[AA] |
| Dissociation | `AB dissociates to A and B` | kf, kr | kf[AB] - kr[A][B] |
| Irreversible dissociation | `AB --> A + B` | kf | kf[AB] |
| State conversion | `uA is phosphorylated <--> pA` | kf, kr | kf[uA] - kr[pA] |
| Saturating dephosphorylation | `pA is dephosphorylated --> uA` | V, K | V[pA]/(K + [pA]) |
| Kinase-mediated phosphorylation | `Kinase phosphorylates uA --> pA` | V, K | V[Kinase][uA]/(K + [uA]) |
| Phosphatase-mediated dephosphorylation | `Phosphatase dephosphorylates pA --> uA` | V, K | V[Phosphatase][pA]/(K + [pA]) |
| Transcription | `TF transcribes mRNA` | V, K, n | V[TF]^n/(K^n + [TF]^n) |
| Repressed transcription | `TF transcribes mRNA, repressed by Repressor` | V, K, n, KF, nF | V[TF]^n/(K^n + [TF]^n + ([Repressor]/KF)^nF) |
| Joint transcription | `TF1 & TF2 transcribe mRNA` | V, K, n | V([TF1][TF2])^n/(K^n + ([TF1][TF2])^n) |
| Regulated synthesis | `Template synthesizes Protein` | kf | kf[Template] |
| Basal synthesis | `A is synthesized` | kf | kf |
| Regulated degradation | `Protease degrades A` | kf | kf[Protease][A] |
| Basal degradation | `A is degraded` | kf | kf[A] |
| Transport | `Acyt translocates from cytoplasm to nucleus (2, 1) <--> Anuc` | kf, kr | kf[Acyt] - kr(1/2)[Anuc] |

Dimerization consumes two A per AA formed. Kinases, phosphatases, transcription
factors, templates, and proteases are modifiers in these rules: the reaction
does not consume them. Their abundance changes only through other reactions.
The dissociation sentence without an arrow is reversible; use the explicit
irreversible form when justified. For transport, parentheses contain numeric
compartment volumes, not names of parameters. The destination derivative is
scaled by source volume / destination volume; omitted volumes default to one.

The transcription repression formula above is BioMASS's actual rule. Do not
silently substitute a different inhibition formula. `A activates B` and
`A inhibits B` are not canonical reaction statements; select a justified
mechanism using docs://biomass/network_to_reactions.

## Values, fixed species, and sharing

Supply finite nonnegative numbers, separated by commas. Scientific notation is
accepted. Missing parameter values default to one and missing initial values to
zero; the server labels them placeholders, including unspecified zero values.

```text2model
E + S <--> ES | kf=0.003, kr=0.001 | E=100, S=50, ES=0
ES --> E + P | kf=0.002 | P=0
```

For built-in rules, a parameter's generated name appends its text line number:
the example has `kf1`, `kr1`, and `kf2`. Comments and blank lines count toward
line numbering. Use generated names in configuration and scenario overrides;
inspect `generate_model` results or the revision resource for the actual names.

`fixed` holds a species constant during simulation, rather than merely setting
its initial value. `const` marks a parameter as fixed for upstream estimation;
this server does not perform estimation. Both need scientific justification.

```text2model
Ligand binds Receptor <--> LR | const kf=0.01, kr=0.1 | fixed Ligand=10, Receptor=1, LR=0
```

In standalone documents, the parameter section can reference an earlier
reaction's line number to share its parameters:

```text2model
A --> B | kf=0.1 | A=1, B=0
B --> C | 1 | C=0
```

In reaction-record mode, use `share_parameters_with` with an earlier record ID
instead; numeric sharing is rejected there. Use compatible reaction types.
Override a shared parameter's source, not its dependent generated parameter.
Reordering records preserves ID-based sharing but changes generated names.

## Custom kinetics

Use `@rxn` with an explicit forward reaction and a rate expression. `p[name]`
denotes a parameter and `u[name]` a species. Custom parameter names do not get
a line-number suffix. Species used only as modifiers must still be declared;
listing a modifier on both sides makes its stoichiometric change zero.

```text2model
@rxn E + S --> E + P: p[kcat] * u[E] * u[S] / (p[Km] + u[S]) | kcat=0.2, Km=1 | E=1, S=10, P=0
```

`0` denotes no species in custom synthesis/degradation reactions:

```text2model
@rxn A --> 0: p[kdeg] * u[A] | kdeg=0.1 | A=1
```

Expressions allow finite numeric constants, parentheses, `+`, `-`, `*`, `/`,
`**` (or `^`), and the allowed symbol references. Calls such as `exp(...)`,
attributes, string indexing, comprehensions, imports, and custom Python
packages are rejected. Expressions must also make numerical sense over the
scenario; syntax checking cannot establish that a denominator stays nonzero.
Custom vocabulary registration is not exposed by this server.

## Observables and simulation directives

Standalone documents support the following complete example. In record mode,
put the reaction in `set_reactions` and the observables, time span, and conditions
in `configure_model`; do not put `@obs`, `@sim`, or `@add` in reaction records.

```text2model
A --> B | kf=0.1 | A=1, B=0
@obs Total: u[A] + u[B]
@sim tspan: [0, 20]
@sim condition control: init[A] = 1
@sim condition fast: p[kf1] = 0.2; init[A] = 1
```

`@obs Name: expression` uses `u` and `p`; condition assignments use `init` and
`p`. `@sim unperturbed: init[A] = 0` is optional steady-state preparation, not a
second simulation condition. Each condition starts from fresh defaults.
`@add species Name` and `@add param Name` declare extra symbols in standalone
documents; declarations do not provide numerical values. Each document needs
at least one reaction. Use integer time bounds with 0 <= start < stop and no
more than 10001 samples. Maximum: 20 conditions, 2000 lines, 1 MiB of text,
500 generated species, and 1000 reactions.

Read docs://biomass/authoring_examples for tool argument objects and evidence
links. After authoring, call `validate_model(check_generation=true)` to test
conversion; a syntax-only result does not prove that BioMASS recognizes a rule.

## Sources and scope

Canonical rules and kinetics were checked against the installed BioMASS 0.14.0
`construction/reaction_rules.py`, and the restrictions against this server's
`services/authoring.py`. Tests convert the table statements and fenced model
examples. Upstream references (may describe capabilities outside this server):

- https://biomass-core.readthedocs.io/en/latest/api/reaction_rules.html
- https://pasmopy.readthedocs.io/en/latest/model_development.html
