"""Render NeKo history diagrams with PyGraphviz's bundled Graphviz.

NeKo's own ``history_html`` pipes its Graphviz digraph through an external
``dot`` executable, which pip and uvx do not install. PyGraphviz wheels bundle
the Graphviz libraries, so the same digraph and per-state explanations render
in-process without any system Graphviz.
"""

import html
import json

import pygraphviz


def history_html(network, div_class: str = "neko-history-graph") -> str:
    """Return NeKo's history HTML: an inline SVG plus one explanation per state."""
    # NeKo's own JSON conversion, imported lazily so protocol-test stubs can load the server.
    from neko.core.history import json_value

    source = network.history_digraph(include_metadata=True).source
    svg = pygraphviz.AGraph(string=source).draw(format="svg", prog="dot").decode("utf-8")
    details = "".join(
        f'<details id="neko-state-{state["id"]}"><summary>State {state["id"]} explanation</summary>'
        f'<pre>{html.escape(json.dumps(json_value(state["metadata"]), indent=2))}</pre></details>'
        for state in network.list_states()  # NeKo's default is the full listing
    )
    return f'<div class="{html.escape(div_class, quote=True)}">{svg}{details}</div>'
