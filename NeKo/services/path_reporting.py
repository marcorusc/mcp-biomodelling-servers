"""Compatibility adapter for the library's printed paths, with honest identity."""
import ast
from collections import defaultdict

from .network_analysis import node_records


def printed_paths(lines):
    paths = []
    for line in lines:
        try:
            path = ast.literal_eval(line)
        except (ValueError, SyntaxError):
            path = [part.strip() for part in line.split("->")] if "->" in line else None
        if not isinstance(path, (list, tuple)) or not path or not all(isinstance(item, str) for item in path):
            raise ValueError("Unrecognized NeKo path output; cannot count it as a discovered path: " + line[:200])
        paths.append(list(path))
    return paths


def path_identities(network, paths):
    aliases = defaultdict(list)
    for record in node_records(network):
        for alias in {record.gene_symbol, record.uniprot} - {None}:
            aliases[alias].append(record)
    result = []
    for path in paths:
        nodes = []
        for label in path:
            matches = aliases[label]
            if len(matches) == 1:
                match = matches[0]
                nodes.append({"label": match.gene_symbol or label, "uniprot": match.uniprot})
            else:
                nodes.append({"label": label, "uniprot": None,
                              "identity": "ambiguous" if matches else "unresolved"})
        result.append(nodes)
    return result
