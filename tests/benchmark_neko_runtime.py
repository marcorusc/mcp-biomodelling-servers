"""Offline paired audit; run in fresh processes with explicit source paths.

Timings exclude external databases/identifier services. They describe synthetic
fixtures, not scientific workload throughput. Heap tracing runs separately from
timing. This is a measurement script, not a timing-sensitive CI test.
"""

from __future__ import annotations

import argparse
import asyncio
import gc
import json
import os
import pickle
import resource
import statistics
import sys
import time
import tracemalloc
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch


def timed(operation, repeats=5):
    values = []
    for _ in range(repeats):
        start = time.perf_counter()
        operation()
        values.append((time.perf_counter() - start) * 1000)
    return {"median_ms": statistics.median(values), "samples_ms": values}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True)
    parser.add_argument("--wrapper", required=True)
    parser.add_argument("--nodes", type=int, default=100)
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--startup-only", action="store_true")
    parser.add_argument("--measure-export", action="store_true")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    sys.path[:0] = [args.wrapper, args.source]
    # OmniPath otherwise performs remote discovery during import. Use its
    # supported offline option with a process-local, disposable config location.
    with TemporaryDirectory() as config_root:
        config = Path(config_root) / ".config" / "omnipathdb.ini"
        config.parent.mkdir()
        config.write_text("[https://omnipathdb.org]\nautoload = false\ncache_dir = memory\nlicense = None\n")
        with patch.object(Path, "home", return_value=Path(config_root)):
            start = time.perf_counter()
            import neko.core.network as library
            from neko._outputs.exports import Exports
            library_import_ms = (time.perf_counter() - start) * 1000
            start = time.perf_counter()
            from mcp import Client

            from NeKo import server
            from NeKo.session_manager import session_manager
            wrapper_import_ms = (time.perf_counter() - start) * 1000
    import pandas as pd

    if args.startup_only:
        Path(args.output).write_text(json.dumps({
            "library_import_ms": library_import_ms,
            "additional_wrapper_import_ms": wrapper_import_ms,
            "process_peak_rss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        }, indent=2) + "\n")
        return
    library.mapping_node_identifier = lambda x: [None, x, x]
    library.check_gene_list_format = lambda _: False
    pairs = [(f"N{i}", f"N{(i + gap) % args.nodes}")
             for i in range(args.nodes) for gap in (1, 5)]
    resources = pd.DataFrame([{
        "source": a, "target": b, "is_directed": True,
        "is_stimulation": True, "is_inhibition": False,
        "form_complex": False, "references": "123456",
    } for a, b in pairs])
    report = {"nodes": args.nodes, "edges": len(pairs), "steps": args.steps,
              "source": args.source, "wrapper": args.wrapper,
              "library_import_ms": library_import_ms,
              "additional_wrapper_import_ms": wrapper_import_ms}
    with TemporaryDirectory() as directory:
        os.environ["MCP_BIOMODELLING_ARTIFACT_ROOT"] = directory
        sif = Path(directory) / "fixture.sif"
        sif.write_text("".join(f"{a}\t1\t{b}\n" for a, b in pairs))

        def build():
            return library.Network(sif_file=str(sif), resources=resources)

        report["sif_construct"] = timed(build, 3)
        network = build()
        report["noop_existing_node"] = timed(lambda: network.add_node("N1"))
        # Same successful mutation in both releases; each creates one state.
        previous_label = "N0"
        def rename():
            nonlocal previous_label
            label = "Renamed" if previous_label == "N0" else "N0"
            network.modify_node_name(previous_label, label)
            previous_label = label

        report["node_label_change"] = timed(rename, args.steps)
        report["state_count"] = len(network.list_states())
        report["library_list_states"] = timed(network.list_states)
        report["pickle_bytes"] = len(pickle.dumps(network))
        report["library_sif_export"] = timed(
            lambda: Exports(network).export_sif(str(Path(directory) / "export.sif")), 3,
        )
        if hasattr(network, "history_dict"):
            report["library_history_dict"] = timed(network.history_dict)
            history = network.history_dict()
            report["history_json_bytes"] = len(json.dumps(history).encode())
            report["history_indented_bytes"] = len(json.dumps(history, indent=2).encode())
            report["history_compact_bytes"] = len(json.dumps(history, separators=(",", ":")).encode())
            del history
        sid = session_manager.create_session()
        session_manager.get_session(sid).set_network(network)

        async def protocol():
            async with Client(server.mcp) as client:
                result = {}
                for name in ("status", "list_network_history", "list_genes_and_interactions"):
                    samples = []
                    for _ in range(5):
                        start = time.perf_counter()
                        response = await client.call_tool(name, {"session_id": sid})
                        samples.append((time.perf_counter() - start) * 1000)
                        assert not response.is_error, response
                    raw = json.dumps(response.model_dump(
                        mode="json", by_alias=True, exclude_none=True,
                    ), separators=(",", ":")).encode()
                    result[name] = {"median_ms": statistics.median(samples),
                                    "samples_ms": samples, "response_bytes": len(raw)}
                if args.measure_export:
                    samples = []
                    for _ in range(3):
                        start = time.perf_counter()
                        response = await client.call_tool("export_network", {
                            "session_id": sid, "format": "bnet",
                        })
                        samples.append((time.perf_counter() - start) * 1000)
                        assert not response.is_error, response
                    result["export_bnet"] = {"median_ms": statistics.median(samples),
                                            "samples_ms": samples}
                return result
        report["protocol"] = asyncio.run(protocol())
        session_manager.delete_session(sid)
        del network
        gc.collect()
        tracemalloc.start()
        network = build()
        initial_heap, _ = tracemalloc.get_traced_memory()
        previous_label = "N0"
        for _ in range(args.steps):
            rename()
        retained_heap, peak_heap = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        report["traced_heap_bytes"] = {
            "constructed": initial_heap, "with_history": retained_heap,
            "peak": peak_heap, "history_growth": retained_heap - initial_heap,
        }
    report["process_peak_rss_kib"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    Path(args.output).write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
