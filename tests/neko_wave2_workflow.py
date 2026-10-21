"""Real NeKo/MCP workflow, executed in a fresh interpreter by the runtime test."""

import asyncio
import copy
import json
import os
import sys
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace

import neko.core.network as library
import pandas as pd
from mcp import Client

if not hasattr(library.Network, "history_dict"):
    sys.exit(77)

from mcp_biomodelling_servers.handoff import (  # noqa: E402
    load_handoff_manifest,
)
from mcp_biomodelling_servers.import_lineage import (  # noqa: E402
    SourceExport,
    capture_import,
    verify_source_ancestry,
)
from mcp_biomodelling_servers.ode_handoff import read_ode_handoff  # noqa: E402
from NeKo import server  # noqa: E402
from NeKo.session_manager import session_manager  # noqa: E402


def edit(operation="add", source="A", target="C", effect="inhibition", pmids=None):
    return {"operation": operation, "source": source, "target": target, "effect": effect,
            "rationale": "Authorized fixture correction", "evidence": {
                "summary": "Synthetic test evidence, not a scientific claim",
                "pmids": ["123456"] if pmids is None else pmids,
                "citations": ["Synthetic publication fixture"],
            }}


async def workflow(directory):
    os.environ["MCP_BIOMODELLING_ARTIFACT_ROOT"] = str(directory)
    library.mapping_node_identifier = lambda x: [
        None, x.removeprefix("ID:"), x if x.startswith("ID:") else "ID:" + x,
    ]
    library.check_gene_list_format = lambda _: True
    resources = pd.DataFrame([{
        "source": "ID:" + a, "target": "ID:" + b, "is_directed": True,
        "is_stimulation": True, "is_inhibition": False, "form_complex": False,
        "references": str(100 + i),
    } for i, (a, b) in enumerate([("A", "B"), ("B", "C"), ("C", "D")])])
    server.signor = lambda: SimpleNamespace(build=lambda: None, interactions=resources)
    sid = session_manager.create_session()

    async with Client(server.mcp) as client:
        async def call(name, **kwargs):
            result = await client.call_tool(name, {"session_id": sid, **kwargs})
            assert not result.is_error, (name, result.content)
            return result.structured_content

        await call("create_network", list_of_initial_genes=["A", "B", "C"],
                   database="signor", max_len=1, consensus=False)
        sess = session_manager.get_session(sid)
        network = sess.network
        status = await call("status")
        assert status["database_name"] == "SIGNOR"
        sess.default_params["database"] = "omnipath"
        assert (await call("status"))["database_name"] == "SIGNOR"
        construction = network.get_state()["metadata"]["transition"]
        assert construction["operations"] and construction["inclusion_reasons"]
        base = network.current_state_id
        expected = {"expected_network_id": status["network_id"], "expected_state_id": base}
        original = copy.deepcopy(network.history_dict())
        failed = await client.call_tool("edit_interactions", {
            "session_id": sid, **expected, "edits": [edit(), edit(source="MISSING")],
        })
        assert failed.is_error and "endpoint_not_found" in failed.content[0].text
        assert network.history_dict() == original
        added = await call("edit_interactions", **expected, edits=[edit()])
        manual_state = added["state_id"]
        assert added["changes"][0]["source"] == "ID:A"
        assert added["changes"][0]["origin"] == "manual"
        resource = await client.read_resource(added["details_uri"])
        details = json.loads(resource.contents[0].text)
        edge = next(e for e in details["state"]["edges"]["records"] if e["source"] == "ID:A" and e["target"] == "ID:C")
        assert edge["Provenance"]["edits"][0]["evidence"]["pmids"] == ["123456"]
        assert edge["Provenance"]["original"]["database_name"] is None
        stale = await client.call_tool("edit_interactions", {
            "session_id": sid, **expected, "edits": [edit("set_effect")],
        })
        assert stale.is_error and "stale_state" in stale.content[0].text
        unchanged = await call("edit_interactions", expected_network_id=status["network_id"],
                               expected_state_id=manual_state, edits=[edit("set_effect")])
        assert not unchanged["changed"] and unchanged["state_id"] == manual_state
        inventory = await call("list_genes_and_interactions", verbosity="full")
        assert any(e.get("provenance", {}).get("origin") == "manual" for e in inventory["interactions"])
        refs = await call("get_references", node1="A", node2="C")
        assert refs["interactions"][0]["provenance"]["edits"][0]["evidence"]["pmids"] == ["123456"]

        await call("navigate_network_history", action="checkout", state_id=base)
        alternate = await call("edit_interactions", **expected,
                               edits=[edit(effect="stimulation", pmids=[])])
        branch = alternate["state_id"]
        assert branch != manual_state
        comparison = await call("compare_network_states", state_a=manual_state, state_b=branch)
        assert comparison["changed_edges"][0]["before"]["Provenance"]["origin"] == "manual"
        assert set(network.history_graph().successors(base)) == {manual_state, branch}
        for obj in network.edges_as_objects():
            assert obj.metadata["Provenance"]
        # Resource-origin effect changes preserve original References separately.
        revised = await call("edit_interactions", expected_network_id=status["network_id"],
                             expected_state_id=branch, edits=[edit("set_effect", target="B")])
        effect_state = revised["state_id"]
        assert revised["changes"][0]["origin"] == "resource"
        row = next(r for r in network.get_state()["edges"]["records"] if r["source"] == "ID:A" and r["target"] == "ID:B")
        assert row["References"] == "100" and row["Provenance"]["edits"][-1]["evidence"]["pmids"] == ["123456"]

        history_export = await call("export_network", format="history")
        restored = library.Network.from_history(history_export["file"]["path"])
        assert restored.history_dict() == network.history_dict()
        restored.checkout(manual_state)
        assert set(restored.history_graph().successors(base)) == {manual_state, branch}
        html = await client.read_resource(f"neko://session/{sid}/history")
        assert "<svg" in html.contents[0].text and "explanation" in html.contents[0].text

        handoff = await call("export_neko_handoff", biological_context="Synthetic fixture", output_nodes=["C"])
        manifest_path = Path(handoff["manifest_file"]["path"])
        manifest = load_handoff_manifest(manifest_path)
        assert manifest.schema_version == "2.1" and manifest.history_state_id == effect_state
        evidence_path = Path(manifest.curation_file.path)
        evidence = json.loads(evidence_path.read_bytes())
        assert evidence["mcp"]["network_id"] == status["network_id"]
        assert evidence["history"]["database_name"] == "SIGNOR"
        projection = evidence["export"]["library_projection"]
        assert projection["sha256"] == manifest.bnet_file.sha256
        assert projection["file"] == manifest.bnet_file.name
        assert library.Network.from_history(evidence_path).history_dict() == network.history_dict()
        # Model conversion uses the existing supported pyMaBoSS API, no engine run.
        import maboss
        maboss.bnet_to_bnd_and_cfg(manifest.bnet_file.path, str(directory / "converted.bnd"), str(directory / "converted.cfg"))
        assert (directory / "converted.bnd").is_file()
        # Follow the real typed handoff through readout curation and portable export.
        from MaBoSS import server as boolean
        from PhysiCell import server as multicellular
        async def checked(client, name, **args):
            response = await client.call_tool(name, args)
            assert not response.is_error, response.content
            return response.structured_content
        async with Client(boolean.mcp) as boolean_client:
            bsid = boolean.session_manager.create_session()
            await checked(boolean_client, "import_neko_handoff", session_id=bsid, manifest_path=str(manifest_path))
            inspected = await checked(boolean_client, "get_maboss_nodes", session_id=bsid)
            core = {n: node.logExp for n, node in boolean.session_manager.get_session(bsid).sim.network.items()}
            await checked(boolean_client, "edit_maboss_readouts", session_id=bsid,
                expected_revision=inspected["model_revision"], edits=[{
                    "operation": "add", "node": "Readout", "rule": "C", "explanation": "Explicit synthetic readout fixture",
                    "rate_up": 1, "rate_down": 1, "initial_probability_on": 0}])
            assert all(boolean.session_manager.get_session(bsid).sim.network[n].logExp == rule for n, rule in core.items())
            await checked(boolean_client, "set_maboss_output_nodes", session_id=bsid, output_nodes=["C", "Readout"])
            exported = await checked(boolean_client, "export_maboss_handoff", session_id=bsid,
                                     target_cell_type="tumour", include_result=False)
            mb_manifest_path = exported["manifest_file"]["path"]
            mb_manifest = load_handoff_manifest(mb_manifest_path)
            assert mb_manifest.schema_version == "2.1" and mb_manifest.readouts["Readout"]["origin"] == "readout"
            assert mb_manifest.parent_manifest.sha256
        async with Client(multicellular.mcp) as pc_client:
            psid = multicellular.session_manager.create_session()
            await checked(pc_client, "create_simulation_domain", session_id=psid, domain_x=100, domain_y=100)
            await checked(pc_client, "add_single_cell_type", session_id=psid, cell_type_name="tumour")
            await checked(pc_client, "import_maboss_handoff", session_id=psid, manifest_path=mb_manifest_path)
            await checked(pc_client, "add_physiboss_output_link", session_id=psid, cell_type="tumour",
                          boolean_node="Readout", physicell_behavior="cycle entry", value=0, base_value=0, steepness=0)
            bundle = await checked(pc_client, "export_configuration_bundle", session_id=psid)
            assert Path(bundle["bundle"]).is_file()
        raw = evidence_path.read_bytes()
        evidence_path.write_bytes(raw + b" ")
        try:
            load_handoff_manifest(manifest_path)
        except ValueError:
            pass
        else:
            raise AssertionError("Tampered provenance accepted")
        evidence_path.write_bytes(raw)

        ode = await call("export_biomass_handoff", biological_context="Synthetic fixture")
        ode_manifest, graph = read_ode_handoff(ode["manifest_file"]["path"])
        assert ode_manifest.schema_version == "2.0"
        assert any(e.provenance and e.provenance["origin"] == "manual" for e in graph.edges)
        # Source lineage embeds the provenance bytes, so archival remains portable.
        sif = await call("export_network", format="sif")
        sif_path = Path(sif["file"]["path"])
        sidecar = sif_path.with_name(sif_path.name + ".lineage.json")
        exported = SourceExport.model_validate_json(sidecar.read_bytes())
        assert exported.schema_version == "2.1"
        imported = capture_import(sif_path, server="NeKo", session_id="next-session", parent_manifest=str(sidecar))
        sif_path.unlink()
        sidecar.unlink()
        Path(exported.curation_file.path).unlink()
        verify_source_ancestry([imported])
        await call("set_network_history_limit", max_states=2)
        summary = await call("list_network_history")
        assert summary["pruned_state_count"] > 0
        assert any(s["ancestry_gaps"] for s in summary["states"])
        print("REAL_NEKO_WORKFLOW_OK")


with TemporaryDirectory() as root:
    asyncio.run(workflow(Path(root)))
