"""Keep the offline authoring reference executable and discoverable."""

from __future__ import annotations

import asyncio
import json
import re
import zipfile

import numpy as np
import pandas as pd
import pytest
from mcp import Client

from BioMASS import server
from BioMASS.services.artifacts import job
from BioMASS.session_manager import BioMASSSessionManager
from BioMASS.tools.guidance import DOCS_DIRECTORY
from tests.test_biomass_workflow import handoff

RESOURCE_NAMES = (
    "reaction_syntax",
    "authoring_examples",
    "network_to_reactions",
    "model_editing",
)
SYNTAX = (DOCS_DIRECTORY / "reaction_syntax.md").read_text()
EXAMPLES = (DOCS_DIRECTORY / "authoring_examples.md").read_text()


def documented_models():
    cases = []
    for line in SYNTAX.splitlines():
        if line.startswith("| ") and "`" in line:
            _, name, statement, parameters, *_ = line.split("|")
            cases.append(
                pytest.param(
                    statement.strip().strip("`") + "\n",
                    {p.strip() + "1" for p in parameters.split(",")},
                    id=name.strip(),
                )
            )
    for i, text in enumerate(re.findall(r"```text2model\n(.*?)```", SYNTAX, re.S)):
        cases.append(pytest.param(text, None, id=f"syntax-block-{i + 1}"))
    return cases


@pytest.mark.parametrize("text,parameters", documented_models())
def test_documented_syntax_converts_in_real_worker(tmp_path, text, parameters):
    _, result = job(
        tmp_path, {"operation": "generate", "text": text, "configuration": {}}, 60
    )
    assert result["species"] and result["reactions"]
    if parameters is not None:
        assert set(result["parameters"]) == parameters


@pytest.mark.parametrize("mode", ["auto", "legacy"])
def test_authoring_resources_are_discoverable_and_linked(mode):
    async def check():
        async with Client(server.mcp, mode=mode) as client:
            listed = await client.list_resources()
            resources = {str(r.uri): r for r in listed.resources}
            manual = await client.read_resource("docs://biomass/agent_manual")
            prompt = await client.get_prompt("biomass_workflow_prompt")
            tools = {t.name: t for t in (await client.list_tools()).tools}
            for name in RESOURCE_NAMES:
                uri = f"docs://biomass/{name}"
                assert resources[uri].mime_type == "text/markdown"
                assert resources[uri].description
                text = (await client.read_resource(uri)).contents[0].text
                assert text == (DOCS_DIRECTORY / f"{name}.md").read_text()
                assert uri in client.instructions
                assert uri in manual.contents[0].text
                assert uri in prompt.messages[0].content.text
                assert uri in tools["set_reactions"].description
            assert "docs://biomass/reaction_syntax" in tools["import_text"].description

    asyncio.run(check())


def test_documented_tool_calls_generate_and_simulate_both_modes(tmp_path, monkeypatch):
    manager = BioMASSSessionManager(tmp_path / "sessions")
    monkeypatch.setattr(server, "session_manager", manager)
    exported = handoff(tmp_path)
    calls = [json.loads(s) for s in re.findall(r"```json\n(.*?)```", EXAMPLES, re.S)]
    assert [entry["tool"] for entry in calls] == [
        "set_evidence",
        "set_reactions",
        "configure_model",
        "run_simulation",
        "import_text",
    ]

    async def check():
        async with Client(server.mcp) as client:

            async def call(tool, arguments):
                response = await client.call_tool(tool, arguments)
                assert not response.is_error, response.content
                return response.structured_content

            sid = (await call("create_session", {}))["session_id"]
            imported = await call(
                "import_neko_handoff",
                {"manifest_path": exported.manifest_file.path, "session_id": sid},
            )
            edge_id = next(
                e["edge_id"]
                for e in imported["document"]["network"]["edges"]
                if e["source"] == "E"
            )

            def arguments(entry, session_id):
                return json.loads(
                    json.dumps(entry["arguments"])
                    .replace("SESSION_ID", session_id)
                    .replace("EDGE_ID", edge_id)
                )

            for entry in calls[:3]:
                await call(entry["tool"], arguments(entry, sid))

            for mode in ("records", "document"):
                if mode == "document":
                    sid = (await call("create_session", {}))["session_id"]
                    await call("import_text", arguments(calls[-1], sid))
                validation = await call(
                    "validate_model", {"session_id": sid, "check_generation": True}
                )
                assert validation["generation_valid"] is True
                generated = await call("generate_model", {"session_id": sid})
                if mode == "records":
                    assert len(generated["coverage"]["unresolved_edges"]) == 1
                    assert generated["coverage"]["assumed_reactions"] == [
                        "binding",
                        "catalysis",
                    ]
                    assert (
                        generated["details"]["parameters"]["kf1"]["units"] == "1/(nM*s)"
                    )
                result = await call("run_simulation", arguments(calls[3], sid))
                species = pd.read_csv(
                    next(
                        f["path"] for f in result["files"] if f["name"] == "species.csv"
                    )
                )
                assert np.isfinite(species[["E", "ES", "S", "P"]]).all().all()
                np.testing.assert_allclose(species.E + species.ES, 100, rtol=1e-7)
                bundle = await call("export_model_bundle", {"session_id": sid})
                with zipfile.ZipFile(bundle["details"]["bundle"]) as archive:
                    document = json.loads(archive.read("model/snapshot.json"))[
                        "document"
                    ]
                    if mode == "records":
                        assert document["evidence"]["enzyme_example"][
                            "source_identifiers"
                        ] == ["example:enzyme-chemistry"]
                        assert all(r["assumption"] for r in document["reactions"])
                    else:
                        assert document["text"] == calls[-1]["arguments"]["text"]
                        assert len(document["line_evidence"]) == 2

    asyncio.run(check())
