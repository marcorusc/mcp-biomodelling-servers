"""Model text passed inline for servers that cannot read the caller's files."""

from pathlib import Path
from typing import Any

import pandas as pd
import pytest

from mcp_biomodelling_servers.inline_sources import resolve_source_path
from tests.test_maboss_mcp_errors import _call_tool as _maboss_call
from tests.test_maboss_mcp_errors import _run
from tests.test_neko_mcp_errors import _call_tool as _neko_call
from tests.test_neko_mcp_errors import _create_session, neko_server

SIF = "TP53\tstimulation\tMDM2\nMDM2\tinhibition\tTP53\n"
BNET = "targets, factors\nTP53, !MDM2\nMDM2, TP53\n"


def test_inline_text_is_stored_verbatim_and_content_addressed(tmp_path):
    first = resolve_source_path(None, SIF, artifact_dir=tmp_path, suffix=".sif",
                                path_name="sif_file", text_name="sif_text")
    again = resolve_source_path(None, SIF, artifact_dir=tmp_path, suffix=".sif",
                                path_name="sif_file", text_name="sif_text")
    assert first == again
    assert Path(first).read_text() == SIF
    assert Path(first).name.startswith("inline-") and first.endswith(".sif")
    assert resolve_source_path("given.sif", None, artifact_dir=tmp_path, suffix=".sif",
                               path_name="sif_file", text_name="sif_text") == "given.sif"
    with pytest.raises(ValueError, match="either sif_file or sif_text"):
        resolve_source_path("given.sif", SIF, artifact_dir=tmp_path, suffix=".sif",
                            path_name="sif_file", text_name="sif_text")


def test_create_network_builds_from_sif_text(monkeypatch: pytest.MonkeyPatch) -> None:
    seen: list[str] = []

    class SifNetwork:
        def __init__(self, *args: Any, sif_file: str, **kwargs: Any) -> None:
            seen.append(Path(sif_file).read_text())
            self.nodes = pd.DataFrame([{"Uniprot": "TP53", "Genesymbol": "TP53"},
                                       {"Uniprot": "MDM2", "Genesymbol": "MDM2"}])
            self.edges = pd.DataFrame([{"source": "TP53", "target": "MDM2"}])

        def convert_edgelist_into_genesymbol(self) -> pd.DataFrame:
            return pd.DataFrame([{"source": "TP53", "target": "MDM2", "Effect": "stimulation"}])

    monkeypatch.setattr(neko_server, "Network", SifNetwork)
    session_id = _create_session()
    result = _run(_neko_call("create_network", {
        "session_id": session_id, "list_of_initial_genes": [], "sif_text": SIF,
    }))
    assert result.is_error is False, result.content[0].text
    assert seen == [SIF]
    source = neko_server.session_manager.get_session(session_id).source_import.source
    assert source.name.startswith("inline-") and source.suffix == ".sif"


def test_create_network_rejects_path_and_text_together() -> None:
    session_id = _create_session()
    result = _run(_neko_call("create_network", {
        "session_id": session_id, "list_of_initial_genes": [],
        "sif_file": "/elsewhere/model.sif", "sif_text": SIF,
    }))
    assert result.is_error is True
    assert "either sif_file or sif_text" in result.content[0].text


def test_missing_sif_path_names_the_inline_alternative(tmp_path: Path) -> None:
    result = _run(_neko_call("create_network", {
        "list_of_initial_genes": [], "sif_file": str(tmp_path / "missing.sif"),
    }))
    assert result.is_error is True
    assert "sif_text" in result.content[0].text


def test_bnet_to_bnd_and_cfg_converts_bnet_text(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("PATH", "")  # prove the bundled engine, not a MaBoSS binary
    created = _run(_maboss_call("create_session", {}))
    session_id = created.structured_content["result"].split()[2]
    result = _run(_maboss_call("bnet_to_bnd_and_cfg", {
        "session_id": session_id, "bnet_text": BNET,
    }))
    assert result.is_error is False, result.content[0].text
    payload = result.structured_content
    assert Path(payload["input_bnet_path"]).read_text() == BNET
    assert "node TP53" in Path(payload["bnd_file"]["path"]).read_text()
    loaded = _run(_maboss_call("build_simulation", {"session_id": session_id}))
    assert loaded.is_error is False, loaded.content[0].text


def test_bnet_conversion_requires_a_source_and_names_bnet_text(tmp_path: Path) -> None:
    neither = _run(_maboss_call("bnet_to_bnd_and_cfg", {}))
    assert neither.is_error is True
    assert "bnet_path or bnet_text" in neither.content[0].text
    missing = _run(_maboss_call("bnet_to_bnd_and_cfg", {"bnet_path": str(tmp_path / "x.bnet")}))
    assert missing.is_error is True
    assert "pass its content as bnet_text" in missing.content[0].text
