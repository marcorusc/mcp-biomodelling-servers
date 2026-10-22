"""Server startup: entry-point flags, offline NeKo start and headless MaBoSS plots."""

from __future__ import annotations

import os
import subprocess
import sys
import textwrap
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest

from mcp_biomodelling_servers import __version__
from mcp_biomodelling_servers import neko as neko_entry
from NeKo.services.nekomata_imports import DeferredAttribute

ROOT = Path(__file__).resolve().parent.parent
ENTRY_POINTS = {
    "neko": "NeKo",
    "maboss": "MaBoSS",
    "physicell": "PhysiCell",
    "biomass": "BioMASS",
}


def _run_python(source: str, cwd: Path) -> subprocess.CompletedProcess[str]:
    environment = os.environ.copy()
    environment["PYTHONPATH"] = str(ROOT)
    completed = subprocess.run(
        [sys.executable, "-c", textwrap.dedent(source)],
        cwd=cwd,
        env=environment,
        capture_output=True,
        text=True,
        check=False,
        timeout=300,
    )
    assert completed.returncode == 0, completed.stderr
    return completed


@pytest.mark.parametrize("module", ENTRY_POINTS)
@pytest.mark.parametrize("flag", ["--help", "-h", "--version"])
def test_informational_flags_exit_without_importing_a_server(
    module: str, flag: str, tmp_path: Path
) -> None:
    completed = _run_python(
        f"""
        import sys
        from mcp_biomodelling_servers import {module}
        {module}.main([{flag!r}])
        heavy = sorted(
            name for name in sys.modules
            if name.split(".")[0] in {{"NeKo", "MaBoSS", "PhysiCell", "BioMASS", "mcp",
                                        "neko", "maboss", "biomass", "pandas", "omnipath"}}
            or name.endswith(".server")
        )
        print("HEAVY", heavy)
        """,
        tmp_path,
    )
    command = f"mcp-{module}-server"
    if flag == "--version":
        assert completed.stdout.startswith(f"{command} (mcp-biomodelling-servers) {__version__}\n")
    else:
        assert completed.stdout.startswith(f"usage: {command} ")
        assert "MCP_BIOMODELLING_ARTIFACT_ROOT" in completed.stdout
    assert "HEAVY []" in completed.stdout


@pytest.mark.parametrize("module", ENTRY_POINTS)
@pytest.mark.parametrize("argv", [[], ["--transport", "stdio"]])
def test_other_arguments_still_start_the_server(
    module: str, argv: list[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    started = []
    package = ModuleType(f"mcp_biomodelling_servers.{ENTRY_POINTS[module]}")
    package.__path__ = []  # type: ignore[attr-defined]
    server = ModuleType(f"{package.__name__}.server")
    server.mcp = SimpleNamespace(run=lambda: started.append(module))  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, package.__name__, package)
    monkeypatch.setitem(sys.modules, server.__name__, server)
    monkeypatch.setenv("PYPATH_LOG", "/unused/pypath.log")
    entry = __import__(f"mcp_biomodelling_servers.{module}", fromlist=["main"])

    entry.main(argv)

    assert started == [module]


def test_pypath_log_defaults_under_the_artifact_root(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("MCP_BIOMODELLING_ARTIFACT_ROOT", str(tmp_path))
    monkeypatch.delenv("PYPATH_LOG", raising=False)

    neko_entry.default_pypath_log()

    log = Path(os.environ["PYPATH_LOG"])
    assert log.parent == tmp_path / "logs" / "pypath"
    assert log.name.startswith("pypath-") and log.name.endswith(f"-{os.getpid()}.log")


def test_pypath_log_keeps_an_explicit_location(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("PYPATH_LOG", "/chosen/by/user.log")

    neko_entry.default_pypath_log()

    assert os.environ["PYPATH_LOG"] == "/chosen/by/user.log"


def test_pypath_log_default_never_blocks_startup(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("MCP_BIOMODELLING_ARTIFACT_ROOT", "relative/root")
    monkeypatch.delenv("PYPATH_LOG", raising=False)

    neko_entry.default_pypath_log()

    assert "PYPATH_LOG" not in os.environ


def test_deferred_attribute_imports_once_on_first_use(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    imports = []

    class Target:
        history_dict = True

        def __init__(self, value: int) -> None:
            self.value = value

    module = ModuleType("deferred_fixture")
    module.Target = Target  # type: ignore[attr-defined]

    def import_module(name: str) -> ModuleType:
        imports.append(name)
        return module

    monkeypatch.setattr("importlib.import_module", import_module)
    deferred = DeferredAttribute("deferred_fixture", "Target")
    assert imports == []

    assert hasattr(deferred, "history_dict")
    assert deferred(3).value == 3
    assert deferred.resolve() is Target
    assert imports == ["deferred_fixture"]


def test_neko_server_lists_tools_offline_without_importing_nekomata(
    tmp_path: Path,
) -> None:
    completed = _run_python(
        """
        import asyncio
        import socket
        import sys

        def refuse(self, address):
            raise AssertionError(f"network access during startup: {address}")

        socket.socket.connect = refuse
        from mcp import Client
        from NeKo import server

        async def list_tools():
            async with Client(server.mcp) as client:
                return (await client.list_tools()).tools

        tools = asyncio.run(list_tools())
        loaded = sorted({name.split(".")[0] for name in sys.modules}
                        & {"neko", "omnipath", "pypath_common"})
        print("TOOLS", len(tools), "LOADED", loaded)
        """,
        tmp_path,
    )
    assert "LOADED []" in completed.stdout
    assert "TOOLS 0 " not in completed.stdout
    assert not (tmp_path / "pypath_log").exists()


def test_maboss_server_renders_plots_off_screen_even_with_a_gui_backend(
    tmp_path: Path,
) -> None:
    # A GUI backend aborts the process when a tool draws in a worker thread.
    completed = _run_python(
        """
        import os
        os.environ["MPLBACKEND"] = "TkAgg"
        import matplotlib
        from MaBoSS import server
        print("BACKEND", matplotlib.get_backend().lower())
        """,
        tmp_path,
    )
    assert "BACKEND agg" in completed.stdout
