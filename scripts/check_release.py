"""Check release metadata and built archives (run with Python 3.12+)."""

import argparse
import ast
import configparser
import json
import tarfile
import zipfile
from email.parser import BytesParser
from email.policy import default
from pathlib import Path, PurePosixPath

import tomllib

ROOT = Path(__file__).resolve().parent.parent
SERVERS = ("NeKo", "MaBoSS", "PhysiCell", "BioMASS")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tag", help="Require this Git tag to match the release")
    args = parser.parse_args()
    project = tomllib.loads((ROOT / "pyproject.toml").read_text())["project"]
    version = project["version"]
    if args.tag is not None and args.tag != f"v{version}":
        raise SystemExit(f"Tag {args.tag!r} does not match v{version}")
    readme = (ROOT / "README.md").read_text()
    manifests = []
    for server in SERVERS:
        manifest = json.loads((ROOT / server / "server.json").read_text())
        package = manifest["packages"][0]
        command = f"mcp-{server.lower()}-server"
        assert manifest["version"] == package["version"] == version, server
        assert package["identifier"] == project["name"], server
        assert package["registryType"] == "pypi", server
        assert package["transport"]["type"] == "stdio", server
        assert package["runtimeHint"] == "uvx", server
        assert package["runtimeArguments"] == [
            {
                "type": "named",
                "name": "--from",
                "value": f"{project['name']}=={version}",
            },
            {"type": "positional", "value": command},
        ], server
        assert f"<!-- mcp-name: {manifest['name']} -->" in readme, server
        plugin_name = server.lower()
        plugin_dir = ROOT / "plugins" / plugin_name
        launch = {
            "command": "uvx",
            "args": ["--from", f"{project['name']}=={version}", command],
        }
        for path, mcp_pointer, mcp_path, mcp_server in (
            ("plugin.json", None, "mcp.json", {"type": "stdio", **launch}),
            (
                ".claude-plugin/plugin.json",
                "./.claude-plugin/mcp.json",
                ".claude-plugin/mcp.json",
                launch,
            ),
        ):
            plugin = json.loads((plugin_dir / path).read_text())
            assert plugin["name"] == plugin_name, f"{server} {path}"
            assert plugin["version"] == version, f"{server} {path}"
            assert plugin.get("mcpServers") == mcp_pointer, f"{server} {path}"
            servers = json.loads((plugin_dir / mcp_path).read_text())["mcpServers"]
            assert servers == {plugin_name: mcp_server}, f"{server} {mcp_path}"
        manifests.append(manifest)
    for path in (".claude-plugin/marketplace.json", ".agents/plugins/marketplace.json"):
        entries = json.loads((ROOT / path).read_text())["plugins"]
        assert sorted(e["name"] for e in entries) == sorted(
            s.lower() for s in SERVERS
        ), path
        for entry in entries:
            # A version or a git ref/sha here would pin a stale plugin release.
            assert "version" not in entry, f"{path} {entry['name']}"
            assert entry["source"] == {
                "source": "git-subdir",
                "url": f"{project['urls']['Repository']}.git",
                "path": f"./plugins/{entry['name']}",
            }, f"{path} {entry['name']}"
    assert (
        f'__version__ = "{version}"'
        in (ROOT / "mcp_biomodelling_servers" / "__init__.py").read_text()
    )
    archives = sorted([*ROOT.glob("dist/*.whl"), *ROOT.glob("dist/*.tar.gz")])
    assert len(archives) == 2, "Build into a clean dist/ directory"
    for archive in archives:
        if archive.suffix == ".whl":
            with zipfile.ZipFile(archive) as wheel:
                members = wheel.namelist()
                metadata_path = next(
                    m for m in members if m.endswith(".dist-info/METADATA")
                )
                metadata = BytesParser(policy=default).parsebytes(
                    wheel.read(metadata_path)
                )
                assert metadata["Name"] == project["name"]
                assert metadata["Version"] == version
                assert metadata["Requires-Python"] == project["requires-python"]
                description = metadata.get_payload()
                for manifest in manifests:
                    assert f"mcp-name: {manifest['name']} -->" in description
                entrypoints_path = next(
                    m for m in members if m.endswith(".dist-info/entry_points.txt")
                )
                entrypoints = configparser.ConfigParser()
                entrypoints.read_string(wheel.read(entrypoints_path).decode())
                assert dict(entrypoints["console_scripts"]) == project["scripts"]
                for server in SERVERS:
                    assert f"mcp_biomodelling_servers/{server}/server.py" in members
                for name in (
                    "reaction_syntax",
                    "authoring_examples",
                    "network_to_reactions",
                    "model_editing",
                ):
                    assert f"mcp_biomodelling_servers/BioMASS/docs/{name}.md" in members
        else:
            with tarfile.open(archive) as sdist:
                members = sdist.getnames()
        for member in members:
            path = PurePosixPath(member)
            assert not (
                {"artifacts", "exports", "pypath_log", "__pycache__"} & set(path.parts)
            ), member
            assert path.name != ".env" and not path.name.startswith(".env."), member
            assert path.suffix not in {".pyc", ".log"}, member
        print(f"Checked {archive.name}")
    # The root module is a compatibility re-export, not a second implementation.
    shim = ast.parse((ROOT / "artifact_manager.py").read_text())
    assert len(shim.body) == 2 and isinstance(shim.body[0], ast.Expr)
    exports = shim.body[1]
    assert isinstance(exports, ast.ImportFrom)
    assert exports.module == "mcp_biomodelling_servers.artifact_manager"
    assert exports.level == 0
    assert {alias.name for alias in exports.names} == {
        "METADATA_FILENAME", "artifact_base", "artifact_bases", "clean_artifacts",
        "get_artifact_dir", "list_artifact_sessions", "list_artifacts",
        "read_session_meta", "safe_artifact_path", "write_session_meta",
    }
    assert all(alias.asname is None for alias in exports.names)
    print(
        f"Release {version}: four server manifests, plugin manifests and marketplaces, ownership markers and entry points agree."
    )


if __name__ == "__main__":
    main()
