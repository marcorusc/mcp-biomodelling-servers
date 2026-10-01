"""Check release metadata and built archives (run with Python 3.12+)."""

import argparse
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
        manifests.append(manifest)
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
    assert (ROOT / "artifact_manager.py").read_bytes() == (
        ROOT / "mcp_biomodelling_servers" / "artifact_manager.py"
    ).read_bytes()
    print(
        f"Release {version}: four server manifests, ownership markers and entry points agree."
    )


if __name__ == "__main__":
    main()
