"""Check namespace-remapped runtime imports from a built wheel, off the source path."""
import os
import subprocess
import sys
import tempfile
from pathlib import Path
from zipfile import ZipFile

MODULES = (
    "MaBoSS.services.model", "MaBoSS.services.runs", "MaBoSS.tools.curation",
    "NeKo.services.path_reporting", "PhysiCell.services.bundle",
    "PhysiCell.services.configuration", "PhysiCell.tools.bundle",
    "PhysiCell.tools.inputs", "boolean_readouts",
    "MaBoSS.server", "PhysiCell.server",
)


def main(wheel):
    with tempfile.TemporaryDirectory(prefix="wave2-wheel-") as temporary:
        root = Path(temporary)
        with ZipFile(wheel) as archive:
            archive.extractall(root)
        environment = os.environ.copy()
        environment["PYTHONPATH"] = str(root)
        environment["MPLBACKEND"] = "Agg"
        code = (
            "import importlib, pathlib\n"
            f"root = pathlib.Path({str(root)!r})\n"
            f"for name in {MODULES!r}:\n"
            "    module = importlib.import_module('mcp_biomodelling_servers.' + name)\n"
            "    assert pathlib.Path(module.__file__).is_relative_to(root), module.__file__\n"
            "print('All Wave 2 wheel modules and MaBoSS/PhysiCell server registrations imported')\n"
        )
        subprocess.run([sys.executable, "-c", code], cwd=root, env=environment,
                       check=True, timeout=60)


if __name__ == "__main__":
    main(Path(sys.argv[1]).resolve())
