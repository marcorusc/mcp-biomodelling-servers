"""Validate or simulate one MaBoSS model with cmaboss in a child process.

Usage: ``python engine_worker.py check|run <directory> <cmaboss module>``.
The directory holds the exact ``input.bnd`` and ``input.cfg`` text. ``run``
writes the engine's standard ``res_*`` output files next to them. Exit status
2 means cmaboss rejected the model; its message is on stderr.
"""

import sys
from pathlib import Path

# Run as a script: keep the sibling service modules off the import path.
if sys.path and Path(sys.path[0] or ".").resolve() == Path(__file__).resolve().parent:
    del sys.path[0]

import importlib  # noqa: E402

ENGINES = {"cmaboss", "cmaboss_128n", "cmaboss_256n", "cmaboss_512n", "cmaboss_1024n"}
REJECTED = 2


def main(action: str, directory: str, module_name: str) -> int:
    if action not in ("check", "run") or module_name not in ENGINES:
        raise ValueError(f"Unsupported engine request: {action} {module_name}")
    root = Path(directory)
    engine = importlib.import_module(module_name)
    try:
        simulation = engine.MaBoSSSim(
            network_str=(root / "input.bnd").read_text(encoding="utf-8"),
            config_str=(root / "input.cfg").read_text(encoding="utf-8"),
        )
    except engine.BNException as exc:
        print(str(exc).strip(), file=sys.stderr)
        return REJECTED
    if action == "run":
        result = simulation.run(False)
        result.display_run(str(root / "res_run.txt"))
        result.display_probtraj(str(root / "res_probtraj.csv"))
        result.display_fp(str(root / "res_fp.csv"))
        result.display_statdist(str(root / "res_statdist.csv"))
    return 0


if __name__ == "__main__":
    sys.exit(main(*sys.argv[1:]))
