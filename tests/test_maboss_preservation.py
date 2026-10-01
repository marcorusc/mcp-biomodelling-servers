"""Mocked file-backed results only: never import MaBoSS or invoke an engine."""
import ast
import asyncio
import hashlib
import importlib.util
import json
import logging
import math
import tempfile
import unittest
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pandas as pd

from mcp_biomodelling_servers.artifact_manager import (
    get_artifact_dir,
    list_artifacts,
    safe_artifact_path,
)

ROOT = Path(__file__).resolve().parents[1]
PACKAGE = ROOT / "MaBoSS"
spec = importlib.util.spec_from_file_location("preservation_artifacts", PACKAGE / "services/artifacts.py")
artifacts = importlib.util.module_from_spec(spec)
spec.loader.exec_module(artifacts)


class FakeResult:
    def __init__(self, directory, final_probability=0.75, optional=True):
        directory.mkdir()
        self._path = str(directory)
        self.prefix = "res"
        self._err = 0
        self._bnd = str(directory / "exact_engine_input.bnd")
        self._cfg = str(directory / "exact_engine_input.cfg")
        Path(self._bnd).write_bytes(b"node A { logic = B; }\n")
        Path(self._cfg).write_bytes(b"[A,B].istate = 1 [0,0];\nmax_time=3;\n")
        (directory / "res_probtraj.csv").write_bytes(b"EXACT RAW ENGINE BYTES\r\n0\t<nil>\t1\r\n")
        (directory / "res_run.txt").write_bytes(b"engine metadata\n")
        if optional:
            (directory / "res_fp.csv").write_bytes(b"fixed points\n")
            (directory / "res_statdist.csv").write_bytes(b"")
            (directory / "res_observed_graph.csv").write_bytes(b"raw graph\n")
        (directory / "unrelated.txt").write_bytes(b"not an engine artifact")
        self.states = pd.DataFrame(
            {"<nil>": [1.0, 0.6, 1-final_probability], "A": [0.0, 0.3, final_probability/2],
             "A -- B": [0.0, 0.1, final_probability/2]}, index=[0.0, 0.75, 2.5])
        self.nodes = pd.DataFrame(
            {"A": [0.0, 0.4, final_probability], "B": [0.0, 0.1, final_probability/2]},
            index=self.states.index)

    def get_probtraj_file(self):
        return str(Path(self._path) / "res_probtraj.csv")

    def get_states_probtraj(self):
        return self.states

    def get_nodes_probtraj(self):
        return self.nodes

    def get_last_states_probtraj(self):
        return self.states.iloc[[-1]]


class FakeContext:
    def __init__(self):
        self.progress = []

    async def report_progress(self, *args):
        self.progress.append(args)


async def immediate_worker(function):
    return function()


class PreservationTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="mock-maboss-preservation-")
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.sid = "fixture-session"
        self.server = self.root / "server"
        self.artifact_dir = get_artifact_dir(self.server, self.sid)

    def result(self, name="engine", **kwargs):
        return FakeResult(self.root / name, **kwargs)

    def load_run_tool(self, results):
        """Compile only the real tool function; all engine/session calls are mocks."""
        queue = list(results)
        session = SimpleNamespace(session_id=self.sid, result=None)
        calls = []

        def run():
            calls.append("run")
            return queue.pop(0)

        session.sim = SimpleNamespace(run=run, network=SimpleNamespace(get_output=lambda: ["A", "B"]))
        session.set_result = lambda value: setattr(session, "result", value)
        session.clear = lambda: setattr(session, "result", None)
        namespace = {
            "logger": logging.getLogger("mock-run"),
            "Field": lambda **kwargs: kwargs.get("default"),
            "session_manager": SimpleNamespace(session_scope=lambda sid: nullcontext()),
            "ensure_session": lambda sid: session,
            "_SERVER_ROOT": self.server,
            "get_artifact_dir": get_artifact_dir,
            "safe_artifact_path": safe_artifact_path,
            "_preserve_simulation_result": artifacts.preserve_simulation_result,
            "anyio": SimpleNamespace(to_thread=SimpleNamespace(run_sync=immediate_worker)),
            "MaBoSSSimulationRunResult": lambda **kwargs: kwargs,
            "artifact_file_summary": lambda path, session_id: {"path": str(path), "session_id": session_id},
            "structured_report": lambda text, payload: {"text": text, "payload": payload},
        }
        tree = ast.parse((PACKAGE / "server.py").read_text())
        function = next(n for n in tree.body if isinstance(n, ast.AsyncFunctionDef) and n.name == "run_simulation")
        function.decorator_list = []
        module = ast.Module(body=[ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0), function], type_ignores=[])
        exec(compile(ast.fix_missing_locations(module), str(PACKAGE / "server.py"), "exec"), namespace)
        return namespace["run_simulation"], session, calls

    def test_full_rows_times_exact_inputs_raw_files_and_inventory(self):
        result = self.result()
        path = artifacts.preserve_simulation_result(result, self.artifact_dir, self.sid)
        manifest = json.loads(path.read_text())
        self.assertEqual(manifest["session_id"], self.sid)
        self.assertEqual(manifest["state_timepoints"], [0.0, 0.75, 2.5])
        self.assertEqual(manifest["node_timepoints"], [0.0, 0.75, 2.5])
        prefix = manifest["run_id"]
        for suffix, expected in (("states.csv", result.states), ("nodes.csv", result.nodes)):
            csv = self.artifact_dir / (prefix + "." + suffix)
            self.assertEqual(csv.read_text().splitlines()[0].split(",")[0], "Time")
            actual = pd.read_csv(csv, index_col="Time")
            pd.testing.assert_frame_equal(actual, expected, check_names=False)
            self.assertEqual(len(actual), 3)
            self.assertFalse(actual.isna().any().any())
            self.assertTrue(all(math.isfinite(float(value)) for value in actual.index))
            self.assertTrue(all(math.isfinite(float(value)) for row in actual.itertuples(index=False) for value in row))
        for suffix, source in (("input.bnd", result._bnd), ("input.cfg", result._cfg)):
            self.assertEqual((self.artifact_dir / (prefix + "." + suffix)).read_bytes(), Path(source).read_bytes())
        raw = sorted(p for p in Path(result._path).iterdir() if p.name.startswith("res"))
        for source in raw:
            self.assertEqual((self.artifact_dir / (prefix + ".engine." + source.name)).read_bytes(), source.read_bytes())
        listed = list_artifacts(self.server, self.sid)
        self.assertEqual({p.name for p in listed}, {f["name"] for f in manifest["files"]} | {path.name})
        self.assertTrue(all(p.parent == self.artifact_dir for p in listed))
        for item in manifest["files"]:
            data = (self.artifact_dir / item["name"]).read_bytes()
            self.assertEqual(len(data), item["size_bytes"])
            self.assertEqual(hashlib.sha256(data).hexdigest(), item["sha256"])

    def test_two_runs_are_immutable_and_legacy_final_result_unchanged(self):
        first, second = self.result("first"), self.result("second", final_probability=0.5)
        Path(second._cfg).write_bytes(b"[A,B].istate = 1 [1,1];\nmax_time=3;\n")
        tool, session, calls = self.load_run_tool([first, second])
        ctx = FakeContext()
        response = asyncio.run(tool(ctx, self.sid))
        self.assertEqual(response["payload"]["trajectory_row_count"], 1)
        self.assertEqual(response["payload"]["trajectory_column_count"], 3)
        self.assertTrue(response["payload"]["result_available"])
        self.assertEqual(response["payload"]["result_file"]["path"], str(self.artifact_dir / "result.csv"))
        self.assertEqual((self.artifact_dir / "result.csv").read_text(), first.get_last_states_probtraj().to_csv(index=False))
        old = {p.name: p.read_bytes() for p in list_artifacts(self.server, self.sid) if p.name != "result.csv"}
        response2 = asyncio.run(tool(ctx, self.sid))
        self.assertEqual(response2["text"], response["text"])
        self.assertEqual((self.artifact_dir / "result.csv").read_text(), second.get_last_states_probtraj().to_csv(index=False))
        self.assertTrue(all((self.artifact_dir / name).read_bytes() == data for name, data in old.items()))
        self.assertEqual(len(list(self.artifact_dir.glob("run_*.json"))), 2)
        self.assertEqual({p.read_bytes() for p in self.artifact_dir.glob("run_*.input.cfg")},
                         {Path(first._cfg).read_bytes(), Path(second._cfg).read_bytes()})
        self.assertIs(session.result, second)
        self.assertEqual(calls, ["run", "run"])
        self.assertEqual(ctx.progress, [(0, 2), (2, 2), (0, 2), (2, 2)])

    def test_optional_engine_files_are_not_invented(self):
        manifest = json.loads(artifacts.preserve_simulation_result(self.result(optional=False), self.artifact_dir, self.sid).read_text())
        raw = [f["source_name"] for f in manifest["files"] if f["role"] == "engine_output"]
        self.assertEqual(raw, ["res_probtraj.csv", "res_run.txt"])

    def test_copy_failure_fails_run_without_retry_or_success_response(self):
        result = self.result()
        tool, session, calls = self.load_run_tool([result])
        (self.artifact_dir / "result.csv").write_text("previous final result\n")
        ctx = FakeContext()
        with patch.object(artifacts.shutil, "copyfile", side_effect=OSError("fixture disk failure")):
            with self.assertLogs("mock-run", level="WARNING"):
                with self.assertRaisesRegex(RuntimeError, "fixture disk failure"):
                    asyncio.run(tool(ctx, self.sid))
        self.assertEqual(calls, ["run"])
        self.assertIsNone(session.result)
        self.assertEqual(ctx.progress, [(0, 2)])
        self.assertEqual((self.artifact_dir / "result.csv").read_text(), "previous final result\n")
        self.assertEqual([p.name for p in list_artifacts(self.server, self.sid)], ["result.csv"])

    def test_namespace_collision_never_overwrites(self):
        result = self.result()
        with patch.object(artifacts.uuid, "uuid4", return_value=SimpleNamespace(hex="fixed")):
            artifacts.preserve_simulation_result(result, self.artifact_dir, self.sid)
            before = {p.name: p.read_bytes() for p in list_artifacts(self.server, self.sid)}
            with self.assertRaises(FileExistsError):
                artifacts.preserve_simulation_result(result, self.artifact_dir, self.sid)
        self.assertEqual(before, {p.name: p.read_bytes() for p in list_artifacts(self.server, self.sid)})

    def test_partial_publication_rolls_back_only_new_files(self):
        existing = self.artifact_dir / "existing.csv"
        existing.write_text("keep")
        actual_link = artifacts.link_artifact_without_overwrite
        count = 0

        def fail_second(source, destination):
            nonlocal count
            count += 1
            if count == 2:
                raise OSError("fixture publication failure")
            actual_link(source, destination)

        with patch.object(artifacts, "link_artifact_without_overwrite", side_effect=fail_second):
            with self.assertRaisesRegex(OSError, "fixture publication failure"):
                artifacts.preserve_simulation_result(self.result(), self.artifact_dir, self.sid)
        self.assertEqual(list_artifacts(self.server, self.sid), [existing])
        self.assertEqual(existing.read_text(), "keep")

    def test_missing_raw_trajectory_and_symlinks_rejected(self):
        result = self.result()
        raw = Path(result.get_probtraj_file())
        raw.unlink()
        with self.assertRaises(FileNotFoundError):
            artifacts.preserve_simulation_result(result, self.artifact_dir, self.sid)
        raw.symlink_to(Path(result._bnd))
        with self.assertRaisesRegex(ValueError, "outside its run directory"):
            artifacts.preserve_simulation_result(result, self.artifact_dir, self.sid)
        self.assertEqual(list_artifacts(self.server, self.sid), [])


if __name__ == "__main__":
    unittest.main()
