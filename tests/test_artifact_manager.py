"""Unit tests for the shared artifact_manager module."""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from artifact_manager import (
    clean_artifacts,
    get_artifact_dir,
    list_artifacts,
    safe_artifact_path,
)

# ---------------------------------------------------------------------------
# get_artifact_dir
# ---------------------------------------------------------------------------

class TestGetArtifactDir:
    def test_creates_directory(self, tmp_path):
        art_dir = get_artifact_dir(tmp_path, "session_1")
        assert art_dir.exists()
        assert art_dir.is_dir()

    def test_layout(self, tmp_path):
        art_dir = get_artifact_dir(tmp_path, "session_42")
        assert art_dir == tmp_path / "artifacts" / "session_42"

    def test_idempotent(self, tmp_path):
        d1 = get_artifact_dir(tmp_path, "session_1")
        d2 = get_artifact_dir(tmp_path, "session_1")
        assert d1 == d2

    @pytest.mark.parametrize("bad_id", ["", ".", "..", "a/b", "a\\b"])
    def test_rejects_invalid_session_id(self, tmp_path, bad_id):
        with pytest.raises(ValueError):
            get_artifact_dir(tmp_path, bad_id)


# ---------------------------------------------------------------------------
# safe_artifact_path
# ---------------------------------------------------------------------------

class TestSafeArtifactPath:
    def test_simple_filename(self, tmp_path):
        art_dir = get_artifact_dir(tmp_path, "s1")
        p = safe_artifact_path(art_dir, "output.bnd")
        assert p.name == "output.bnd"
        assert p.parent.resolve() == art_dir.resolve()

    def test_strips_directory_prefix(self, tmp_path):
        art_dir = get_artifact_dir(tmp_path, "s1")
        p = safe_artifact_path(art_dir, "subdir/output.cfg")
        assert p.name == "output.cfg"

    def test_rejects_empty_filename(self, tmp_path):
        art_dir = get_artifact_dir(tmp_path, "s1")
        with pytest.raises(ValueError):
            safe_artifact_path(art_dir, "")

    def test_rejects_traversal_attempt(self, tmp_path):
        # Path.name of "../../etc/passwd" is "passwd" – should be safe after stripping
        # but we keep this to document the stripping behaviour is the defence
        art_dir = get_artifact_dir(tmp_path, "s1")
        p = safe_artifact_path(art_dir, "../../etc/passwd")
        assert p.name == "passwd"
        assert str(p).startswith(str(art_dir.resolve()))


# ---------------------------------------------------------------------------
# list_artifacts
# ---------------------------------------------------------------------------

class TestListArtifacts:
    def test_empty_when_no_artifacts_dir(self, tmp_path):
        assert list_artifacts(tmp_path, "s1") == []

    def test_empty_when_session_dir_missing(self, tmp_path):
        get_artifact_dir(tmp_path, "other_session")
        assert list_artifacts(tmp_path, "s1") == []

    def test_lists_files_for_session(self, tmp_path):
        art_dir = get_artifact_dir(tmp_path, "s1")
        (art_dir / "a.txt").write_text("a")
        (art_dir / "b.txt").write_text("b")
        files = list_artifacts(tmp_path, "s1")
        assert len(files) == 2
        assert all(f.is_file() for f in files)

    def test_lists_all_sessions(self, tmp_path):
        for sid in ("s1", "s2"):
            d = get_artifact_dir(tmp_path, sid)
            (d / "file.txt").write_text("x")
        files = list_artifacts(tmp_path, session_id=None)
        assert len(files) == 2

    def test_does_not_include_subdirectories(self, tmp_path):
        art_dir = get_artifact_dir(tmp_path, "s1")
        (art_dir / "sub").mkdir()
        (art_dir / "file.txt").write_text("x")
        files = list_artifacts(tmp_path, "s1")
        assert len(files) == 1

    def test_sorted_order(self, tmp_path):
        art_dir = get_artifact_dir(tmp_path, "s1")
        for name in ("c.txt", "a.txt", "b.txt"):
            (art_dir / name).write_text(name)
        files = list_artifacts(tmp_path, "s1")
        names = [f.name for f in files]
        assert names == sorted(names)


# ---------------------------------------------------------------------------
# clean_artifacts
# ---------------------------------------------------------------------------

class TestCleanArtifacts:
    def test_returns_zero_when_no_directory(self, tmp_path):
        assert clean_artifacts(tmp_path, "ghost") == 0

    def test_removes_all_files_and_returns_count(self, tmp_path):
        art_dir = get_artifact_dir(tmp_path, "s1")
        for name in ("x.bnd", "x.cfg", "plot.png"):
            (art_dir / name).write_text("data")
        count = clean_artifacts(tmp_path, "s1")
        assert count == 3
        # clean_artifacts removes the session dir when empty; guard iterdir() accordingly
        assert not art_dir.exists() or not any(art_dir.iterdir())

    def test_removes_session_directory_when_empty(self, tmp_path):
        art_dir = get_artifact_dir(tmp_path, "s1")
        (art_dir / "f.txt").write_text("x")
        clean_artifacts(tmp_path, "s1")
        assert not art_dir.exists()

    def test_does_not_affect_other_sessions(self, tmp_path):
        for sid in ("s1", "s2"):
            d = get_artifact_dir(tmp_path, sid)
            (d / "f.txt").write_text("x")
        clean_artifacts(tmp_path, "s1")
        remaining = list_artifacts(tmp_path, "s2")
        assert len(remaining) == 1

