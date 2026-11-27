"""Checks before the test split is opened (protocol section 14), on a temporary git repository."""
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from dstc25 import guard  # noqa: E402
from dstc25.runinfo import sha256  # noqa: E402

FILES = ("data/a.bin", "data/sealed/test.bin")


def git(root, *args):
    subprocess.run(["git", *args], cwd=root, check=True, capture_output=True,
                   env={**os.environ, "GIT_AUTHOR_NAME": "t", "GIT_AUTHOR_EMAIL": "t@t",
                        "GIT_COMMITTER_NAME": "t", "GIT_COMMITTER_EMAIL": "t@t"})


@pytest.fixture
def repo(tmp_path):
    (tmp_path / "data" / "sealed").mkdir(parents=True)
    for f in FILES:
        (tmp_path / f).write_bytes(f.encode())
    manifest = {"sha256": {f: sha256(tmp_path / f) for f in FILES}}
    (tmp_path / "data" / "MANIFEST.json").write_text(json.dumps(manifest))
    (tmp_path / ".gitignore").write_text("data/*\n!data/MANIFEST.json\n")
    git(tmp_path, "init", "-q")
    git(tmp_path, "config", "core.autocrlf", "false")
    git(tmp_path, "add", ".")
    git(tmp_path, "commit", "-q", "-m", "freeze")
    git(tmp_path, "tag", "protocol-vX")
    (tmp_path / "code.py").write_text("x = 1\n")
    git(tmp_path, "add", "code.py")
    git(tmp_path, "commit", "-q", "-m", "after freeze")
    return tmp_path


def test_passes_on_clean_descendant(repo):
    assert len(guard.check(repo, "protocol-vX", FILES)) == 40


def test_refuses_modified_data(repo):
    (repo / "data" / "sealed" / "test.bin").write_bytes(b"changed")
    with pytest.raises(guard.GuardError, match="SHA-256"):
        guard.check(repo, "protocol-vX", FILES)


def test_refuses_missing_tag_or_non_descendant(repo):
    with pytest.raises(guard.GuardError, match="not found"):
        guard.check(repo, "protocol-vY", FILES)
    git(repo, "checkout", "-q", "--orphan", "other")
    git(repo, "commit", "-q", "-m", "unrelated")
    with pytest.raises(guard.GuardError, match="does not descend"):
        guard.check(repo, "protocol-vX", FILES)


def test_refuses_dirty_repository(repo):
    (repo / "code.py").write_text("x = 2\n")
    with pytest.raises(guard.GuardError, match="uncommitted"):
        guard.check(repo, "protocol-vX", FILES)
    git(repo, "checkout", "--", "code.py")
    (repo / "new.py").write_text("")
    with pytest.raises(guard.GuardError, match="untracked"):
        guard.check(repo, "protocol-vX", FILES)


def test_openings_are_logged(tmp_path):
    log = tmp_path / "results" / "test_openings.json"
    assert guard.log_opening(log, "a" * 40, "E-main") == 1
    assert guard.log_opening(log, "b" * 40, "E-main") == 2
    assert [e["commit"][0] for e in json.loads(log.read_text())] == ["a", "b"]
