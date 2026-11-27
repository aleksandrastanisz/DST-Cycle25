"""Checks before the test split is opened (protocol section 14).

The final evaluation refuses to run unless
  - the SHA-256 of every data file it reads matches data/MANIFEST.json,
  - the current commit descends from the tag of the latest protocol version,
  - the repository is clean (no changes to tracked files and no untracked files outside
    .gitignore).
Every opening of the test split is appended to an openings log with time and commit, so a
repeated run (for example after a crash) is visible in the record.
"""
from __future__ import annotations

import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path

from dstc25.runinfo import sha256

LATEST_PROTOCOL_TAG = "protocol-v1.2"
DATA_FILES = ("data/processed/train.parquet", "data/processed/calib.parquet", "data/sealed/test.parquet")


class GuardError(RuntimeError):
    pass


def _git(root: Path, *args: str) -> subprocess.CompletedProcess:
    return subprocess.run(["git", *args], cwd=root, capture_output=True, text=True)


def check(root: Path, tag: str = LATEST_PROTOCOL_TAG, files=DATA_FILES) -> str:
    """Raise GuardError if any condition fails; return the current commit."""
    manifest = json.loads((root / "data" / "MANIFEST.json").read_text())["sha256"]
    for f in files:
        if f not in manifest or sha256(root / f) != manifest[f]:
            raise GuardError(f"{f} does not match its SHA-256 in data/MANIFEST.json")
    if _git(root, "rev-parse", "--verify", f"refs/tags/{tag}").returncode != 0:
        raise GuardError(f"tag {tag} not found")
    if _git(root, "merge-base", "--is-ancestor", tag, "HEAD").returncode != 0:
        raise GuardError(f"the current commit does not descend from {tag}")
    status = _git(root, "status", "--porcelain")
    if status.returncode != 0 or status.stdout.strip():
        raise GuardError("the repository has uncommitted changes or untracked files")
    return _git(root, "rev-parse", "HEAD").stdout.strip()


def log_opening(log_path: Path, commit: str, purpose: str) -> int:
    """Append an opening of the test split to the log; return how many openings it now holds."""
    log = json.loads(log_path.read_text()) if log_path.exists() else []
    log.append({"opened_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
                "commit": commit, "purpose": purpose})
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.write_text(json.dumps(log, indent=1))
    return len(log)
