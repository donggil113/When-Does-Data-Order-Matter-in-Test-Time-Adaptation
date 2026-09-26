"""Run manifest: code/config/data/model hashes, environment, wall time and peak memory."""

from __future__ import annotations

import datetime as _dt
import hashlib
import json
import os
import platform
import resource
import subprocess
import sys
from typing import Dict, Optional


def _git(args, cwd) -> Optional[str]:
    try:
        return subprocess.run(["git", *args], cwd=cwd, capture_output=True, text=True, check=True).stdout.strip()
    except Exception:  # noqa: BLE001 - git absent or no commits yet
        return None


def git_info(repo_root: str) -> Dict[str, object]:
    """Commit and dirty state of code/config; run outputs under runs/ are excluded from 'dirty'."""
    head = _git(["rev-parse", "HEAD"], repo_root)
    porcelain = _git(["status", "--porcelain", "--", ".", ":(exclude)runs"], repo_root)
    diff = _git(["diff", "HEAD", "--", ".", ":(exclude)runs"], repo_root) if head else None
    return {
        "commit": head,
        "branch": _git(["rev-parse", "--abbrev-ref", "HEAD"], repo_root),
        "dirty": bool(porcelain) if porcelain is not None else None,
        "dirty_files": porcelain.splitlines() if porcelain else [],
        "diff_sha256": hashlib.sha256(diff.encode()).hexdigest() if diff else None,
    }


def file_sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 16), b""):
            h.update(chunk)
    return h.hexdigest()


def canonical_sha256(obj) -> str:
    return hashlib.sha256(json.dumps(obj, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def environment() -> Dict[str, object]:
    return {
        "python": sys.version,
        "implementation": platform.python_implementation(),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "cpu_count": os.cpu_count(),
        "third_party_packages_used": [],
        "gpu": None,
    }


def peak_rss_mib() -> float:
    """Peak resident set size of this process (Linux reports KiB, macOS bytes)."""
    r = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return r / 1024.0 if sys.platform != "darwin" else r / (1024.0 * 1024.0)


def utc_now() -> str:
    return _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds")


__all__ = ["git_info", "file_sha256", "canonical_sha256", "environment", "peak_rss_mib", "utc_now"]
