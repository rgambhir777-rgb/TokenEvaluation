"""Resolve a GitHub URL, ``owner/repo`` shorthand or local path to a directory."""

from __future__ import annotations

import os
import re
import shutil
import subprocess
from pathlib import Path

DEFAULT_CACHE = Path(os.environ.get("TOKEN_EVAL_REPOS", ".token_eval/repos"))
_SHORTHAND = re.compile(r"^[\w.-]+/[\w.-]+$")


class RepoError(RuntimeError):
    pass


def normalize_url(src: str) -> str:
    """'owner/repo' -> https://github.com/owner/repo.git; URLs are returned unchanged."""
    if _SHORTHAND.match(src) and not Path(src).exists():
        return f"https://github.com/{src}.git"
    return src


def repo_name(src: str) -> str:
    """A filesystem/tag friendly name, e.g. 'owner__repo'."""
    s = src.rstrip("/")
    if Path(s).exists():
        return Path(s).resolve().name
    s = re.sub(r"\.git$", "", s)
    s = re.sub(r"^(https?://|git@|ssh://|file://)", "", s)
    parts = [p for p in re.split(r"[/:]", s) if p]
    return "__".join(parts[-2:]) if len(parts) >= 2 else (parts[-1] if parts else "repo")


def resolve_repo(src: str, cache_dir: Path | None = None, ref: str | None = None,
                 fresh: bool = False) -> Path:
    """Return a local directory for ``src``, shallow-cloning remote repos into ``cache_dir``."""
    local = Path(src)
    if local.is_dir():
        return local.resolve()
    url = normalize_url(src)
    dest = (cache_dir or DEFAULT_CACHE) / repo_name(src)
    if dest.exists() and fresh:
        shutil.rmtree(dest)
    if dest.exists():
        return dest.resolve()
    if shutil.which("git") is None:
        raise RepoError("git is not installed or not on PATH")
    dest.parent.mkdir(parents=True, exist_ok=True)
    # core.longpaths avoids "Filename too long" checkouts on Windows (no-op elsewhere).
    cmd = ["git", "-c", "core.longpaths=true", "clone", "--depth", "1"]
    if ref:
        cmd += ["--branch", ref]
    cmd += [url, str(dest)]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        shutil.rmtree(dest, ignore_errors=True)  # don't cache a half-checked-out repo
        raise RepoError(f"git clone failed: {proc.stderr.strip()}")
    return dest.resolve()
