# -*- coding: utf-8 -*-
"""Physical-copy sandbox for agentic-judge harness execution.

Materializes candidate evidence (a workspace directory and/or an execution
transcript) into a brand-new temporary directory that a harness subprocess
uses as its working directory.

This is process-level isolation only (fresh temp dir, physical file copies,
symlink skipping) -- it is not an OS-level security sandbox. Do not rely on
it alone to run adversarial/untrusted code; it exists to keep the judged
candidate's files from cross-contaminating between k parallel samples and
to stop symlinks from smuggling content in or out of the sandbox, not to
contain a malicious subprocess.
"""
import json
import os
import shutil
import tempfile
from pathlib import Path
from typing import Any, Optional

__all__ = ["ProcessSandbox"]


def _copytree_no_symlinks(src: Path, dest: Path) -> int:
    """Recursively copy `src` into `dest`, skipping symlinks entirely.

    Symlinks are skipped rather than dereferenced (which could pull file
    contents from outside the candidate directory into the sandbox) or
    preserved (which would leave a path a sandboxed process could still
    follow back out of the sandbox).

    Args:
        src: Source directory to copy from.
        dest: Destination directory to copy into (created if missing).

    Returns:
        The number of symlinks skipped, counted recursively.
    """
    dest.mkdir(parents=True, exist_ok=True)
    skipped = 0
    for entry in src.iterdir():
        target = dest / entry.name
        if entry.is_symlink():
            skipped += 1
            continue
        if entry.is_dir():
            skipped += _copytree_no_symlinks(entry, target)
        elif entry.is_file():
            shutil.copy2(entry, target)
    return skipped


def _materialize_transcript(dest_path: Path, transcript: Any) -> None:
    """Write transcript evidence to `dest_path` as JSONL.

    Args:
        dest_path: Destination file path (e.g. `sandbox_dir / "transcript.jsonl"`).
        transcript: Either a path (str/PathLike) to an existing JSONL file to copy,
            or a list of JSON-serializable message dicts to serialize one-per-line.

    Raises:
        FileNotFoundError: If `transcript` is a path-like value that does not exist.
        TypeError: If `transcript` is neither a path-like value nor a list.
    """
    if isinstance(transcript, (str, os.PathLike)):
        src = Path(transcript)
        if not src.exists():
            raise FileNotFoundError(f"transcript path not found: {src}")
        shutil.copy(str(src), dest_path)
        return
    if isinstance(transcript, list):
        with open(dest_path, "w", encoding="utf-8") as f:
            for msg in transcript:
                f.write(json.dumps(msg, ensure_ascii=False) + "\n")
        return
    raise TypeError("transcript must be a file path (str/PathLike) or a list of message dicts")


class ProcessSandbox:
    """Context manager that builds and tears down an isolated sandbox directory.

    Attributes:
        workspace_path: Path to the candidate's produced artifacts directory, if any.
            Must exist and be a directory -- `__enter__` raises `FileNotFoundError`/
            `NotADirectoryError` otherwise, the same fail-loud contract as `transcript`.
            A caller-side misconfiguration (typo'd/not-yet-materialized path) must never
            be allowed to silently look identical to "the candidate produced no artifacts".
        transcript: Candidate execution transcript (path or message list), if any.
        keep_on_exit: If True, do not delete the sandbox directory on `__exit__`
            (for debugging failed harness runs).
        sandbox_dir: The created sandbox `Path`, set once `__enter__` has run.
        symlinks_skipped: Count of symlinks skipped while copying `workspace_path`.

    Example:
        >>> with ProcessSandbox(workspace_path="/tmp/candidate") as sandbox_dir:
        ...     # sandbox_dir / "workspace" holds a physical copy of /tmp/candidate
        ...     subprocess.run([...], cwd=str(sandbox_dir))
    """

    def __init__(
        self,
        workspace_path: Optional[str] = None,
        transcript: Optional[Any] = None,
        keep_on_exit: bool = False,
    ):
        self.workspace_path = workspace_path
        self.transcript = transcript
        self.keep_on_exit = keep_on_exit
        self.sandbox_dir: Optional[Path] = None
        self.symlinks_skipped: int = 0

    def __enter__(self) -> Path:
        self.sandbox_dir = Path(tempfile.mkdtemp(prefix="openjudge_harness_"))
        try:
            if self.workspace_path is not None:
                dest = self.sandbox_dir / "workspace"
                src = Path(self.workspace_path)
                if not src.exists():
                    raise FileNotFoundError(f"workspace_path not found: {src}")
                if not src.is_dir():
                    raise NotADirectoryError(f"workspace_path is not a directory: {src}")
                self.symlinks_skipped = _copytree_no_symlinks(src, dest)
            if self.transcript is not None:
                _materialize_transcript(self.sandbox_dir / "transcript.jsonl", self.transcript)
        except Exception:
            shutil.rmtree(self.sandbox_dir, ignore_errors=True)
            raise
        return self.sandbox_dir

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        if not self.keep_on_exit and self.sandbox_dir is not None:
            shutil.rmtree(self.sandbox_dir, ignore_errors=True)
