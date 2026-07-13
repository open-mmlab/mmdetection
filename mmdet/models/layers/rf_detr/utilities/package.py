# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
"""Package version and git-status helpers."""

import os
import subprocess
from importlib.metadata import PackageNotFoundError, version


def get_version(package_name: str = "rfdetr") -> str | None:
    """Get the current version of the specified package.

    Args:
        package_name: The name of the package to get the version for. Defaults to ``'rfdetr'``.

    Returns:
        The version string of the specified package, or ``None`` if the version cannot be determined.
    """
    try:
        return version(package_name)
    except PackageNotFoundError:
        return None


def get_sha() -> str:
    """Return a short status string for the current git repo, or 'unknown' if unavailable.

    Returns:
        String describing the current git HEAD, status, and branch.
    """
    cwd = os.path.dirname(os.path.abspath(__file__))

    def _run(command: list[str]) -> str:
        return subprocess.check_output(command, cwd=cwd).decode("ascii").strip()

    try:
        sha = _run(["git", "rev-parse", "HEAD"])
        diff_result = subprocess.run(
            ["git", "diff-index", "--quiet", "HEAD", "--"],
            cwd=cwd,
            check=False,
            capture_output=True,
            text=True,
        )
        if diff_result.returncode not in (0, 1):
            raise subprocess.CalledProcessError(
                returncode=diff_result.returncode,
                cmd=["git", "diff-index", "--quiet", "HEAD", "--"],
                output=diff_result.stdout,
                stderr=diff_result.stderr,
            )
        has_diff = diff_result.returncode == 1
        status = "has uncommitted changes" if has_diff else "clean"
        branch = _run(["git", "rev-parse", "--abbrev-ref", "HEAD"])
        return f"sha: {sha}, status: {status}, branch: {branch}"
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "unknown"
