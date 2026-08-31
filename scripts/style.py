"""Apply or verify PyGrad's mechanical Python style refactors."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
TARGETS = ("pygrad", "tests", "usage_examples")


def run_ruff(*arguments: str) -> int:
    """Run Ruff from the same Python environment as this script."""
    command = (sys.executable, "-m", "ruff", *arguments, *TARGETS)
    return subprocess.run(command, cwd=PROJECT_ROOT, check=False).returncode


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Apply safe Ruff fixes and formatting across the project."
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="report required style refactors without changing files",
    )
    args = parser.parse_args()

    if args.check:
        lint_status = run_ruff("check", "--diff")
        format_status = run_ruff("format", "--check")
    else:
        lint_status = run_ruff("check", "--fix-only")
        format_status = run_ruff("format")

    return 1 if lint_status or format_status else 0


if __name__ == "__main__":
    raise SystemExit(main())
