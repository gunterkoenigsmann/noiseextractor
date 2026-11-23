"""The ``noiseextractor-cli`` command of the package: the same command line
tool as the native ``noiseextractor-cli`` program, running inside this
Python process."""

from __future__ import annotations

import sys

from . import _native


def main() -> int:
    """Runs ``noiseextractor-cli`` with this process's arguments and exits
    with its status."""
    # Python may have buffered output of its own; the native code writes to
    # the file descriptors directly.
    sys.stdout.flush()
    sys.stderr.flush()
    # argv[0] as the command users type, whatever the launcher script is
    # called, so that usage messages name it.
    sys.exit(_native.run_cli(["noiseextractor-cli", *sys.argv[1:]]))
