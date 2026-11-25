"""Errors raised while building an nnU-Net dataset.

Every expected failure derives from `NnUNetBuildError`, so the CLI can
report it as a one-line message instead of a traceback. This module has no
dependencies so any other module, including worker code, can import it.
"""

from __future__ import annotations


class NnUNetBuildError(Exception):
    """Base class for expected nnU-Net build failures."""


class DatasetConfigError(NnUNetBuildError, ValueError):
    """The config file is missing, unreadable or invalid."""


class CaseSelectionError(NnUNetBuildError, ValueError):
    """The query results can't be turned into cases."""


class DatasetExistsError(NnUNetBuildError, FileExistsError):
    """The dataset folder (or its ID) is already in use."""


class DatasetOutputError(NnUNetBuildError, OSError):
    """The output folder couldn't be prepared, written or verified."""
