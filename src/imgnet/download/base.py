from pathlib import Path
from typing import Any


class BaseDownloader:
    """Base class for all downloaders."""

    def __init__(self) -> None:
        pass

    def download(
        self,
        output_path: Path,
        instance_ids: list[str] | None = None,
        **kwargs: Any,  # noqa: ANN401
    ) -> None:
        """Download the instances from the source."""
        raise NotImplementedError

    @property
    def size(self) -> float:
        """Return the size of the instances in GB."""
        raise NotImplementedError

    @property
    def members(self) -> list[str]:
        """Return the members of the source."""
        raise NotImplementedError
