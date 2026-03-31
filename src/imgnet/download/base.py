from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any


class BaseDownloader(ABC):
    """Base class for all downloaders."""

    @abstractmethod
    def download(
        self,
        output_path: Path,
        instance_ids: list[str] | None = None,
        **kwargs: Any,  # noqa: ANN401
    ) -> None:
        """Download the instances from the source."""
        ...

    @property
    @abstractmethod
    def size(self) -> float:
        """Return the size of the instances in GB."""
        ...

    @property
    @abstractmethod
    def members(self) -> list[str]:
        """Return the members of the source."""
        ...
