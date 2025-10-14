"""Deferred file close handling to ensure data integrity during file operations."""

import logging
import shutil
from pathlib import Path

logger = logging.getLogger(__name__)

_MAX_CLOSE_ATTEMPTS = 2


class DeferredFileClose:
    """Class to handle deferred file closing operations."""

    def __init__(self, *, original_path: Path, temp_path: Path):
        self._original_path = original_path
        self._temp_path = temp_path
        self._close_attempts = 0

    def close(self) -> None:
        """Attempt to move the temporary file to the original path up to a maximum number of attempts."""
        while self._close_attempts < _MAX_CLOSE_ATTEMPTS:
            try:
                logger.debug(
                    f"Moving temporary file '{self._temp_path}' to destination '{self._original_path}'."
                )
                shutil.move(self._temp_path, self._original_path)
                return
            except PermissionError as e:
                self._close_attempts += 1
                logger.error(
                    f"Could not write to destination file {self._original_path} because the following error was raised: {e}"
                )
            except FileNotFoundError:
                logger.warning(
                    f"Could not find temporary file {self._temp_path}. It was likely already deleted by another component that updates the same dataset."
                )
                return

        # already tried to close this too many times
        logger.error(
            f"Max write attempts to file {self._original_path} exceeded. Skipping... "
            f"Instead, data was written to a temporary file: {self._temp_path}."
        )
