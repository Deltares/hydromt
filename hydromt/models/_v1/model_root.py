"""the new model root class"""

from os import PathLike
from pathlib import Path
from typing import Literal

from hydromt.typing import ModelModes


class ModelRoot:
    def __init__(self, path: PathLike, mode: ModelModes = "r"):
        self.set_path(path)
        self.set_mode(mode)

    def set_path(self, path: PathLike) -> None:
        self.path: Path = Path(path)

    def set_mode(self, mode: ModelModes) -> None:
        self.mode = mode

    def assert_writing_mode(self) -> bool:
        return self.mode[0] == "w"

    def assert_reading_mode(self) -> bool:
        return self.mode[0] == "r"
