from re import compile
from string import Formatter
from typing import (
    Pattern,
    Tuple,
)


# TODO integrate with convention resolver?
def capture_glob(s: str) -> Tuple[str, Pattern]:
    glob = ""
    regex = ""

    for lead, _field_name, _, _ in Formatter().parse(s):
        glob += lead
        regex += lead
        if _field_name is not None:
            glob += "*"
            regex += "(.*)"

    return glob, compile(regex)
