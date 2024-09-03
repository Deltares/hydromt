from re import compile as compile_regex
from re import error as regex_error
from string import Formatter
from typing import List, Optional, Pattern, Tuple

_placeholders = frozenset({"year", "month", "variable", "name", "overview_level"})


def _expand_uri_placeholders(
    uri: str,
    *,
    placeholders: Optional[List[str]] = None,
    time_range: Optional[Tuple[str, str]] = None,
    variables: Optional[List[str]] = None,
) -> Tuple[str, List[str], Pattern[str]]:
    """Expand known placeholders in the URI."""
    if placeholders is None:
        placeholders = []
    keys: list[str] = []
    pattern: str = ""

    if "{" in uri:
        uri_expanded = ""
        for literal_text, key, fmt, _ in Formatter().parse(uri):
            uri_expanded += literal_text
            pattern += literal_text
            if key is None:
                continue
            pattern += "(.*)"
            key_str = "{" + f"{key}:{fmt}" + "}" if fmt else "{" + key + "}"
            # remove unused fields
            if key in ["year", "month"] and time_range is None:
                uri_expanded += "*"
            elif key == "variable" and variables is None:
                uri_expanded += "*"
            elif key == "name":
                uri_expanded += "*"
            # escape unknown fields
            elif key is not None and key not in placeholders:
                uri_expanded = uri_expanded + "{" + key_str + "}"
            else:
                uri_expanded = uri_expanded + key_str
                keys.append(key)
        uri = uri_expanded

    # windows paths creating invalid escape sequences
    try:
        regex = compile_regex(pattern)
    except regex_error:
        # try it as raw path if regular string fails
        regex = compile_regex(pattern.encode("unicode_escape").decode())

    return (uri, keys, regex)
