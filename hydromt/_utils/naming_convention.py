import re
from string import Formatter
from typing import Pattern

_PLACEHOLDERS = frozenset({"year", "month", "variable", "name", "overview_level"})
_SEGMENT_PATTERN = r"[^/\\]+"


def _expand_uri_placeholders(
    uri: str,
    *,
    placeholders: list[str] | None = None,
    time_range: tuple[str, str] | None = None,
    variables: list[str] | None = None,
) -> tuple[str, list[str], Pattern[str]]:
    """Expand known placeholders in the URI.

    This function takes a URI with placeholders and expands it into a regex pattern that can be used to match actual URIs.
    It also returns a list of keys corresponding to the placeholders that were captured in the regex.

    """
    if placeholders is None:
        placeholders = []
    keys: list[str] = []
    pattern: str = ""
    uri_expanded = ""

    if "{" in uri:
        for literal_text, key, fmt, _ in Formatter().parse(uri):
            safe_literal = (
                re.escape(literal_text).replace(r"\*", ".*").replace(r"\?", ".")
            )
            pattern += safe_literal
            uri_expanded += literal_text
            if key is None:
                continue

            if key in placeholders:
                pattern += (
                    f"({_SEGMENT_PATTERN})"  # capture only requested placeholders
                )
                keys.append(key)
            else:
                pattern += f"(?:{_SEGMENT_PATTERN})"  # match segment, but don't capture

            key_str = "{" + f"{key}:{fmt}" + "}" if fmt else "{" + key + "}"
            # Determine if this key should become a wildcard
            if key in ["year", "month"] and time_range is None:
                uri_expanded += "*"
            elif key == "variable" and variables is None:
                uri_expanded += "*"
            elif key == "name":
                uri_expanded += "*"
            else:
                uri_expanded += key_str
        uri = uri_expanded

    # Anchor the regex to make sure the entire path matches, not just a substring
    pattern = f"^{pattern}$"

    # windows paths creating invalid escape sequences
    try:
        regex = re.compile(pattern)
    except re.error:
        # try it as raw path if regular string fails
        regex = re.compile(pattern.encode("unicode_escape").decode())

    return (uri, keys, regex)
