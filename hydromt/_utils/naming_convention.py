from glob import glob
from pathlib import Path
from re import compile as compile_regex
from re import error as regex_error
from re import escape
from string import Formatter
from typing import List, Optional, Pattern, Tuple

_PLACEHOLDERS = frozenset({"year", "month", "variable", "name", "overview_level"})
_SEGMENT_PATTERN = r"[^/\\]+"


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
    uri_expanded = ""

    if "{" in uri:
        for literal_text, key, fmt, _ in Formatter().parse(uri):
            safe_literal = escape(literal_text).replace(r"\*", ".*").replace(r"\?", ".")
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
        regex = compile_regex(pattern)
    except regex_error:
        # try it as raw path if regular string fails
        regex = compile_regex(pattern.encode("unicode_escape").decode())

    return (uri, keys, regex)


def expand_uri_paths(
    uri: str,
    *,
    placeholders: Optional[List[str]] = None,
) -> dict[str, str]:
    """
    Expand a URI template into concrete paths and unique stable dataset names.

    Parameters
    ----------
    uri : str
        URI template containing placeholders (e.g., `{variable}`, `{year}`) and/or wildcards (`*`).
    placeholders : list of str, optional
        List of placeholder names to capture and use for naming datasets.
        If not provided, no placeholders will be captured for naming and
        the filename (without extension) will be used as the dataset name.

    Returns
    -------
    dict[str, str]
        Mapping of dataset names to paths
    """
    path_glob, _, regex = _expand_uri_placeholders(uri, placeholders=placeholders)
    results = {}

    for path in glob(path_glob):
        match = regex.match(path)
        if match and match.groups():
            name = ".".join(match.groups())
        else:
            # fallback to filename without extension if regex doesn't match
            name = Path(path).stem

        if name in results:
            raise ValueError(
                f"Duplicate dataset name '{name}' generated from placeholder/wildcard uri '{uri}'. "
                f"The duplicate name was generated from '{path}' and '{results[name]}'. "
                "Consider using more specific placeholders to ensure unique names."
            )
        results[name] = path

    return results
