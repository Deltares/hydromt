import warnings


class HydroMTDeprecationWarning(FutureWarning):
    """Deprecation warning for HydroMT users (shown by default).

    Based on FutureWarning rather than DeprecationWarning because the latter
    is hidden by default outside ``__main__``, so users loading a data catalog
    would never see it.
    """


def _normalize_version(version: str) -> str:
    return version if version.startswith("v") else f"v{version}"


def deprecation_warning(
    message: str,
    *,
    removed_in_release: str | None = None,
    alternative: str | None = None,
    stacklevel: int = 2,
) -> None:
    """Issue a HydroMTDeprecationWarning.

    Parameters
    ----------
    message : str
        What is deprecated, e.g. "importing x from y is deprecated".
    removed_in_release : str | None, optional
        Release in which the feature will be removed (with or without leading "v").
    alternative : str | None, optional
        What to use instead, e.g. "import x from z instead".
    stacklevel : int, optional
        Passed to `warnings.warn`. Note that inside pydantic validators the
        stack points into pydantic, so put identifying context in `message`
        instead of relying on the location.
    """
    parts = [message.strip().rstrip(".")]
    if alternative:
        parts.append(alternative.strip().rstrip("."))
    removal = (
        _normalize_version(removed_in_release)
        if removed_in_release
        else "a future release"
    )
    parts.append(f"It will be removed in {removal}")
    warnings.warn(
        ". ".join(parts) + ".", HydroMTDeprecationWarning, stacklevel=stacklevel
    )
