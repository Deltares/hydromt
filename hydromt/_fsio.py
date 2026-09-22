"""Filesystem primitives shared by the readers in :mod:`hydromt.readers`.

The readers take an fsspec :class:`~fsspec.AbstractFileSystem` plus plain string
URIs. The filesystem knows how to reach the bytes, the reader knows which form
its underlying library needs: a path for local files, a file handle for
``h5netcdf``, raw bytes for OGR or a mapper for zarr. This module holds those
primitives so every reader resolves them the same way.

``None`` and a local filesystem are equivalent: URIs are passed straight through
to the underlying library so that engine-specific fast paths (the ``netcdf4``
engine, GDAL's own drivers) keep working.
"""

import logging
from contextlib import contextmanager
from functools import wraps
from importlib import import_module
from typing import IO, Any, Callable, Iterable, Iterator, Optional, Protocol, TypeVar

from fsspec import AbstractFileSystem
from fsspec.core import split_protocol
from fsspec.mapping import FSMap

logger = logging.getLogger(__name__)

__all__ = [
    "is_local",
    "open_handles",
    "read_bytes",
    "get_mapper",
    "attach_close",
    "assert_local",
    "assert_local_uri",
    "is_local_uri",
    "normalize_io_errors",
    "reraise_as_permission_error",
]

T = TypeVar("T")


class Closeable(Protocol):
    """An xarray object whose close callback can be replaced."""

    def set_close(self, close: Callable[[], None] | None) -> None: ...


C = TypeVar("C", bound=Closeable)


def _optional_error_types(module: str, *names: str) -> tuple[type[BaseException], ...]:
    """Import the named exception types; empty when the backend isn't installed."""
    try:
        imported = import_module(module)
    except ImportError:  # pragma: no cover - optional dependency
        return ()
    return tuple(getattr(imported, name) for name in names if hasattr(imported, name))


# Errors that carry an HTTP status; only 401/403 mean "not allowed".
# Extend this tuple as more backends are folded in (e.g. google.api_core
# exceptions if/when a gcsfs-specific path needs different handling).
PERMISSION_ERROR_TYPES: tuple[type[BaseException], ...] = _optional_error_types(
    "aiohttp", "ClientResponseError"
) + _optional_error_types("botocore.exceptions", "ClientError")

# Errors that mean "we never had usable credentials"; these are always a
# permission problem regardless of any status code.
CREDENTIAL_ERROR_TYPES: tuple[type[BaseException], ...] = _optional_error_types(
    "botocore.exceptions",
    "NoCredentialsError",
    "PartialCredentialsError",
    "TokenRetrievalError",
) + _optional_error_types("azure.core.exceptions", "ClientAuthenticationError")

_LOCAL_PROTOCOLS = ("file", "local")


def is_local(filesystem: AbstractFileSystem | None) -> bool:
    """Return True if ``filesystem`` refers to the local filesystem.

    ``None`` counts as local: readers treat "no filesystem given" and "the local
    filesystem" identically.
    """
    if filesystem is None:
        return True
    protocol = filesystem.protocol
    protocols = (protocol,) if isinstance(protocol, str) else tuple(protocol)
    return any(p in _LOCAL_PROTOCOLS for p in protocols)


def open_handles(
    filesystem: AbstractFileSystem | None,
    uris: Iterable[str],
    mode: str = "rb",
) -> list[IO[bytes]]:
    """Open binary file handles for ``uris``.

    The caller owns the returned handles and is responsible for closing them,
    either eagerly or by attaching them to the returned object with
    :func:`attach_close`.
    """
    if filesystem is None:
        raise ValueError("open_handles requires a filesystem")
    handles: list[IO[bytes]] = []
    try:
        for uri in uris:
            with reraise_as_permission_error(uri):
                handles.append(filesystem.open(uri, mode))
    except Exception:
        _close_all(handles)
        raise
    return handles


@contextmanager
def open_handle(
    filesystem: AbstractFileSystem | None, uri: str, mode: str = "rb"
) -> Iterator[IO[bytes]]:
    """Open a single binary file handle for ``uri`` and close it on exit."""
    (handle,) = open_handles(filesystem, [uri], mode)
    try:
        yield handle
    finally:
        handle.close()


def read_bytes(filesystem: AbstractFileSystem | None, uri: str) -> bytes:
    """Read the full contents of ``uri`` into memory."""
    if filesystem is None:
        raise ValueError("read_bytes requires a filesystem")
    with reraise_as_permission_error(uri):
        return filesystem.cat_file(uri)


def get_mapper(filesystem: AbstractFileSystem | None, uri: str) -> FSMap:
    """Return a mutable mapping interface to ``uri``, as zarr expects."""
    if filesystem is None:
        raise ValueError("get_mapper requires a filesystem")
    with reraise_as_permission_error(uri):
        return filesystem.get_mapper(root=uri)


def attach_close(obj: C, handles: Iterable[IO[bytes]]) -> C:
    """Close ``handles`` when ``obj`` is closed.

    Used for lazily loaded xarray objects: the returned object stays backed by
    open handles until the caller closes it.
    """
    handles = list(handles)
    if not handles:
        return obj

    org_close: Callable[[], None] | None = getattr(obj, "_close", None)

    def _close() -> None:
        try:
            if org_close is not None:
                org_close()
        finally:
            _close_all(handles)

    obj.set_close(_close)
    return obj


def _close_all(handles: Iterable[IO[bytes]]) -> None:
    for handle in handles:
        try:
            handle.close()
        except Exception:  # pragma: no cover - best effort cleanup
            logger.debug("Could not close file handle", exc_info=True)


def assert_local(
    filesystem: AbstractFileSystem | None, *, reader: str, uri: Any = None
) -> None:
    """Raise a clear error when a local-only reader is handed a remote filesystem.

    Readers without a ``filesystem`` parameter only understand local paths.
    Without this check a remote URI surfaces as a confusing "file not found".
    """
    if is_local(filesystem):
        return
    assert filesystem is not None  # is_local(None) is True
    protocol = filesystem.protocol
    protocol = protocol if isinstance(protocol, str) else protocol[0]
    uri_msg = f" (got '{uri}')" if uri is not None else ""
    raise ValueError(
        f"{reader} does not support remote paths{uri_msg}: got a '{protocol}' "
        "filesystem, but only the local filesystem is supported."
    )


def is_local_uri(uri: Any) -> bool:
    """Return True if ``uri`` has no scheme, or a local one."""
    scheme, _ = split_protocol(str(uri))
    return scheme is None or scheme in _LOCAL_PROTOCOLS


def assert_local_uri(uri: Any, *, reader: str) -> None:
    """Raise a clear error when a local-only reader is handed a remote URI.

    Readers without a ``filesystem`` parameter only understand local paths.
    Without this check a remote URI surfaces as a confusing "file not found".
    """
    if is_local_uri(uri):
        return
    raise ValueError(
        f"{reader} does not support remote paths, got '{uri}'. "
        "Download the file first or use a reader that accepts a filesystem."
    )


def _extract_status_code(exc: BaseException) -> Optional[int]:
    """Normalize 'what HTTP status was this' across aiohttp- and botocore-shaped errors."""
    status = getattr(exc, "status", None)  # aiohttp.ClientResponseError
    if status is not None:
        return status
    response = getattr(exc, "response", None)  # botocore.exceptions.ClientError
    if response is not None:
        return response.get("ResponseMetadata", {}).get("HTTPStatusCode")
    return None


@contextmanager
def reraise_as_permission_error(context: Any):
    """Reraise various errors raised by file or network access as PermissionError."""
    try:
        yield
    except CREDENTIAL_ERROR_TYPES as e:
        raise PermissionError(
            f"Unauthorized access to {context}. Check your credentials."
        ) from e
    except PERMISSION_ERROR_TYPES as e:
        status_code = _extract_status_code(e)
        if status_code is not None and status_code in (401, 403):
            raise PermissionError(
                f"Unauthorized access to {context}. Check your credentials."
            ) from e
        raise


def normalize_io_errors(func: Callable[..., T]) -> Callable[..., T]:
    """Normalize credential and 401/403 errors raised by a reader.

    The context of the error message is taken from the reader's first argument,
    which by convention is the URI (or URIs) being read.
    """

    @wraps(func)
    def wrapper(*args, **kwargs) -> T:
        with reraise_as_permission_error(_error_context(func, args, kwargs)):
            return func(*args, **kwargs)

    return wrapper


_URI_ARG_NAMES = ("uri", "uris", "path", "paths", "loc_path", "tindex_path", "filepath")


def _error_context(
    func: Callable[..., Any], args: tuple[Any, ...], kwargs: dict[str, Any]
) -> str:
    if args:
        target = args[0]
    else:
        target = next(
            (kwargs[name] for name in _URI_ARG_NAMES if name in kwargs),
            None,
        )
    if target is None:
        return func.__name__
    if isinstance(target, (list, tuple, set)):
        return f"one of {list(target)}"
    if isinstance(target, dict):
        return f"one of {list(target.values())}"
    return str(target)
