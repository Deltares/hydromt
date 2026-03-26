"""URIResolver for Azure Blob Storage and Azure Data Lake Storage Gen2 URIs."""

import json
import logging
import os
import re
import urllib.request
from itertools import product
from typing import Any, Iterable

import geopandas as gpd
import pandas as pd
from fsspec import AbstractFileSystem
from fsspec.core import split_protocol

from hydromt._utils.naming_convention import _expand_uri_placeholders
from hydromt.data_catalog.uri_resolvers.uri_resolver import URIResolver
from hydromt.error import NoDataStrategy, exec_nodata_strat
from hydromt.typing import SourceMetadata, TimeRange, Zoom
from hydromt.typing.fsspec_types import FSSpecFileSystem

logger = logging.getLogger(__name__)

# Matches the ADLS Gen2 / fsspec abfs scheme: abfs://container/path
_ABFS_RE = re.compile(r"^abfs://", re.IGNORECASE)

# Matches Azure Blob Storage HTTPS URLs:
# https://<account>.blob.core.windows.net/<container>/<path>
_HTTPS_BLOB_RE = re.compile(
    r"^https://(?P<account>[^.]+)\.blob\.core\.windows\.net/(?P<rest>.+)$",
    re.IGNORECASE,
)

# Matches AzureML datastore URIs:
# azureml://subscriptions/<sub>/resourcegroups/<rg>/workspaces/<ws>/datastores/<ds>/paths/<path>
_AZUREML_RE = re.compile(
    r"^azureml://subscriptions/(?P<subscription>[^/]+)"
    r"/resourcegroups/(?P<resource_group>[^/]+)"
    r"/workspaces/(?P<workspace>[^/]+)"
    r"/datastores/(?P<datastore>[^/]+)"
    r"/paths/(?P<path>.+)$",
    re.IGNORECASE,
)

# Placeholders supported by this resolver (superset of ConventionResolver).
_AZURE_PLACEHOLDERS = frozenset({"year", "month", "day", "variable"})


class _SafeDict(dict):
    """A dict that returns the placeholder string for missing keys instead of raising KeyError."""

    def __missing__(self, key):
        return "{" + key + "}"


class AzureBlobResolver(URIResolver):
    """URIResolver for Azure Blob Storage and ADLS Gen2 URIs.

    Handles three URI styles:

    * ``abfs://container/path/to/data.zarr`` (ADLS Gen2 / fsspec)
    * ``https://<account>.blob.core.windows.net/<container>/path`` (HTTPS Blob)
    * ``azureml://subscriptions/<sub>/resourcegroups/<rg>/workspaces/<ws>/datastores/<ds>/paths/<path>`` (AzureML datastore)

    All styles are normalised to ``abfs://`` internally and passed on to
    ``adlfs.AzureBlobFileSystem`` (fsspec-compatible).  For ``azureml://``
    URIs the ``azure-ai-ml`` package is required.

    Parameters
    ----------
    options : dict, optional
        Key/value options forwarded to ``adlfs.AzureBlobFileSystem``
        (e.g. ``account_name``, ``account_key``, ``sas_token``,
        ``connection_string``, ``tenant_id``, ``client_id``, ``client_secret``).
        The special key ``sas_token_url`` points to a URL that returns JSON
        with a ``"token"`` key (e.g. the Planetary Computer SAS API).  When
        set and no explicit ``sas_token`` is provided, a fresh token is
        fetched automatically before each ``resolve()`` call.
    filesystem : FSSpecFileSystem, optional
        Pre-constructed fsspec filesystem.  When supplied it is used as-is
        and ``options`` are ignored for filesystem construction.

    Notes
    -----
    **Authentication**

    Credentials are resolved in the following order of precedence:

    1. Explicit values in the catalog YAML (``account_name``, ``account_key``,
       ``sas_token``, ``connection_string``).
    2. Environment variables: ``AZURE_STORAGE_ACCOUNT_NAME``,
       ``AZURE_STORAGE_ACCOUNT_KEY``, ``AZURE_STORAGE_SAS_TOKEN``,
       ``AZURE_STORAGE_CONNECTION_STRING``.
    3. ``azure.identity.DefaultAzureCredential`` (Managed Identity, Azure CLI,
       VS Code, environment-variable service principals, etc.).

    **Time-templated URIs**

    Placeholders ``{year}``, ``{month}``, ``{day}``, ``{variable}`` are expanded
    using the same ``_expand_uri_placeholders`` utility as ``ConventionResolver``.

    **ABFS to HTTPS conversion**

    rasterio and GDAL do not natively understand the ``abfs://`` scheme.  When
    a SAS token is available the resolver converts resolved ``abfs://`` URIs to
    HTTPS blob URLs
    (``https://<account>.blob.core.windows.net/<container>/<path>?<sas_token>``)
    so that rasterio / GDAL can open the data via their built-in HTTPS /
    vsicurl handler.  For drivers that go through fsspec (e.g. xarray with
    zarr), ``abfs://`` URIs work as-is via ``adlfs``.

    Examples
    --------
    Minimal catalog entry (anonymous public container)::

        uk_coastal_dem:
          data_type: RasterDataset
          driver: raster
          uri_resolver:
            name: azure_blob
          uri: "abfs://public-data/dem/uk_2m.tif"

    With explicit SAS token and time-templated path::

        rainfall_ensemble:
          data_type: RasterDataset
          driver: raster_xarray
          uri_resolver:
            name: azure_blob
            options:
              account_name: mystorageaccount
              sas_token: "sv=2022-11-02&ss=b&..."
          uri: "abfs://hydrodata/rainfall/{year}/{month}/precip.nc"

    With automatic SAS token fetching (Planetary Computer)::

        cop_dem:
          data_type: RasterDataset
          driver: rasterio
          uri_resolver:
            name: azure_blob
            options:
              account_name: elevationeuwest
              sas_token_url: "https://planetarycomputer.microsoft.com/api/sas/v1/token/cop-dem-glo-30"
          uri: "abfs://copernicus-dem/COP30_hh/{variable}.tif"
    """

    name = "azure_blob"

    @staticmethod
    def _get_dates(
        keys: list[str],
        time_range: TimeRange,
    ) -> pd.PeriodIndex:
        """Return a PeriodIndex covering *time_range* at the required granularity.

        Parameters
        ----------
        keys : list[str]
            Placeholder names found in the URI (e.g. ``["year", "month"]``).
        time_range : TimeRange
            Start and end of the requested time window.

        Returns
        -------
        pd.PeriodIndex
            Period index with daily, monthly, or yearly frequency depending
            on whether ``"day"``, ``"month"``, or only ``"year"`` is in *keys*.
        """
        t_range = pd.to_datetime([time_range.start, time_range.end])
        if "day" in keys:
            freq = "D"
        elif "month" in keys:
            freq = "M"
        else:
            freq = "Y"
        return pd.period_range(*t_range, freq=freq)

    def resolve(
        self,
        uri: str,
        *,
        time_range: TimeRange | None = None,
        zoom: Zoom | None = None,
        mask: gpd.GeoDataFrame | None = None,
        variables: list[str] | None = None,
        metadata: SourceMetadata | None = None,
        handle_nodata: NoDataStrategy = NoDataStrategy.RAISE,
    ) -> list[str]:
        """Resolve an Azure Blob / ADLS Gen2 URI into a list of concrete paths.

        Parameters
        ----------
        uri : str
            URI to resolve.  Accepted forms:

            * ``abfs://container/path/to/file.tif``
            * ``abfs://container/path/{year}/{month}/file.nc``  (time template)
            * ``https://<account>.blob.core.windows.net/<container>/path``
            * ``azureml://subscriptions/…/datastores/…/paths/…``
        time_range : TimeRange | None, optional
            Left-inclusive start/end time of the data, by default None.
            Required when *uri* contains ``{year}`` or ``{month}`` placeholders.
        zoom : Zoom | None, optional
            Ignored — included for interface compatibility, by default None.
        mask : gpd.GeoDataFrame | None, optional
            Ignored — included for interface compatibility, by default None.
        variables : list[str] | None, optional
            Variable names used to expand ``{variable}`` placeholders,
            by default None.
        metadata : SourceMetadata | None, optional
            DataSource metadata, by default None.
        handle_nodata : NoDataStrategy, optional
            How to react when no data is found,
            by default ``NoDataStrategy.RAISE``.

        Returns
        -------
        list[str]
            Concrete URIs that downstream drivers can open.  When a SAS
            token is available, ``abfs://`` paths are converted to HTTPS blob
            URLs so that rasterio / GDAL can read them directly (see Notes
            in the class docstring).

        Raises
        ------
        ValueError
            If the URI scheme is not recognised as an Azure path.
        NoDataException
            When no data is found and ``handle_nodata`` is
            ``NoDataStrategy.RAISE``.
        """
        logger.debug(f"AzureBlobResolver: resolving uri '{uri}'")

        # Normalise URI
        normalised_uri, account_from_uri = _normalise_uri(uri)
        effective_options: dict[str, Any] = dict(self.options or {})
        if account_from_uri and "account_name" not in effective_options:
            effective_options["account_name"] = account_from_uri

        # Auto-fetch SAS token when sas_token_url is provided.
        sas_token_url = effective_options.pop("sas_token_url", None)
        if sas_token_url and "sas_token" not in effective_options:
            effective_options["sas_token"] = _fetch_sas_token(sas_token_url)

        # Expand placeholders
        uri_expanded, keys, _ = _expand_uri_placeholders(
            normalised_uri,
            placeholders=list(_AZURE_PLACEHOLDERS),
            time_range=(time_range.start, time_range.end) if time_range else None,
            variables=variables,
        )

        if time_range and any(k in keys for k in ("year", "month", "day")):
            dates = self._get_dates(keys, time_range)
        else:
            dates = pd.PeriodIndex(["1970-01-01"], freq="D")

        if not variables:
            variables = [""]

        fmts: Iterable[dict[str, Any]] = (
            {
                "year": dt.year,
                "month": dt.month,
                "day": dt.day if hasattr(dt, "day") else 1,
                "variable": var,
            }
            for dt, var in product(dates, variables)
        )
        uris: list[str] = list(
            dict.fromkeys(  # deduplicate while preserving order
                uri_expanded.format_map(_SafeDict(**fmt)) for fmt in fmts
            )
        )

        # Resolve wildcards and verify existence on the filesystem.
        self._ensure_azure_filesystem(effective_options)
        fs = self.filesystem.get_fs()

        uris = list(self._resolve_wildcards(uris, fs))

        if not uris:
            exec_nodata_strat(
                f"AzureBlobResolver: no data found for uri '{uri}'.",
                strategy=handle_nodata,
            )
            return []

        # Convert to HTTPS blob URLs when credentials are available so
        # that rasterio/GDAL (which do not understand abfs://) can open
        # the files directly.
        account_name = effective_options.get("account_name") or account_from_uri
        sas_token = effective_options.get("sas_token")
        if account_name and sas_token:
            uris = [_abfs_to_https(u, account_name, sas_token) for u in uris]

        logger.debug(f"AzureBlobResolver: resolved {len(uris)} URI(s)")
        return uris

    def _ensure_azure_filesystem(self, effective_options: dict[str, Any]) -> None:
        """Build an ``abfs`` filesystem from *effective_options* if needed.

        Parameters
        ----------
        effective_options : dict[str, Any]
            Merged resolver options including account name, credentials, and
            any user-supplied kwargs.  Passed to
            ``_resolve_azure_credentials`` and then to
            ``FSSpecFileSystem(protocol='abfs', ...)``.

        Raises
        ------
        PermissionError
            If authentication with Azure Blob Storage fails.
        """
        if self.filesystem.protocol != "file":
            # Already configured (injected by DataSource or a previous call).
            return

        storage_options = _resolve_azure_credentials(effective_options)
        try:
            self.filesystem = FSSpecFileSystem(
                protocol="abfs", storage_options=storage_options
            )
        except ImportError:
            raise
        except Exception as exc:
            raise PermissionError(
                "AzureBlobResolver: failed to authenticate with Azure Blob Storage. "
                "Ensure credentials are supplied via the catalog YAML "
                "(account_name, account_key, sas_token, connection_string) "
                "or environment variables (AZURE_STORAGE_ACCOUNT_NAME, "
                "AZURE_STORAGE_ACCOUNT_KEY, AZURE_STORAGE_SAS_TOKEN, "
                "AZURE_STORAGE_CONNECTION_STRING). "
                "For anonymous public containers set 'anon: true'."
            ) from exc

    @staticmethod
    def _resolve_wildcards(uris: Iterable[str], fs: AbstractFileSystem) -> set[str]:
        """Expand wildcards and return the set of existing paths.

        Parameters
        ----------
        uris : Iterable[str]
            URIs to expand; may contain glob patterns (``*``, ``?``).
        fs : AbstractFileSystem
            fsspec filesystem used for globbing.

        Returns
        -------
        set[str]
            Deduplicated set of concrete URIs that exist on *fs*.
        """

        def _glob_one(uri: str) -> list[str]:
            protocol, _ = split_protocol(uri)
            if protocol in ("https", "http"):
                return [uri]
            matches = fs.glob(uri)
            if protocol is not None:
                matches = [
                    fs.unstrip_protocol(m) if not m.startswith(protocol) else m
                    for m in matches
                ]
            return matches

        result: set[str] = set()
        for uri in uris:
            result.update(_glob_one(uri))
        return result


def _fetch_sas_token(url: str) -> str:
    """Fetch a SAS token from a token endpoint.

    Parameters
    ----------
    url : str
        URL of the token endpoint (e.g. the Planetary Computer SAS API).
        Must return JSON with a ``"token"`` key.

    Returns
    -------
    str
        The SAS token string.

    Raises
    ------
    PermissionError
        If the request fails or the response cannot be parsed.
    """
    try:
        with urllib.request.urlopen(url, timeout=10) as resp:  # noqa: S310
            data = json.loads(resp.read())
        return data["token"]
    except Exception as exc:
        raise PermissionError(
            f"AzureBlobResolver: failed to fetch SAS token from '{url}'. "
            "Check that the URL is correct and reachable."
        ) from exc


def _abfs_to_https(uri: str, account_name: str, sas_token: str) -> str:
    """Convert an ``abfs://`` URI to an HTTPS blob URL with SAS token.

    rasterio and GDAL do not natively understand the ``abfs://`` scheme.
    By rewriting the URI to
    ``https://<account>.blob.core.windows.net/<container>/<path>?<sas_token>``
    the data can be opened via rasterio / GDAL's built-in HTTPS (vsicurl)
    handler without requiring ``adlfs`` at the driver level.

    Parameters
    ----------
    uri : str
        URI to convert.  If it does not start with ``abfs://`` or already
        contains a query string it is returned unchanged.
    account_name : str
        Azure Storage account name.
    sas_token : str
        Shared Access Signature token (without leading ``?``).

    Returns
    -------
    str
        HTTPS blob URL with the SAS token appended as a query string.
    """
    scheme, path = split_protocol(uri)
    # Handle ABFS URIs (case-insensitive scheme)
    if scheme and scheme.lower() == "abfs":
        return f"https://{account_name}.blob.core.windows.net/{path}?{sas_token}"
    # Handle HTTPS URIs (case-insensitive scheme) that do not yet have a query string
    if scheme and scheme.lower() == "https" and "?" not in uri:
        return f"{uri}?{sas_token}"
    return uri


def _normalise_uri(uri: str) -> tuple[str, str | None]:
    """Normalise an Azure URI to ``abfs://`` form.

    Parameters
    ----------
    uri : str
        One of the supported Azure URI styles (``abfs://``, HTTPS blob, or
        ``azureml://``).

    Returns
    -------
    tuple[str, str | None]
        ``(normalised_uri, account_name)``.  *account_name* is ``None``
        when using ``abfs://`` (the account is not embedded in the URI).

    Raises
    ------
    ValueError
        If the URI scheme is not recognised.
    """
    if _ABFS_RE.match(uri):
        return uri, None

    m = _HTTPS_BLOB_RE.match(uri)
    if m:
        account = m.group("account")
        rest = m.group("rest")  # "<container>/<path>"
        normalised = f"abfs://{rest}"
        logger.debug(
            f"AzureBlobResolver: converted HTTPS URI to '{normalised}' "
            f"(account='{account}')"
        )
        return normalised, account

    m = _AZUREML_RE.match(uri)
    if m:
        return _resolve_azureml_uri(
            subscription_id=m.group("subscription"),
            resource_group=m.group("resource_group"),
            workspace_name=m.group("workspace"),
            datastore_name=m.group("datastore"),
            path=m.group("path"),
        )

    raise ValueError(
        f"AzureBlobResolver: unrecognised URI scheme in '{uri}'. "
        "Expected 'abfs://', 'https://<account>.blob.core.windows.net/…', "
        "or 'azureml://subscriptions/…/datastores/…/paths/…'."
    )


def _resolve_azureml_uri(
    *,
    subscription_id: str,
    resource_group: str,
    workspace_name: str,
    datastore_name: str,
    path: str,
) -> tuple[str, str]:
    """Resolve an AzureML datastore URI to ``(abfs_uri, account_name)``.

    Uses the AzureML SDK (``azure-ai-ml``) to look up the datastore and
    extract the underlying storage account and container.

    Parameters
    ----------
    subscription_id : str
        Azure subscription ID.
    resource_group : str
        Azure resource group name.
    workspace_name : str
        AzureML workspace name.
    datastore_name : str
        Name of the registered datastore.
    path : str
        Blob path beneath the datastore container.

    Returns
    -------
    tuple[str, str]
        ``(abfs_uri, account_name)``.

    Raises
    ------
    ImportError
        If ``azure-ai-ml`` or ``azure-identity`` is not installed.
    """
    try:
        from azure.ai.ml import MLClient
    except ImportError as exc:
        raise ImportError(
            "AzureML datastore URIs require the 'azure-ai-ml' package. "
            "Install it with: pip install azure-ai-ml"
        ) from exc

    try:
        from azure.identity import DefaultAzureCredential
    except ImportError as exc:
        raise ImportError(
            "AzureML authentication requires the 'azure-identity' package. "
            "Install it with: pip install azure-identity"
        ) from exc

    credential = DefaultAzureCredential()
    ml_client = MLClient(
        credential=credential,
        subscription_id=subscription_id,
        resource_group_name=resource_group,
        workspace_name=workspace_name,
    )
    datastore = ml_client.datastores.get(datastore_name)

    account_name = datastore.account_name
    container_name = datastore.container_name
    normalised = f"abfs://{container_name}/{path}"
    logger.debug(
        f"AzureBlobResolver: resolved AzureML datastore '{datastore_name}' "
        f"to '{normalised}' (account='{account_name}')"
    )
    return normalised, account_name


def _resolve_azure_credentials(options: dict[str, Any]) -> dict[str, Any]:
    """Build ``storage_options`` for ``FSSpecFileSystem(protocol='abfs', ...)``.

    Parameters
    ----------
    options : dict[str, Any]
        User-supplied options.  Recognised credential keys are
        ``account_name``, ``account_key``, ``sas_token``,
        ``connection_string``, ``client_id``, ``client_secret``,
        ``tenant_id``, and ``anon``.

    Returns
    -------
    dict[str, Any]
        ``storage_options`` dict ready to pass to
        ``FSSpecFileSystem(protocol='abfs', ...)``.

    Notes
    -----
    Credential resolution order:

    1. Explicit values in *options* (``account_key``, ``sas_token``,
       ``connection_string``, ``client_id`` / ``client_secret`` /
       ``tenant_id`` for service principal).
    2. Environment variables (``AZURE_STORAGE_ACCOUNT_NAME``,
       ``AZURE_STORAGE_ACCOUNT_KEY``, ``AZURE_STORAGE_SAS_TOKEN``,
       ``AZURE_STORAGE_CONNECTION_STRING``).
    3. ``azure.identity.DefaultAzureCredential`` — covers Managed Identity,
       Azure CLI, VS Code, and environment-variable service principals.
    """
    opts = options.copy()

    # When anon=True is requested, skip all credential resolution and let
    # adlfs handle anonymous (unauthenticated) access directly.
    anon = opts.get("anon", False)
    if anon:
        account_name = opts.pop("account_name", None) or os.environ.get(
            "AZURE_STORAGE_ACCOUNT_NAME"
        )
        storage_options: dict[str, Any] = {}
        if account_name:
            storage_options["account_name"] = account_name
        storage_options.update(opts)
        logger.debug("AzureBlobResolver: anonymous access requested (anon=True)")
        return storage_options

    account_name: str | None = opts.pop("account_name", None) or os.environ.get(
        "AZURE_STORAGE_ACCOUNT_NAME"
    )
    account_key: str | None = opts.pop("account_key", None) or os.environ.get(
        "AZURE_STORAGE_ACCOUNT_KEY"
    )
    sas_token: str | None = opts.pop("sas_token", None) or os.environ.get(
        "AZURE_STORAGE_SAS_TOKEN"
    )
    connection_string: str | None = opts.pop(
        "connection_string", None
    ) or os.environ.get("AZURE_STORAGE_CONNECTION_STRING")

    client_id: str | None = opts.pop("client_id", None)
    client_secret: str | None = opts.pop("client_secret", None)
    tenant_id: str | None = opts.pop("tenant_id", None)

    storage_options: dict[str, Any] = {}

    if connection_string:
        storage_options["connection_string"] = connection_string
        logger.debug("AzureBlobResolver: authenticating via connection string")

    elif sas_token:
        storage_options["account_name"] = account_name
        storage_options["sas_token"] = sas_token
        logger.debug("AzureBlobResolver: authenticating via SAS token")

    elif account_key:
        storage_options["account_name"] = account_name
        storage_options["account_key"] = account_key
        logger.debug("AzureBlobResolver: authenticating via account key")

    elif client_id and client_secret and tenant_id:
        try:
            from azure.identity import ClientSecretCredential
        except ImportError as exc:
            raise ImportError(
                "Service-principal authentication requires the 'azure-identity' "
                "package. Install it with: pip install azure-identity  "
                "(or: pip install hydromt[io])"
            ) from exc
        credential = ClientSecretCredential(
            tenant_id=tenant_id,
            client_id=client_id,
            client_secret=client_secret,
        )
        storage_options["account_name"] = account_name
        storage_options["credential"] = credential
        logger.debug(
            "AzureBlobResolver: authenticating via service principal "
            f"(client_id='{client_id}')"
        )

    elif account_name:
        try:
            from azure.identity import DefaultAzureCredential
        except ImportError:
            logger.warning(
                "azure-identity is not installed; attempting anonymous/no-credential "
                "access. Install it with: pip install azure-identity  "
                "(or: pip install hydromt[io])"
            )
            storage_options["account_name"] = account_name
        else:
            storage_options["account_name"] = account_name
            storage_options["credential"] = DefaultAzureCredential()
            logger.debug(
                "AzureBlobResolver: authenticating via DefaultAzureCredential "
                f"(account='{account_name}')"
            )

    else:
        logger.debug(
            "AzureBlobResolver: no credentials found; attempting anonymous access. "
            "Set 'account_name' (at minimum) for non-public containers."
        )

    # Forward remaining user-supplied kwargs verbatim (e.g. 'anon', 'timeout').
    storage_options.update(opts)
    return storage_options
