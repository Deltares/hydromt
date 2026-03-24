"""Tests for the AzureBlobResolver and its module-level helper functions."""

from datetime import datetime
from unittest.mock import MagicMock, patch

import pytest

from hydromt.data_catalog.uri_resolvers.azure_blob_resolver import (
    AzureBlobResolver,
    _abfs_to_https,
    _fetch_sas_token,
    _normalise_uri,
    _resolve_azure_credentials,
    _resolve_azureml_uri,
)
from hydromt.error import NoDataException, NoDataStrategy
from hydromt.typing.fsspec_types import FSSpecFileSystem
from hydromt.typing.type_def import TimeRange

# ---------------------------------------------------------------------------
# _normalise_uri
# ---------------------------------------------------------------------------


class TestNormaliseUri:
    def test_abfs_uri_returned_unchanged(self):
        uri = "abfs://mycontainer/path/to/file.tif"
        normalised, account = _normalise_uri(uri)
        assert normalised == uri
        assert account is None

    def test_abfs_uri_case_insensitive(self):
        uri = "ABFS://Container/Data.nc"
        normalised, account = _normalise_uri(uri)
        assert normalised == uri
        assert account is None

    def test_https_blob_uri_converted(self):
        uri = "https://myaccount.blob.core.windows.net/mycontainer/path/data.tif"
        normalised, account = _normalise_uri(uri)
        assert normalised == "abfs://mycontainer/path/data.tif"
        assert account == "myaccount"

    def test_https_blob_uri_case_insensitive(self):
        uri = "HTTPS://StorageAcct.Blob.Core.Windows.Net/container/file.nc"
        normalised, account = _normalise_uri(uri)
        assert normalised == "abfs://container/file.nc"
        assert account == "StorageAcct"

    def test_invalid_scheme_raises(self):
        with pytest.raises(ValueError, match="unrecognised URI scheme"):
            _normalise_uri("s3://bucket/key")

    def test_plain_path_raises(self):
        with pytest.raises(ValueError, match="unrecognised URI scheme"):
            _normalise_uri("/local/path/file.tif")

    def test_https_non_blob_raises(self):
        with pytest.raises(ValueError, match="unrecognised URI scheme"):
            _normalise_uri("https://example.com/data.tif")

    def test_azureml_uri_dispatches(self):
        """azureml:// URIs are recognised and dispatched to _resolve_azureml_uri."""
        mock_ds = MagicMock()
        mock_ds.account_name = "mystorageacct"
        mock_ds.container_name = "mycontainer"

        with patch(
            "hydromt.data_catalog.uri_resolvers.azure_blob_resolver._resolve_azureml_uri",
            return_value=("abfs://mycontainer/some/path", "mystorageacct"),
        ) as mock_resolve:
            result, account = _normalise_uri(
                "azureml://subscriptions/sub123/resourcegroups/rg1/"
                "workspaces/ws1/datastores/ds1/paths/some/path"
            )
            mock_resolve.assert_called_once_with(
                subscription_id="sub123",
                resource_group="rg1",
                workspace_name="ws1",
                datastore_name="ds1",
                path="some/path",
            )
        assert result == "abfs://mycontainer/some/path"
        assert account == "mystorageacct"

    def test_azureml_uri_case_insensitive(self):
        """azureml:// scheme matching is case-insensitive."""
        with patch(
            "hydromt.data_catalog.uri_resolvers.azure_blob_resolver._resolve_azureml_uri",
            return_value=("abfs://c/p", "acct"),
        ) as mock_resolve:
            _normalise_uri(
                "AzureML://subscriptions/s/resourcegroups/r/"
                "workspaces/w/datastores/d/paths/p"
            )
            mock_resolve.assert_called_once()


# ---------------------------------------------------------------------------
# _resolve_azureml_uri
# ---------------------------------------------------------------------------


class TestResolveAzuremlUri:
    _ML_CLIENT_PATH = "hydromt.data_catalog.uri_resolvers.azure_blob_resolver.MLClient"
    _DEFAULT_CRED_PATH = (
        "hydromt.data_catalog.uri_resolvers.azure_blob_resolver.DefaultAzureCredential"
    )

    def _patch_azureml(
        self, account_name="mystorageacct", container_name="mycontainer"
    ):
        """Return a context manager that patches MLClient + DefaultAzureCredential."""
        mock_ds = MagicMock()
        mock_ds.account_name = account_name
        mock_ds.container_name = container_name

        mock_ml_client = MagicMock()
        mock_ml_client.datastores.get.return_value = mock_ds

        # We need to patch the imports inside the function.
        azure_identity_mod = MagicMock()
        azure_ai_ml_mod = MagicMock()
        azure_ai_ml_mod.MLClient.return_value = mock_ml_client

        return patch.dict(
            "sys.modules",
            {
                "azure": MagicMock(),
                "azure.identity": azure_identity_mod,
                "azure.ai": MagicMock(),
                "azure.ai.ml": azure_ai_ml_mod,
            },
        )

    def test_resolves_to_abfs(self):
        """A valid AzureML URI resolves to abfs://container/path."""
        mock_ds = MagicMock()
        mock_ds.account_name = "mystorageacct"
        mock_ds.container_name = "mycontainer"

        mock_ml_client_instance = MagicMock()
        mock_ml_client_instance.datastores.get.return_value = mock_ds

        mock_ml_client_cls = MagicMock(return_value=mock_ml_client_instance)
        mock_cred_cls = MagicMock()

        with patch.dict(
            "sys.modules",
            {
                "azure": MagicMock(),
                "azure.ai": MagicMock(),
                "azure.ai.ml": MagicMock(MLClient=mock_ml_client_cls),
                "azure.identity": MagicMock(DefaultAzureCredential=mock_cred_cls),
            },
        ):
            uri, account = _resolve_azureml_uri(
                subscription_id="sub123",
                resource_group="rg1",
                workspace_name="ws1",
                datastore_name="ds1",
                path="default/bucket",
            )

        assert uri == "abfs://mycontainer/default/bucket"
        assert account == "mystorageacct"
        mock_ml_client_instance.datastores.get.assert_called_once_with("ds1")


# ---------------------------------------------------------------------------
# _resolve_azure_credentials
# ---------------------------------------------------------------------------


class TestResolveAzureCredentials:
    def test_connection_string(self):
        opts = _resolve_azure_credentials(
            {"connection_string": "DefaultEndpointsProtocol=https;..."}
        )
        assert opts["connection_string"] == "DefaultEndpointsProtocol=https;..."

    def test_sas_token(self):
        opts = _resolve_azure_credentials(
            {"account_name": "acct", "sas_token": "sv=2022&ss=b"}
        )
        assert opts["account_name"] == "acct"
        assert opts["sas_token"] == "sv=2022&ss=b"

    def test_account_key(self):
        opts = _resolve_azure_credentials(
            {"account_name": "acct", "account_key": "key123"}
        )
        assert opts["account_name"] == "acct"
        assert opts["account_key"] == "key123"

    def test_env_vars_fallback(self, monkeypatch):
        monkeypatch.setenv("AZURE_STORAGE_ACCOUNT_NAME", "env_acct")
        monkeypatch.setenv("AZURE_STORAGE_ACCOUNT_KEY", "env_key")
        opts = _resolve_azure_credentials({})
        assert opts["account_name"] == "env_acct"
        assert opts["account_key"] == "env_key"

    def test_anonymous_access(self, monkeypatch):
        monkeypatch.delenv("AZURE_STORAGE_ACCOUNT_NAME", raising=False)
        monkeypatch.delenv("AZURE_STORAGE_ACCOUNT_KEY", raising=False)
        monkeypatch.delenv("AZURE_STORAGE_SAS_TOKEN", raising=False)
        monkeypatch.delenv("AZURE_STORAGE_CONNECTION_STRING", raising=False)
        opts = _resolve_azure_credentials({})
        assert "account_key" not in opts
        assert "sas_token" not in opts
        assert "connection_string" not in opts

    def test_extra_kwargs_forwarded(self):
        opts = _resolve_azure_credentials({"anon": True, "timeout": 30})
        assert opts["anon"] is True
        assert opts["timeout"] == 30

    def test_anon_skips_credential_resolution(self):
        """When anon=True, no credential should be resolved even if account_name is set."""
        opts = _resolve_azure_credentials({"account_name": "myacct", "anon": True})
        assert opts["account_name"] == "myacct"
        assert opts["anon"] is True
        assert "credential" not in opts
        assert "account_key" not in opts
        assert "sas_token" not in opts

    def test_does_not_mutate_input(self):
        options = {"account_name": "acct", "account_key": "key"}
        original = options.copy()
        _resolve_azure_credentials(options)
        assert options == original

    def test_service_principal(self):
        with patch(
            "hydromt.data_catalog.uri_resolvers.azure_blob_resolver.ClientSecretCredential",
            create=True,
        ) as mock_cred_cls:
            mock_cred = MagicMock()
            mock_cred_cls.return_value = mock_cred
            # Patch the import inside the function
            with patch.dict(
                "sys.modules",
                {"azure": MagicMock(), "azure.identity": MagicMock()},
            ):
                with patch(
                    "hydromt.data_catalog.uri_resolvers.azure_blob_resolver.ClientSecretCredential",
                    create=True,
                ):
                    opts = _resolve_azure_credentials(
                        {
                            "account_name": "acct",
                            "client_id": "cid",
                            "client_secret": "csecret",
                            "tenant_id": "tid",
                        }
                    )
        assert opts["account_name"] == "acct"
        assert "credential" in opts

    def test_precedence_connection_string_over_sas(self):
        """Connection string takes precedence over SAS token."""
        opts = _resolve_azure_credentials(
            {
                "connection_string": "conn_str",
                "sas_token": "sas",
                "account_name": "acct",
            }
        )
        assert "connection_string" in opts
        assert "sas_token" not in opts


# ---------------------------------------------------------------------------
# AzureBlobResolver._get_dates
# ---------------------------------------------------------------------------


class TestGetDates:
    def test_yearly(self):
        tr = TimeRange(start=datetime(2020, 1, 1), end=datetime(2022, 12, 31))
        dates = AzureBlobResolver._get_dates(["year"], tr)
        assert len(dates) == 3
        assert dates[0].year == 2020
        assert dates[-1].year == 2022

    def test_monthly(self):
        tr = TimeRange(start=datetime(2021, 10, 1), end=datetime(2022, 1, 31))
        dates = AzureBlobResolver._get_dates(["year", "month"], tr)
        assert len(dates) == 4
        assert dates[0].month == 10
        assert dates[-1].month == 1

    def test_daily(self):
        tr = TimeRange(start=datetime(2021, 3, 28), end=datetime(2021, 4, 1))
        dates = AzureBlobResolver._get_dates(["year", "month", "day"], tr)
        assert len(dates) == 5

    def test_single_date(self):
        tr = TimeRange(start=datetime(2021, 6, 15), end=datetime(2021, 6, 15))
        dates = AzureBlobResolver._get_dates(["year"], tr)
        assert len(dates) == 1


# ---------------------------------------------------------------------------
# AzureBlobResolver._ensure_azure_filesystem
# ---------------------------------------------------------------------------


_RESOLVE_CREDS_PATH = (
    "hydromt.data_catalog.uri_resolvers.azure_blob_resolver._resolve_azure_credentials"
)
_FSSPEC_FS_PATH = (
    "hydromt.data_catalog.uri_resolvers.azure_blob_resolver.FSSpecFileSystem"
)


class TestEnsureAzureFilesystem:
    def test_skips_when_already_configured(self):
        resolver = AzureBlobResolver()
        mock_fs = MagicMock(spec=FSSpecFileSystem)
        mock_fs.protocol = "abfs"
        resolver.filesystem = mock_fs
        resolver._ensure_azure_filesystem({"account_name": "acct"})
        # filesystem should not have been replaced
        assert resolver.filesystem is mock_fs

    def test_builds_when_default_local(self):
        resolver = AzureBlobResolver()
        with patch(_RESOLVE_CREDS_PATH, return_value={"account_name": "acct"}):
            with patch(_FSSPEC_FS_PATH) as mock_fs_cls:
                mock_new_fs = MagicMock()
                mock_fs_cls.return_value = mock_new_fs
                resolver._ensure_azure_filesystem({"account_name": "acct"})
                mock_fs_cls.assert_called_once_with(
                    protocol="abfs",
                    storage_options={"account_name": "acct"},
                )

    def test_auth_failure_raises_permission_error(self):
        resolver = AzureBlobResolver()
        with patch(_RESOLVE_CREDS_PATH, return_value={"account_name": "acct"}):
            with patch(
                _FSSPEC_FS_PATH,
                side_effect=Exception("credential chain failed"),
            ):
                with pytest.raises(PermissionError, match="failed to authenticate"):
                    resolver._ensure_azure_filesystem({"account_name": "acct"})


# ---------------------------------------------------------------------------
# AzureBlobResolver.resolve  (integration-level tests with mocked filesystem)
# ---------------------------------------------------------------------------


class TestAzureBlobResolverResolve:
    def _make_resolver(self, glob_results=None, options=None):
        """Create an AzureBlobResolver with a mocked FSSpecFileSystem.

        *glob_results* maps glob patterns to lists of matched paths.
        """
        resolver = AzureBlobResolver(options=options or {})
        mock_fs = MagicMock()
        mock_fs.protocol = "abfs"

        glob_map = glob_results or {}

        def _glob_side_effect(pattern):
            return glob_map.get(pattern, [])

        mock_inner_fs = MagicMock()
        mock_inner_fs.glob.side_effect = _glob_side_effect
        mock_inner_fs.unstrip_protocol.side_effect = lambda p: f"abfs://{p}"

        mock_fsspec = MagicMock(spec=FSSpecFileSystem)
        mock_fsspec.protocol = "abfs"
        mock_fsspec.get_fs.return_value = mock_inner_fs

        resolver.filesystem = mock_fsspec
        return resolver

    def test_simple_abfs_uri(self):
        resolver = self._make_resolver(
            {"abfs://container/data.tif": ["container/data.tif"]}
        )
        result = resolver.resolve("abfs://container/data.tif")
        assert "abfs://container/data.tif" in result

    def test_https_blob_uri_normalises_and_resolves(self):
        resolver = self._make_resolver(
            {"abfs://mycontainer/path/data.tif": ["mycontainer/path/data.tif"]}
        )
        result = resolver.resolve(
            "https://acct.blob.core.windows.net/mycontainer/path/data.tif"
        )
        assert "abfs://mycontainer/path/data.tif" in result

    def test_account_from_options_takes_precedence(self):
        resolver = self._make_resolver(
            {"abfs://c/f.tif": ["c/f.tif"]},
            options={"account_name": "explicit_acct"},
        )
        result = resolver.resolve("https://uri_acct.blob.core.windows.net/c/f.tif")
        assert len(result) >= 1

    def test_time_templated_uri(self):
        resolver = self._make_resolver(
            {
                "abfs://data/2020/1/precip.nc": ["data/2020/1/precip.nc"],
                "abfs://data/2020/2/precip.nc": ["data/2020/2/precip.nc"],
                "abfs://data/2020/3/precip.nc": ["data/2020/3/precip.nc"],
            }
        )
        tr = TimeRange(start=datetime(2020, 1, 1), end=datetime(2020, 3, 31))
        result = resolver.resolve("abfs://data/{year}/{month}/precip.nc", time_range=tr)
        assert len(result) == 3
        assert "abfs://data/2020/1/precip.nc" in result
        assert "abfs://data/2020/2/precip.nc" in result
        assert "abfs://data/2020/3/precip.nc" in result

    def test_time_templated_missing_months(self):
        resolver = self._make_resolver(
            {"abfs://data/2020/1/precip.nc": ["data/2020/1/precip.nc"]}
        )
        tr = TimeRange(start=datetime(2020, 1, 1), end=datetime(2020, 3, 31))
        result = resolver.resolve("abfs://data/{year}/{month}/precip.nc", time_range=tr)
        assert result == ["abfs://data/2020/1/precip.nc"]

    def test_no_data_raises(self):
        resolver = self._make_resolver()  # empty glob results
        with pytest.raises(NoDataException):
            resolver.resolve("abfs://container/missing.tif")

    def test_no_data_ignore_returns_empty(self):
        resolver = self._make_resolver()
        result = resolver.resolve(
            "abfs://container/missing.tif",
            handle_nodata=NoDataStrategy.IGNORE,
        )
        assert result == []

    def test_invalid_scheme_raises(self):
        resolver = self._make_resolver()
        with pytest.raises(ValueError, match="unrecognised URI scheme"):
            resolver.resolve("s3://bucket/key")

    def test_variable_expansion(self):
        resolver = self._make_resolver(
            {
                "abfs://c/temp/data.nc": ["c/temp/data.nc"],
                "abfs://c/precip/data.nc": ["c/precip/data.nc"],
            }
        )
        result = resolver.resolve(
            "abfs://c/{variable}/data.nc", variables=["temp", "precip"]
        )
        assert len(result) == 2
        assert "abfs://c/temp/data.nc" in result
        assert "abfs://c/precip/data.nc" in result

    def test_time_and_variable_expansion(self):
        resolver = self._make_resolver(
            {
                "abfs://d/2020/temp.nc": ["d/2020/temp.nc"],
                "abfs://d/2020/precip.nc": ["d/2020/precip.nc"],
                "abfs://d/2021/temp.nc": ["d/2021/temp.nc"],
                "abfs://d/2021/precip.nc": ["d/2021/precip.nc"],
            }
        )
        tr = TimeRange(start=datetime(2020, 1, 1), end=datetime(2021, 12, 31))
        result = resolver.resolve(
            "abfs://d/{year}/{variable}.nc",
            time_range=tr,
            variables=["temp", "precip"],
        )
        assert len(result) == 4

    def test_azureml_uri(self):
        """An azureml:// URI is normalised then resolved like any abfs:// URI."""
        resolver = self._make_resolver(
            {"abfs://mycontainer/default/bucket": ["mycontainer/default/bucket"]}
        )
        azureml_uri = (
            "azureml://subscriptions/sub123/resourcegroups/rg1/"
            "workspaces/ws1/datastores/ds1/paths/default/bucket"
        )
        with patch(
            "hydromt.data_catalog.uri_resolvers.azure_blob_resolver._resolve_azureml_uri",
            return_value=("abfs://mycontainer/default/bucket", "mystorageacct"),
        ):
            result = resolver.resolve(azureml_uri)
        assert "abfs://mycontainer/default/bucket" in result

    def test_sas_token_url_fetches_and_converts(self):
        """Fetch a SAS token and returns HTTPS URLs instead of abfs:// URIs."""
        resolver = self._make_resolver(
            {"abfs://c/data.tif": ["c/data.tif"]},
            options={
                "account_name": "myacct",
                "sas_token_url": "https://example.com/token",
            },
        )
        with patch(
            "hydromt.data_catalog.uri_resolvers.azure_blob_resolver._fetch_sas_token",
            return_value="sv=2021&sig=abc",
        ):
            result = resolver.resolve("abfs://c/data.tif")
        assert len(result) == 1
        assert (
            result[0]
            == "https://myacct.blob.core.windows.net/c/data.tif?sv=2021&sig=abc"
        )

    def test_explicit_sas_token_skips_fetch(self):
        """An explicit sas_token takes precedence over sas_token_url."""
        resolver = self._make_resolver(
            {"abfs://c/data.tif": ["c/data.tif"]},
            options={
                "account_name": "myacct",
                "sas_token": "sv=existing",
                "sas_token_url": "https://example.com/token",
            },
        )
        with patch(
            "hydromt.data_catalog.uri_resolvers.azure_blob_resolver._fetch_sas_token",
        ) as mock_fetch:
            result = resolver.resolve("abfs://c/data.tif")
        mock_fetch.assert_not_called()
        assert "sv=existing" in result[0]


class TestFetchSasToken:
    def test_success(self):
        with patch("urllib.request.urlopen") as mock_urlopen:
            mock_resp = MagicMock()
            mock_resp.read.return_value = b'{"token": "sv=2021&sig=xyz"}'
            mock_resp.__enter__ = lambda s: s
            mock_resp.__exit__ = MagicMock(return_value=False)
            mock_urlopen.return_value = mock_resp
            assert _fetch_sas_token("https://example.com/token") == "sv=2021&sig=xyz"

    def test_failure_raises_permission_error(self):
        with patch("urllib.request.urlopen", side_effect=Exception("timeout")):
            with pytest.raises(PermissionError, match="failed to fetch SAS token"):
                _fetch_sas_token("https://example.com/token")


class TestAbfsToHttps:
    def test_converts_abfs(self):
        assert (
            _abfs_to_https("abfs://c/path/file.tif", "acct", "sv=2021")
            == "https://acct.blob.core.windows.net/c/path/file.tif?sv=2021"
        )

    def test_appends_to_https(self):
        url = "https://acct.blob.core.windows.net/c/file.tif"
        assert _abfs_to_https(url, "acct", "sv=2021") == f"{url}?sv=2021"

    def test_skips_https_with_existing_query(self):
        url = "https://acct.blob.core.windows.net/c/file.tif?existing=1"
        assert _abfs_to_https(url, "acct", "sv=2021") == url

    def test_other_scheme_unchanged(self):
        assert _abfs_to_https("s3://bucket/key", "acct", "sv=2021") == "s3://bucket/key"
