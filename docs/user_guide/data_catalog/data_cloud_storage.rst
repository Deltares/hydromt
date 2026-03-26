.. _cloud_storage:

=============
Cloud Storage
=============

HydroMT can read data directly from cloud object stores — **Amazon S3**,
**Google Cloud Storage**, and **Microsoft Azure Blob Storage / ADLS Gen2** —
without downloading files manually.  All cloud access is built on `fsspec
<https://filesystem-spec.readthedocs.io>`_, so any protocol that fsspec
supports can be used.

Install the optional ``io`` dependencies to enable cloud storage access:

.. code-block:: bash

   pip install "hydromt[io]"

This installs ``s3fs`` (AWS), ``gcsfs`` (GCS), ``adlfs`` (Azure), and
``azure-identity`` / ``azure-ai-ml`` (Azure authentication and AzureML
datastore support).


Quick comparison
----------------

.. list-table::
   :widths: 15 25 25 35
   :header-rows: 1

   * - Provider
     - fsspec protocol
     - Required package
     - Example URI
   * - Amazon S3
     - ``s3``
     - ``s3fs``
     - ``s3://bucket/path/file.tif``
   * - Google Cloud Storage
     - ``gcs``
     - ``gcsfs``
     - ``gs://bucket/path/file.zarr``
   * - Azure Blob / ADLS Gen2
     - ``abfs``
     - ``adlfs``
     - ``abfs://container/path/file.nc``


.. _cloud_simple:

Simple cloud access (any provider)
-----------------------------------

The simplest way to read from any cloud store is to set the **filesystem**
on the driver — exactly as you would for a local file, but with a cloud
protocol.  This works identically for S3, GCS, and Azure and uses the
default :py:class:`~hydromt.data_catalog.uri_resolvers.ConventionResolver`.

**AWS S3 (anonymous)**

.. code-block:: yaml

   esa_worldcover:
     data_type: RasterDataset
     uri: s3://esa-worldcover/v100/2020/ESA_WorldCover_10m_2020_v100_Map_AWS.vrt
     driver:
       name: rasterio
       filesystem:
         protocol: s3
         anon: true

**Google Cloud Storage**

.. code-block:: yaml

   cmip6_historical:
     data_type: RasterDataset
     uri: gs://cmip6/CMIP6/CMIP/MPI-ESM1-2-HR/historical/r1i1p1f1/day/tas/*/*
     driver:
       name: raster_xarray
       filesystem:
         protocol: gcs

**Azure Blob Storage (anonymous)**

.. code-block:: yaml

   noaa_isd:
     data_type: DataFrame
     uri: abfs://isdweatherdatacontainer/ISDWeather/year=2020/month=1/*.parquet
     driver:
       name: pandas
       filesystem:
         protocol: abfs
         account_name: azureopendatastorage
         anon: true

In all three cases, HydroMT:

1. Creates an fsspec filesystem from the ``filesystem:`` block (e.g.
   ``adlfs.AzureBlobFileSystem(account_name=..., anon=True)``).
2. Passes the URI to :py:class:`~hydromt.data_catalog.uri_resolvers.ConventionResolver`,
   which calls ``fs.glob()`` to resolve wildcards.
3. Hands the resolved URIs to the driver for reading.

The Convention Resolver is **cloud-agnostic** — it doesn't know or care which
provider is behind the filesystem.  All provider-specific logic lives in the
fsspec implementation (``s3fs``, ``gcsfs``, ``adlfs``).

**When to use this approach:**

- Public / anonymous containers
- Containers where you manage credentials via environment variables that the
  fsspec implementation picks up automatically
- Simple ``abfs://`` URIs without SAS tokens, HTTPS blob URLs, or AzureML
  datastore URIs


.. _azure_blob_resolver:

Azure Blob Resolver
-------------------

For Azure-specific scenarios that go beyond what the generic approach offers,
HydroMT provides a dedicated
:py:class:`~hydromt.data_catalog.uri_resolvers.AzureBlobResolver`.

Use it when you need any of:

- **URI normalisation** — ``https://<account>.blob.core.windows.net/…`` or
  ``azureml://subscriptions/…/datastores/…/paths/…`` URIs, which are
  automatically converted to ``abfs://`` internally.
- **Automatic SAS token fetching** — e.g. from the `Planetary Computer
  <https://planetarycomputer.microsoft.com>`_ SAS API.
- **Azure credential chain** — the resolver walks explicit options →
  environment variables → ``DefaultAzureCredential`` (Managed Identity, Azure
  CLI, VS Code login, service principals) without manual configuration.
- **HTTPS output for GDAL / rasterio** — when a SAS token is available,
  ``abfs://`` URIs are transparently converted to signed HTTPS blob URLs that
  rasterio/GDAL can open directly (since those libraries do not understand the
  ``abfs://`` scheme).


When *not* to use it
^^^^^^^^^^^^^^^^^^^^

If you only have ``abfs://`` URIs and can pass credentials via the
``filesystem:`` block (or environment variables), the simpler
:ref:`generic approach <cloud_simple>` is sufficient and keeps your catalog
entries provider-agnostic.


Configuration
^^^^^^^^^^^^^

Enable the resolver by adding ``uri_resolver: name: azure_blob`` to the data
source.  Options are passed under ``uri_resolver.options``.

.. code-block:: yaml

   my_dataset:
     data_type: RasterDataset
     uri: abfs://container/path/to/data.tif
     driver:
       name: rasterio
     uri_resolver:
       name: azure_blob
       options:
         account_name: mystorageaccount
         # ... authentication options, see below

.. note::

   When a ``uri_resolver`` is specified, the resolver manages filesystem
   creation.  You do **not** need to set ``filesystem:`` on the driver — the
   resolver will create an ``abfs`` filesystem and propagate it to the driver
   automatically.


Supported URI styles
^^^^^^^^^^^^^^^^^^^^

.. list-table::
   :widths: 30 70
   :header-rows: 1

   * - Style
     - Example
   * - ADLS Gen2 (native)
     - ``abfs://mycontainer/path/to/data.tif``
   * - HTTPS Blob endpoint
     - ``https://myaccount.blob.core.windows.net/mycontainer/path/data.tif``
   * - AzureML datastore
     - ``azureml://subscriptions/<sub>/resourcegroups/<rg>/workspaces/<ws>/datastores/<ds>/paths/<path>``

All styles are normalised to ``abfs://`` internally.  HTTPS blob URLs have
their ``account_name`` extracted from the URL automatically, so you do not
need to specify it again.  AzureML URIs require the ``azure-ai-ml`` package
and resolve the datastore to its underlying storage account / container via
the AzureML SDK.


Authentication options
^^^^^^^^^^^^^^^^^^^^^^

Credentials are resolved in the following order of precedence:

1. **Explicit values** in ``uri_resolver.options``:

   .. list-table::
      :widths: 30 70
      :header-rows: 1

      * - Option
        - Description
      * - ``account_name``
        - Storage account name (required for most methods)
      * - ``account_key``
        - Storage account access key
      * - ``sas_token``
        - Shared Access Signature token string
      * - ``connection_string``
        - Full connection string (takes highest precedence)
      * - ``client_id``, ``client_secret``, ``tenant_id``
        - Service principal credentials
      * - ``anon: true``
        - Anonymous access (public containers, skips all credential resolution)
      * - ``sas_token_url``
        - URL returning JSON with a ``"token"`` key; a fresh SAS token is
          fetched automatically before each ``resolve()`` call

2. **Environment variables** (recognised by ``adlfs`` / ``azure-storage-blob``):

   - ``AZURE_STORAGE_ACCOUNT_NAME``
   - ``AZURE_STORAGE_ACCOUNT_KEY``
   - ``AZURE_STORAGE_SAS_TOKEN``
   - ``AZURE_STORAGE_CONNECTION_STRING``

3. **DefaultAzureCredential** (from ``azure-identity``): covers Managed
   Identity, Azure CLI login, VS Code login, environment-variable service
   principals, and more.  Activated automatically when ``account_name`` is set
   but no explicit key/token is provided.


Time-templated URIs
^^^^^^^^^^^^^^^^^^^

The resolver supports the same placeholder expansion as the Convention
Resolver: ``{year}``, ``{month}``, ``{day}``, and ``{variable}``.

.. code-block:: yaml

   rainfall:
     data_type: RasterDataset
     uri: abfs://hydrodata/rainfall/{year}/{month}/precip.nc
     driver:
       name: raster_xarray
     uri_resolver:
       name: azure_blob
       options:
         account_name: mystorageaccount
         account_key: "..."


ABFS to HTTPS conversion
^^^^^^^^^^^^^^^^^^^^^^^^^

Internally, ``AzureBlobResolver`` normalises every URI to the ``abfs://``
scheme so that fsspec-based drivers (e.g. xarray with zarr) can open data via
``adlfs`` directly.

However, **rasterio and GDAL** do not understand the ``abfs://`` scheme.
When a SAS token is available (either supplied explicitly or fetched from a
``sas_token_url``), the resolver automatically rewrites the resolved
``abfs://`` URIs to HTTPS blob URLs of the form::

    https://<account>.blob.core.windows.net/<container>/<path>?<sas_token>

This allows rasterio / GDAL to open the data through their built-in HTTPS
(``/vsicurl/``) handler without any additional configuration.  The
conversion only takes place when **both** an ``account_name`` **and** a
``sas_token`` are available; otherwise the ``abfs://`` URIs are returned
as-is for fsspec-based drivers.


Examples
^^^^^^^^

**Anonymous public container**

.. code-block:: yaml

   noaa_isd:
     data_type: DataFrame
     uri: abfs://isdweatherdatacontainer/ISDWeather/year=2020/month=1/*.parquet
     driver:
       name: pandas
     uri_resolver:
       name: azure_blob
       options:
         account_name: azureopendatastorage
         anon: true

**Planetary Computer with automatic SAS token fetching**

.. code-block:: yaml

   esa_worldcover:
     data_type: RasterDataset
     uri: abfs://esa-worldcover/v200/2021/map/ESA_WorldCover_10m_2021_v200_N51E003_Map.tif
     driver:
       name: rasterio
     uri_resolver:
       name: azure_blob
       options:
         account_name: ai4edataeuwest
         sas_token_url: https://planetarycomputer.microsoft.com/api/sas/v1/token/esa-worldcover

**AzureML datastore URI**

.. code-block:: yaml

   uk_compass_flood:
     data_type: RasterDataset
     uri: azureml://subscriptions/d1d42766-9d6a-4571-8b38-bbc3e60604bb/resourcegroups/rg-ukcrcollab-ukcrcompassflood/workspaces/mlw-ukcrcompassflood-uksouth-01/datastores/large_datastore/paths/default/bucket
     driver:
       name: raster_xarray
     uri_resolver:
       name: azure_blob

The AzureML SDK will look up the datastore, extract the storage account and
container, and authentication is handled via ``DefaultAzureCredential``.

**HTTPS blob URL with explicit SAS token**

.. code-block:: yaml

   coastal_dem:
     data_type: RasterDataset
     uri: https://myaccount.blob.core.windows.net/public-data/dem/uk_2m.tif
     driver:
       name: rasterio
     uri_resolver:
       name: azure_blob
       options:
         sas_token: "sv=2022-11-02&ss=b&srt=co&sp=rl&se=..."


Decision guide
--------------

.. list-table::
   :widths: 55 22 23
   :header-rows: 1

   * - Scenario
     - Convention Resolver
     - Azure Blob Resolver
   * - Public ``abfs://`` container (anonymous)
     - Yes
     - Yes
   * - Private ``abfs://`` container (key/SAS via env vars)
     - Yes
     - Yes
   * - HTTPS blob URLs (``https://<acct>.blob.core.windows.net/…``)
     - No
     - **Yes**
   * - AzureML datastore URIs (``azureml://…``)
     - No
     - **Yes**
   * - Automatic SAS token fetching from a token API
     - No
     - **Yes**
   * - Azure credential chain (DefaultAzureCredential)
     - No
     - **Yes**
   * - rasterio / GDAL needs signed HTTPS URLs
     - No
     - **Yes** (automatic)
   * - S3 or GCS data
     - **Yes**
     - No (Azure only)
