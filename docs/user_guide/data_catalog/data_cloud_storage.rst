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


See :ref:`choosing_resolver` for guidance on when the Convention Resolver is
sufficient instead.


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

   flood_hazard_maps:
     data_type: RasterDataset
     uri: azureml://subscriptions/00000000-aaaa-bbbb-cccc-123456789abc/resourcegroups/rg-my-project/workspaces/mlw-my-workspace/datastores/project_datastore/paths/hazard/flood_depth_100yr.tif
     driver:
       name: raster_xarray
     uri_resolver:
       name: azure_blob

The AzureML SDK will look up the datastore, extract the underlying storage
account and container, and build an ``abfs://`` path automatically.
Authentication is handled via ``DefaultAzureCredential``.

For examples with explicit SAS tokens, see :ref:`azure_sas_quickstart`.


.. _choosing_resolver:

Choosing between resolvers
--------------------------

Start with the Convention Resolver for simple, public, or
environment-variable-authenticated ``abfs://`` access.  Switch to the Azure
Blob Resolver the moment you need SAS tokens, non-``abfs://`` URIs, or
GDAL/rasterio compatibility with private data.

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


.. _azure_sas_quickstart:

Step-by-step: accessing private Azure Blob Storage with a SAS token
-------------------------------------------------------------------

This section walks through the steps to access data stored in a **private**
Azure storage account.  The workflow is: sign in to Azure, generate a SAS
token with the required permissions, and reference that token in your data
catalog.

1 — Generate a SAS token
^^^^^^^^^^^^^^^^^^^^^^^^^

A SAS (Shared Access Signature) token grants time-limited, scoped access to a
container or blob.  You can create one from the Azure CLI, the Azure Portal,
or Azure Storage Explorer.

**Azure Portal**

1. Navigate to **Storage accounts** → your storage account.
2. Open **Containers** → select the container → **Shared access tokens**
   (or use **Shared access signature** from the storage account menu for
   account-level tokens).
3. Set **Allowed permissions** to *Read* and *List*, choose an expiry
   date/time, and click **Generate SAS token and URL**.
4. Copy the **SAS token** value (starts with ``sp=`` or ``sv=``).


2 — Add the SAS token to your data catalog
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Open your data catalog YAML file and add a source that uses the
``azure_blob`` resolver.  Below are examples for each supported URI style.

**``abfs://`` URI**

.. code-block:: yaml

   my_dataset:
     data_type: RasterDataset
     uri: abfs://<container>/<path-to-data>/*.tif
     driver:
       name: rasterio
     uri_resolver:
       name: azure_blob
       options:
         account_name: <storage-account-name>
         sas_token: <paste-your-sas-token-here>

**AzureML datastore URI**

.. code-block:: yaml

   my_dataset:
     data_type: RasterDataset
     uri: azureml://subscriptions/<sub-id>/resourcegroups/<rg>/workspaces/<ws>/datastores/<datastore>/paths/<path>.tif
     driver:
       name: rasterio
     uri_resolver:
       name: azure_blob
       options:
         sas_token: <paste-your-sas-token-here>

**HTTPS blob URL**

.. code-block:: yaml

   my_dataset:
     data_type: RasterDataset
     uri: https://<account>.blob.core.windows.net/<container>/<path>.tif
     driver:
       name: rasterio
     uri_resolver:
       name: azure_blob
       options:
         sas_token: "<paste-your-sas-token-here>"

With HTTPS blob URLs the ``account_name`` is extracted from the URL
automatically, so you do not need to specify it separately.  You can also
append the SAS token directly to the URL as a query string
(``https://…/<path>.tif?sp=rl&st=…``) and omit the ``sas_token`` option.

.. tip::

   To keep credentials out of version control, set the
   ``AZURE_STORAGE_SAS_TOKEN`` environment variable instead.  The resolver
   picks it up automatically when no explicit ``sas_token`` is provided:

   .. code-block:: bash

      export AZURE_STORAGE_SAS_TOKEN="sp=rl&st=2026-03-30T09:00:00Z&se=..."

   On Windows (PowerShell):

   .. code-block:: powershell

      $env:AZURE_STORAGE_SAS_TOKEN = "sp=rl&st=2026-03-30T09:00:00Z&se=..."


3 — Verify access
^^^^^^^^^^^^^^^^^^

Test that HydroMT can read the data:

.. code-block:: python

   import hydromt

   cat = hydromt.DataCatalog("path/to/my_catalog.yml")
   ds = cat.get_rasterdataset("my_dataset")
   print(ds)
