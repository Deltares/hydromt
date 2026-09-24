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

For private S3 buckets, see :ref:`private_s3`.  For private Azure storage,
see :doc:`azure_blob_storage`.


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
- Private S3 buckets, using credentials from your AWS configuration files
  (see :ref:`private_s3`)
- Simple ``abfs://`` URIs without SAS tokens, HTTPS blob URLs, or AzureML
  datastore URIs


.. _private_s3:

Reading from private S3 buckets
-------------------------------

Private S3 buckets (including S3-compatible object stores with a custom
endpoint) are accessed through the same ``filesystem:`` block as public
buckets, but with credentials.  HydroMT hands the ``filesystem:`` options to
``s3fs``, which reads credentials from your AWS configuration files, so no
secrets need to end up in the data catalog.


Configure credentials
^^^^^^^^^^^^^^^^^^^^^

Create the AWS credentials file.  On Windows this is
``%USERPROFILE%\.aws\credentials``, on Linux and macOS ``~/.aws/credentials``.
Add one section per profile, for example one per bucket or account:

.. code-block:: ini

   [<profile-name>]
   aws_access_key_id     = <your-access-key>
   aws_secret_access_key = <your-secret-key>

   [default]
   aws_access_key_id     = <your-access-key>
   aws_secret_access_key = <your-secret-key>

Then create the AWS config file (``%USERPROFILE%\.aws\config`` on Windows,
``~/.aws/config`` on Linux and macOS).  Note that in this file, named
profiles use the ``profile`` prefix, while ``[default]`` does not:

.. code-block:: ini

   [profile <profile-name>]
   region = <region>
   endpoint_url = <endpoint-url>

   [default]
   region = <region>

``endpoint_url`` is only needed for S3-compatible object stores that are not
hosted on AWS; omit it for buckets on AWS S3.

.. warning::

   The credentials file contains secrets.  Never commit it, or the keys in
   it, to version control.


Verify the configuration
^^^^^^^^^^^^^^^^^^^^^^^^

Check that the AWS CLI picks up your configuration.  If you use pixi, run the
CLI through your pixi environment:

.. code-block:: powershell

   pixi run aws configure list

Then verify that you can list the bucket with the matching profile:

.. code-block:: powershell

   pixi run aws s3 ls s3://<bucket-name> --profile <profile-name>


Use the profile in a data catalog
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Select the profile with the ``profile`` option of the ``filesystem:`` block
and set ``anon: false`` so that credentials are used:

.. code-block:: yaml

   my_dataset:
     data_type: RasterDataset
     uri: s3://<bucket-name>/<path-to-data>/<file>.nc
     driver:
       name: raster_xarray
       filesystem:
         protocol: s3
         anon: false
         profile: <profile-name>

Alternatively, leave out ``profile`` and select the profile with the
``AWS_PROFILE`` environment variable.  If neither is set, the ``[default]``
profile is used.

.. code-block:: powershell

   $env:AWS_PROFILE = "<profile-name>"

.. tip::

   If you get a ``NoCredentialsError`` or a ``403`` error, run
   ``pixi run aws configure list`` to check that the profile is found, and
   check that the bucket name and profile in your catalog match the commands
   you used to verify access above.


.. _cloud_azure:

Azure Blob Storage
------------------

Public Azure containers and containers that authenticate through environment
variables can be read with the generic approach above.  For everything
Azure-specific — SAS tokens, HTTPS blob URLs, AzureML datastore URIs, the
Azure credential chain, and signed HTTPS URLs for rasterio / GDAL — HydroMT
provides a dedicated resolver.  See :doc:`azure_blob_storage` for details,
including :ref:`choosing_resolver` and a step-by-step guide to
:ref:`azure_sas_quickstart`.
