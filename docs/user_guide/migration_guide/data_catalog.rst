.. _data_catalog_migration:

Migrating the Data Catalog
==========================

Overview
--------
The data catalog structure has been refactored to introduce a more modular design and
clearer separation of responsibilities across several new classes (``DataSource``, ``Driver``, ``URIResolver``, and ``DataAdapter``):

- ``URIResolver`` is in charge of parsing the path or URI of the file (e.g if you are using some keywords like
  ``{year}`` or ``{month}`` in your paths or if you want to read tiled raster)
- ``Driver`` is in charge of reading the data from the source (e.g reading a netcdf file from a local disk or from cloud)
- ``DataAdapter`` is in charge of harmonizing the data to standard HydroMT data structures (e.g. renaming variables, setting attributes, units conversion, etc.)
- ``DataSource`` is the main class that ties everything together and is used by the ``DataCatalog`` to load data.

Key format changes:

- ``path`` is renamed to ``uri``
- **driver**: ``driver_kwargs`` moved under ``driver``. ``driver`` can
  be a single string or a dictionnary with name and options (passed to underlying function
  that will read the data, e.g. xarray.open_mfdataset, etc.).
- **filesystem**: moved to the top level of each data source (see :ref:`catalog format v2 <data_catalog_format_v2>`).
- **data_adapter**:``unit_add``, ``unit_mult``, ``rename``, etc. moved under ``data_adapter``
- **uri_resolver**: can be specified mostly in the case of tiled rasters to pass required options.
- **metadata**: ``crs`` and ``nodata`` are moved under ``metadata`` (renamed from ``meta``)
- A single catalog entry can now reference multiple data variants or versions

See more information about the current format in the :ref:`data catalog documentation <get_data>`.

.. _data_catalog_format_v2:

Catalog format v2
-----------------
In HydroMT 1.5, filesystem configuration moved from under ``driver`` or
``uri_resolver`` to the top level of each data source. For example, move
``driver.filesystem`` to ``filesystem`` alongside ``uri`` and ``driver``, preserving
its options. This is a breaking catalog format change: HydroMT versions before 1.5
cannot read catalogs using the new layout.

For predefined catalogs, v2.0.0 versions are available for ``aws_data``,
``earthdatahub_data`` and ``gcs_cmip6_data``. Earlier catalog versions remain
available for older HydroMT versions.

The catalog format is determined by the ``meta.format_version`` key of the catalog
(``format_version: 2``); catalogs without this key are format v1. The ``meta.version``
key is only the version of the catalog content. Each HydroMT release supports a range
of formats:

- A catalog with a **newer** format than HydroMT supports raises an error; upgrade
  HydroMT or use an older version of the catalog.
- A catalog with an **older but still supported** format (v1) is read, with a
  deprecation warning, and upgraded in memory to the latest format.
- A catalog with a **legacy** format (v0) cannot be read and raises an error with the
  upgrade instructions below.

The ``meta.hydromt_version`` key is deprecated and ignored.

How to upgrade
--------------
All existing pre-defined catalogs have been updated to the new format. For your own catalogs, use the
HydroMT ``check`` command to upgrade the catalog layout. Use ``--format v0`` for catalogs in the
legacy v0 format, or ``--format v1`` for v1 catalogs that still define ``filesystem`` under
``driver`` or ``uri_resolver``. The upgraded catalog is written next to the original and its
``meta.format_version`` is set to the latest format.

.. code-block:: bash

   hydromt check -d /path/to/data_catalog.yml --format v0 --upgrade -v
   hydromt check -d /path/to/data_catalog.yml --format v1 --upgrade -v
