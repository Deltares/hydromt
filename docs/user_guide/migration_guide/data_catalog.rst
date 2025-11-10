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
- **driver**: ``filesystem`` or ``driver_kwargs`` moved under ``driver``. ``driver`` can
  be a single string or a dictionnary with name and options (passed to underlying function
  that will read the data, e.g. xarray.open_mfdataset, etc.).
- **data_adapter**:``unit_add``, ``unit_mult``, ``rename``, etc. moved under ``data_adapter``
- **uri_resolver**: can be specified mostly in the case of tiled rasters to pass required options.
- **metadata**: ``crs`` and ``nodata`` are moved under ``metadata`` (renamed from ``meta``)
- A single catalog entry can now reference multiple data variants or versions

See more information about the current format in the :ref:`data catalog documentation <get_data>`.

How to upgrade
--------------
All existing pre-defined catalogs have been updated to the new format. For your own catalogs, you can upgrade
easily with the HydroMT ``check`` command:

.. code-block:: bash

   hydromt check -d /path/to/data_catalog.yml --format v0 --upgrade -v
