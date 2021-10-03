What's new
==========
All notable changes to this project will be documented in this page.

The format is based on `Keep a Changelog`_, and this project adheres to
`Semantic Versioning`_.

[Unreleased]
------------

New
^^^
- log hydromt_data.yml with write_datata_catalog (needs to be implemented in various plugins)

Fixed
^^^^^
- Fix bug in io.open_vector and io.open_vector_from_table with WindowsPath fn
- Fix data_libs usage from [global] section of config in cli/main.py

improved
^^^^^^^^
- generalize DataCatalog artifact kwargs to allow for multiple yml files from artifacts

Deprecated
^^^^^^^^^^
- data_adapter.parse_data_sources method deprecated



v0.4.2 (28 July 2021)
---------------------
Noticeable changes include new import of model plugins and improvements of reading methods for tile index and geodataset.

Added
^^^^^

- Small patch for geoms/bbox regions when upscaling flow dir.
- Mask option in merge.merge method for improved open_raster_from_tindex.

Changed
^^^^^^^

- New import of model plugins. Before plugins were only loaded when import MODELS or xxxModel from hydromt.models and not when importing hydromt as before.
- Dropped dask version pins
- read-only check in write_config; dropped write_results
- results objects of Model API can also contain xarray.Dataset. To split a Dataset into DataArrays use the split_dataset option of set_results.

Deprecated
^^^^^^^^^^

- Importing model plugins via "hydromt import xxxModel" or "import hydromt.xxxModel" will be deprecated. Instead use "from hydromt.models import xxxModel" 
  or "from hydromt_xxx import xxxModel".

Fixed
^^^^^

- Fix error when deriving basin mask for subbasin with multiple xy.
- Fix passing timeseries and crs for get_geodataset with vector driver

v0.4.1 (18 May 2021)
--------------------
Noticeable changes are a new CLI region option based on ``grid``.

Added
^^^^^

- New REGION option of the **build** CLI methods for model region based on a ``grid``.
- Keep track of the hydroMT plugin versions in the logging and ``--models`` CLI flag.
- deltares_data and artifact_data options in DataCatalog class and Model API

Changed
^^^^^^^

- Changed the **data-artifacts** version to **v0.0.4**. This includes renaming from hydrom_merit to merit_hydro.
- moved binder to seperate folder with postBuild script
- Bump Black version (formatting).

Fixed
^^^^^

- Multiple ``--opt`` arguments from CLI are now taken into account (instead of only the first).
- Bugfix for crs without an EPSG code.
- Bugfix for Path type path in DataCatalog
- Bugfix missing rasterio in gis_utils.write_map() method
- Bugfix handling of fn_ts in DataCatalog.get_geodataset() method

Documentation
^^^^^^^^^^^^^

- Now **latest** and **stable** versions.
- Added **read_raster_data** notebooks to the examples.

v0.4.0 (23 April 2021)
----------------------
This is the first stable release of hydroMT. Noticeable changes are the addition of the ``deltares-data`` flag, improvements with basin masking functionnalities, and the creation of examples notebooks available 
in the documentation and in Binder.

Added
^^^^^

- Support the use of data stored at `Deltares`_ by introducing the ``--deltares-data`` flag to the CLI and according property to the ``DataCatalog`` and ``Model API``.
- Added ``outlet_map`` and ``stream_map`` functions in flw.py.
- Added ``mask`` function to raster.py for ``RasterDataArray`` and ``RasterDataset`` class.
- Binder environment to run examples notebooks.

Changed
^^^^^^^

- Bump pyflwdir version and dependencies to dask, gdal, numba and netcdf.
- Basin mask functions have been moved from **models/region.py** to **workflows/basin_mask.py**.
- In ``flwdir_from_da`` (flw.py), the **mask** argument can now be a xr.DataArray and not just a boolean. The default behavior has been changed from True to None. This impacts previous use of the function.
- In ``get_basin_geometry`` (workflows/basin_mask.py), basins geometry data are passed via **basin_index** argument instead of **gdf_bas**. GeoDataFrameAdapter are supported as well as geopandas.GeoDataFrame.

Deprecated
^^^^^^^^^^

- The ``build-base`` CLI flag is deprecated since the ini file is now fully in control of each model compoenents to run.

Fixed
^^^^^

- CLI method ``clip``.
- Basin delineation using basin ID (basid).
- Fixed the ``set_config`` and ``get_config`` methods of the model API in order to always try first to read available config file before editing.

Documentation
^^^^^^^^^^^^^

- Documentation moved to GitHub Pages.
- Notebooks examples are added in the documentation.
- Added **delineate_basin** notebooks to the examples.
- Workflows documented in the API docs.
- Update installation instructions.

Tests
^^^^^

- Added unit tests for **workflows/basin_mask.py**.

v0.3.9 (16 April 2021)
----------------------
Initial open source pre-release of hydroMT.


.. _Keep a Changelog: https://keepachangelog.com/en/1.0.0/
.. _Semantic Versioning: https://semver.org/spec/v2.0.0.html
.. _Deltares: https://www.deltares.nl/en/