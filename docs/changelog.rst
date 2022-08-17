==========
What's new
==========
All notable changes to this project will be documented in this page.

The format is based on `Keep a Changelog`_, and this project adheres to
`Semantic Versioning`_.

unreleased
==========

Added
-----
- Support for 2-dimensional tabular data through the new DataFrameAdapter. `PR #153 <https://github.com/Deltares/hydromt/pull/153>`_

Changed
-------

Fixed
-----

Deprecated
----------


v0.5.0 (4 August 2022)
======================

Added
-----
- New raster method for adding gdal_compliant() attributes to xarray object.
- Function ``to_datetimeindex`` in available preprocess functions for xr.open_dataset in the data adapter.
- Function ``remove_duplicates`` in available preprocess functions for xr.open_dataset in the data adapter.
- New ``DataCatalog.from_predefined_catalogs`` and ``DataCatalog.from_archive`` to support predefined data catalogs and archive
  in a generic way through the data/predefined_catalogs.yml file.
- Optional formatting for year and month variables in path of data sources.

Changed
-------
- splitted data_adapter.py into a  data_catalog and data_adapter submodule with py scripts per adapter
- Add rioxarray dependency to read raster data
- In build or update methods, the setup_config component is not forced to run first anymore but according to order of the components in the ini config (opt dict).
- In DataCatalog.get_RasterDataset & DataCatalog.get_GeoDataset methods, variables can now also be a str as well as a list of strings.
- In DataCatalog.get_RasterDataset & DataCatalog.get_GeoDataset methods, automatic renaming of single variable datasets based on the variables argument will be deprecated
- Interpolate missing values based on D4 neighbors of missing value cells only. This largely improves the performance without loosing accuracy.
  Changes have been observed when `nearest` method is used but this should not impact quality of the interpolation.
- New source_names argument to DataCatalog.to_yml

Fixed
-----
- Fixed DataAdapter.resolve_paths with unknown keys #121
- Fixed the WGS84 datum in the gis_utils.utm_crs method.
- In merge.merge the grid is now aligned with input dataset with the largest overlap if no dst_bounds & dst_res are given.
- Fixed the predicate not being passed in get_geodataframe method.
- Removed deprecated xr.ufuncs calls.

Deprecated
----------
- Automatic renaming of single var dataset if variables is provided in get_rasterdataset. Data catalog should be used instead.
- ``DataCatalog.from_artifacts``. Use ``DataCatalog.from_predefined_catalogs`` instead.

v0.4.5 (16 February 2022)
=========================

Added
-----
- New skill scores: KGE 2012, KGE non-parametric (2018), KGE non-parametric flood (2018).
- new rasterio inverse distance weighting method ("rio_idw") in raster.interpolate_na
- Add option to add placeholders in yml file to explode a single yml entry to multiple yml entries (useful for e.g. climate datasets).
- general Model.setup_region method

Changed
-------
- stats.py is now in stats/skills.py in order to include more and different type of new statistics later.
- improved flw.reproject_hydrography_like and flw.dem_adjust methods
- file handlers of loggers are replaced in Model.set_root
- log.setuplog replaces old handlers if these exist to avoid duplicates.
- setup_basemaps method no longer required for build method
- improved interbasin regions in workflows.get_basin_geometry
- drop non-serializable entries from yml file when writing data catalog to avoid it getting corrupt
- data catalog yml entries get priority over local files or folders with the same name in the data_adapter.get_* methods
  multi-file rasterdatasets are only supported through the data catalog yml file 

Fixed
-----
- fix incorrect nodata values at valid cells from scipy.griddata method in raster.interpolate_na

Deprecated
----------
- workflows.basemaps methods (hydrography and topography) moved to hydromt_wflow

v0.4.4 (19 November 2021)
=========================

Added
-----
- flw.d8_from_dem to derive a flow direction raster from a DEM
- flw.reproject_hydrography_like to reproject flow direction raster data
- flw.floodplain_elevation method which returns floodplain classification and hydrologically adjusted elevation
- raster.flipud method to flip data along y-axis
- raster.area_grid to get the raster cell areas [m2]
- raster.density_grid to convert the values to [unit/m2]
- gis_utils.spread2d method (wrapping its pyflwdir equivalent) to spread values on a raster
- gis_utils.nearest and gis_utils.nearest_merge methods to merge GeoDataFrame based on proximity
- river_width to estimate a segment average river width based on a river mask raster 
- river_depth to get segment average river depth estimates based bankfull discharge (requires pyflwdir v0.5.2)

Changed
-------
- bumped hydromt-artifacts version to v0.0.6
- In model API build and update functions, if any write* are called in the ini file (opt), 
  the final self.write() call is skipped. This enables passing custom arguments to the write* 
  functions without double writing files or customizing the order in which write* functions 
  are called. If any write* function is called we assume the user manages the writing and
  a the global write method is skipped.
- default GTiff lwz compression with DataCatalog.export_data method
- rename DataAdapter.export_data to DataAdapter.to_file to avoid confusion with DataCatalog.export_data method
- allow "alias" with attributes in DataCatalog yml files / dictionaries

Fixed
-----
- DataCatalog.to_yml Path objects written as normal strings 
- Bugfix in basin_mask.get_basin_geometry when using bbox or geom arguments
- Bugfix DataAdapter.__init__ setting None value in meta data
- Bugfix DataAdapter.resolve_paths with argument in root

Deprecated
----------
- flw.gaugemap is replaced by flw.gauge_map for a more consistent interface of flw.*map methods
- flw.basin_shape is redundant

v0.4.3 (3 October 2021)
=======================

Added
-----
- log hydromt_data.yml with write_data_catalog (needs to be implemented in various plugins)
- add alias option in data catalog yml files
- use mamba for github actions 

Changed
-------
- generalize DataCatalog artifact kwargs to allow for multiple yml files from artifacts
- keep geom attributes with <Dataset/DataArray>.vector.to_gdf method

Fixed
-----
- Fix bug in io.open_vector and io.open_vector_from_table with WindowsPath fn
- Fix data_libs usage from [global] section of config in cli/main.py
- Bugfix sampling for rasters with 'mask' coordinate
- Bugfix logical operator in merge method

Deprecated
----------
- data_adapter.parse_data_sources method deprecated



v0.4.2 (28 July 2021)
=====================
Noticeable changes include new import of model plugins and improvements of reading methods for tile index and geodataset.

Added
-----

- Small patch for geoms/bbox regions when upscaling flow dir.
- Mask option in merge.merge method for improved open_raster_from_tindex.

Changed
-------

- New import of model plugins. Before plugins were only loaded when import MODELS or xxxModel from hydromt.models and not when importing hydromt as before.
- Dropped dask version pins
- read-only check in write_config; dropped write_results
- results objects of Model API can also contain xarray.Dataset. To split a Dataset into DataArrays use the split_dataset option of set_results.

Deprecated
----------

- Importing model plugins via "hydromt import xxxModel" or "import hydromt.xxxModel" will be deprecated. Instead use "from hydromt.models import xxxModel" 
  or "from hydromt_xxx import xxxModel".

Fixed
-----

- Fix error when deriving basin mask for subbasin with multiple xy.
- Fix passing timeseries and crs for get_geodataset with vector driver

v0.4.1 (18 May 2021)
====================
Noticeable changes are a new CLI region option based on ``grid``.

Added
-----

- New REGION option of the **build** CLI methods for model region based on a ``grid``.
- Keep track of the hydroMT plugin versions in the logging and ``==models`` CLI flag.
- deltares_data and artifact_data options in DataCatalog class and Model API

Changed
-------

- Changed the **data-artifacts** version to **v0.0.4**. This includes renaming from hydrom_merit to merit_hydro.
- moved binder to seperate folder with postBuild script
- Bump Black version (formatting).

Fixed
-----

- Multiple ``==opt`` arguments from CLI are now taken into account (instead of only the first).
- Bugfix for crs without an EPSG code.
- Bugfix for Path type path in DataCatalog
- Bugfix missing rasterio in gis_utils.write_map() method
- Bugfix handling of fn_ts in DataCatalog.get_geodataset() method

Documentation
-------------

- Now **latest** and **stable** versions.
- Added **read_raster_data** notebooks to the examples.

v0.4.0 (23 April 2021)
======================
This is the first stable release of hydroMT. Noticeable changes are the addition of the ``deltares-data`` flag, improvements with basin masking functionnalities, and the creation of examples notebooks available 
in the documentation and in Binder.

Added
-----

- Support the use of data stored at `Deltares`_ by introducing the ``==deltares-data`` flag to the CLI and according property to the ``DataCatalog`` and ``Model API``.
- Added ``outlet_map`` and ``stream_map`` functions in flw.py.
- Added ``mask`` function to raster.py for ``RasterDataArray`` and ``RasterDataset`` class.
- Binder environment to run examples notebooks.

Changed
-------

- Bump pyflwdir version and dependencies to dask, gdal, numba and netcdf.
- Basin mask functions have been moved from **models/region.py** to **workflows/basin_mask.py**.
- In ``flwdir_from_da`` (flw.py), the **mask** argument can now be a xr.DataArray and not just a boolean. The default behavior has been changed from True to None. This impacts previous use of the function.
- In ``get_basin_geometry`` (workflows/basin_mask.py), basins geometry data are passed via **basin_index** argument instead of **gdf_bas**. GeoDataFrameAdapter are supported as well as geopandas.GeoDataFrame.

Deprecated
----------

- The ``build-base`` CLI flag is deprecated since the ini file is now fully in control of each model compoenents to run.

Fixed
-----

- CLI method ``clip``.
- Basin delineation using basin ID (basid).
- Fixed the ``set_config`` and ``get_config`` methods of the model API in order to always try first to read available config file before editing.

Documentation
-------------

- Documentation moved to GitHub Pages.
- Notebooks examples are added in the documentation.
- Added **delineate_basin** notebooks to the examples.
- Workflows documented in the API docs.
- Update installation instructions.

Tests
-----

- Added unit tests for **workflows/basin_mask.py**.

v0.3.9 (16 April 2021)
======================
Initial open source pre-release of hydroMT.


.. _Keep a Changelog: https://keepachangelog.com/en/1.0.0/
.. _Semantic Versioning: https://semver.org/spec/v2.0.0.html
.. _Deltares: https://www.deltares.nl/en/
