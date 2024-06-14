==========
What's new
==========
All notable changes to this project will be documented in this page.

The format is based on `Keep a Changelog`_, and this project adheres to
`Semantic Versioning`_.

Unreleased
==========

New
---

Changed
-------

Fixed
-----

Deprecated
----------

v0.10.0 (2024-06-14)
====================

New
---
- New `PredefinedCatalog` class to handle predefined catalog version based on pooch registry files. (#849)


Changed
-------
- Development environment is now set up via pixi instead of mamba / conda. See the documentation for more information on how to install.
- Use the native data CRS when determining zoom levels over the data catalog crs. (#851)
- Improved `flw.d8_from_dem` method with different options to use river vector data to aid the flow direction derivation. (#305)
- DataCatalog.predefined_catalogs retrieves predefined_catalogs specified in predefined_catalogs.py. There is no need for setting the predefined_catalogs anymore. (#844)


Fixed
-----
- Bug in `raster.transform` with lazy coordinates. (#801)
- Bug in `workflows.mesh.mesh2d_from_rasterdataset` with multi-dimensional coordinates. (#843)
- Bug in `MeshModel.get_mesh` after xugrid update to 0.9.0. (#848)
- Bug in `raster.clip_bbox` when bbox doesn't overlap with raster. (#860)
- Allow for string format in zoom_level path, e.g. `{zoom_level:02d}` (#851)
- Fixed incorrect renaming of single variable raster datasets (#883)
- Provide better error message for 0D geometry arrays in GeoDataset (#885)
- Fixed index error when the number of peaks varies between stations in get_hydrographs method (#933)

Deprecated
----------
- The `DataCatalog.from_archive` method is deprecated. Use `DataCatalog.from_yml` with the root pointing to the archive instead. (#849)

v0.9.4 (2024-02-26)
===================
This release fixes a performance regression when reading geometry masks, relaxed warnings for empty raster datasets and updated the documention of the new hydromt commands.

Fixed
-----
- Added back geometry mask when reading vector files with `fiona` as engine/ driver. (#777)
- Relaxed empty data checking on `RasterDatasetAdapter`. (#782)
- Add documentation for `hydromt check` and `hydromt export` commands. (#767)

v0.9.3 (2024-02-08)
===================
This release fixes several bugs. Most notably the `NoDataSrategy` is available in much more data reading methods so plugins can use it more directly. Additionally there are some bug fixes relating to reading shapefiles and reading COGs.

Added
-----
- Test script for testing predefined catalogs locally. (#735)
- Option to write a data catalog to a csv file (#425)

Fixed
-----
- Reading Vector formats that consist of more than one file via geopandas. (#691)
- Handle NoDataStrategy consistently when reading data in adapters (#738)
- add option to ignore empty data sets when exporting data (#743)
- Fix bug in `raster._check_dimensions` for datasets with multiple variables with varying dimension size (#761)
- Fix bug when reading COGs at requested zoom level (#758)

v0.9.2 (2024-01-09)
===================
This release adds additional bug fixes for the meridian offset functinality, and improvements to the new CLI commands.

Added
-----
- Export CLI now also accepts time tuples (#660)
- New stats.skills VE and RSR (#666)
- Check CLI command can now validate bbox and geom regions (#664)

Changed
-------
- Export CLI now uses '-s' for source, '-t' for time and '-i' for config. (#660)

Fixed
-----
- Double reading of model components when in appending mode. (#695)
- Removed deprecated entrypoints library. (#676)
- Bug in `raster.set_crs` if input_crs is of type CRS. (#659)
- Export CLI now actually parses provided geoms. (#660)
- Bug in stats.skills for computation of pbias and MSE / RMSE. (#666)
- `Model.write_geoms` ow has an option to write GeoJSON coordinates in WGS84 if specified (#510)
- Fix bug with lazy spatial_ref coordinate (#682)
- Bug in gis_utils.meridian_offset. (#692)


v0.9.1 (2023-11-16)
===================
This release contains several bugfixes to 0.9.0 as well two new CLI methods and support for STAC.

Added
-----
- Support for exporting data catalogs to STAC catalog formats. (#617)
- Support for reading data catalogs from STAC catalog formats. (#625)
- Pixi is now available as an additional task runner. (#634)
- Support exporting data from catalogs from the CLI. (#627)
- Support for validating data catalogs from the CLI. (#632)
- Support for validating model configurations from the CLI. (#643)


Changed
-------
- `DataAdapter._slice_data` and `DataCatalog.get_<data type>` now have a `handle_nodata` argument.

Fixed
-----
- Bug in zoom level detection in `RasterDatasetAdapter` for Tifs without overviews and paths with placeholders. (#642)
- Bug in gis_utils.meridian_offset for grids with rounding errors. (#649)

v0.9.0 (2023-10-19)
===================
This release contains several new features, here we highlight a few:
- Support in the DataCatalog for data sources from different providers or versions with better support for cloud and http data.
- Developers documentation to start your own plugin and accompanying template.
- Support multigrids in meshmodel (with example) and improved implementation VectorModel (was LumpedModel)
- Support for reading overviews (zoom levels) of Cloud Optimized GeoTIFFs (COGs).

Added
-----

Documentation
^^^^^^^^^^^^^
- docs now include a dropdown for selecting older versions of the docs. (PR #457)
- docs include a new example for MeshModel. (PR #595)
- Added documentation for how to start your own plugin (PR #446)

Data
^^^^
- Support for loading the same data source but from different providers (e.g., local & aws) and versions  (PR #438)
- Add support for reading and writing tabular data in ``parquet`` format. (PR #445)
- Add support for reading model configs in ``TOML`` format. (PR #444)
- add ``open_mfcsv`` function in ``io`` module for combining multiple CSV files into one dataset. (PR #486)
- Adapters can now clip data that is passed through a python object the same way as through the data catalog. (PR #481)
- Relevant data adapters now have functionality for reporting and detecting the spatial and temporal extent they cover (PR #503)
- Data catalogs have a ``hydromt_version`` meta key that is used to determine compatibility between the catalog and the installed hydromt version. (PR #506)
- Allow the root of a data catalog to point to an archive, this will be extracted to the ~/.hydromt_data folder. (PR #512)
- Support for reading overviews from (Cloud Optimized) GeoTIFFs using the zoom_level argument of ``DataCatalog.get_rasterdataset``. (PR #514)
- Support for http and other *filesystems* in path of data source (PR #515).

Model
^^^^^
- new ``force-overwrite`` option in ``hydromt update`` CLI to force overwritting updated netcdf files. (PR #460)
- Model objects now have a _MODEL_VERSION attribute that plugins can use for compatibility purposes (PR # 495)
- ``set_forcing`` can now add pandas.DataFrame object to forcing. (PR #534)

Raster
^^^^^^
- Model class now has methods for getting, setting, reading and writing arbitrary tabular data. (PR #502)
- Support for writing overviews to (Cloud Optimized) GeoTIFFs in the ``raster.to_raster`` method. (PR #514)
- New raster method ``to_slippy_tiles``: tiling of a raster dataset according to the slippy tile structure for e.g., webviewers (PR #440).

Changed
-------

Model
^^^^^
- Updated ``MeshModel`` and related methods to support multigrids instead of one single 2D grid. (PR #412)
- Renamed ``LumpedModel.response_units`` to ``VectorModel.vector`` and updated the base set, read, write methods. (#531)
- possibility to ``load`` the data in the model read\_ functions for netcdf files (default for read_grid in r+ mode). (PR #460)
- Internal model components (e.g. `Models._maps`, `GridModel._grid``) are now initialized with None and should not be accessed directly,
  call the corresponding model property  (e.g. `Model.maps`, `GridModel.grid`) instead. (PR #473)
- ``setup_mesh2d_from_rasterdataset`` and ``setup_mesh2d_from_raster_reclass`` now use xugrid Regridder methods. (PR #535)
- Use the Model.data_catalog to read the model region if defined by a geom or grid. (PR #479)

Vector
^^^^^^
- ``vector.GeoDataset.from_gdf`` can use the gdf columns as data_vars instead of external xarray. (PR #412)
- ``io.open_vector`` now can use `pyogrio` if reading from a non-tabular dataset (PR #583)

Fixed
-----
- when a model component (eg maps, forcing, grid) is updated using the set\_ methods, it will first be read to avoid loosing data. (PR #460)
- open_geodataset with driver vector also works for other geometry type than points. (PR #509)
- overwrite model in update mode. (PR #534)
- fix stats.extremes methods for (dask) 3D arrays. (PR #505)
- raster gives better error on incompatible nodata (PR #544)

Deprecated
----------
- the dependencies ``pcraster`` and ``pygeos`` are no longer used and were removed. (PR #467)


v0.8.0 (2023-07-18)
===================
This release contains several new features, including extreme value analysis, new generic methods for the ``GridModel`` class, setting variable attributes like units through the data catalog, and the ability to detect compatability issues between Datacatalog and HydroMT versions. It also includes a minor breaking change since now geometry masks are only set if the `mask` in `raster.clip_geom` is set to `True` to improve memory usage.


Added
-----
- Support for unit attributes for all data types in the DataCatalog. PR #334
- Data catalog can now handle specification of HydroMT version
- New generic methods for ``GridModel``: ``setup_grid``, ``setup_grid_from_constant``, ``setup_grid_from_rasterdataset``, ``setup_grid_from_raster_reclass``, ``setup_grid_from_geodataframe``. PR #333
- New ``grid`` workflow methods to support the setup methods in ``GridModel``: ``grid_from_constant``, ``grid_from_rasterdataset``, ``grid_from_raster_reclass``, ``grid_from_geodataframe``. PR #333
- New raster method ``rasterize_geometry``.
- New extreme valua analysis and design event (creation hydrographs) methods in stats submodule.
  Note that these methods are experimental and may be moved elsewhere / change in signature. PR #85

Changed
-------
- Arguments to drivers in data catalog files and the `DataCatalog.get_` methods should now explicitly be called driver_kwargs instead of kwargs. PR #334
- New geom_type argument in `RasterDataArray.vector_grid` to specify the geometry type {'polygon', 'line', 'points'} of the vector grid. PR #351
- Added extrapolate option to `raster.interpolate_na` method. PR #348
- Name of methods ``setup_maps_from_raster`` and ``setup_mesh_from_raster`` to ``setup_maps_from_rasterdataset`` and ``setup_mesh_from_rasterdataset``. PR #333
- Add rename argument to ``setup_*_from_rasterdataset``, ``setup_*_from_raster_reclass`` to maps and mesh for consistency with grid. PR #333
- Introduced different merge options in `GeoDataset.from_gdf` and `GeoDataFrame.from_gdf`. PR #441
- ``DataCatalog.get_rasterdataset`` always uses bbox to clip raster data. PR #434
- ``raster.clip_geom`` only set a geometry mask if the mask argument is true to avoid memory issues. PR #434
- ``raster.clip_mask`` interface and behavior changed to be consistent with ``raster.clip_geom``. PR #318

Fixed
-----
- Order of renaming variables in ``DataCatalog.get_rasterdataset`` for x,y dimensions. PR #324
- fix bug in ``get_basin_geometry`` for region kind 'subbasin' if no stream or outlet option is specified.
- fix use of Path objects in ``DataCatalog.from_dict``. PR #429
- ``raster.reproject_like`` first clips the data to the target extent before reprojecting. PR #434


v0.7.1 (14 April 2023)
======================

This release contains several small updates of the code.
Most prominently is the support for yml configuration files.

Added
-----
- Support for in-memory data like objects instead of source name or path in DataCatalog().get* methods. PR #313
- Support for yaml configuration files. The support for ini files will be deprecated in the future. PR #292
- Option to export individual variables from a data source and append to an existing data catalog in DataCatalog.export_data. PR #302


v0.7.0 (22 February 2023)
=========================

This release contains several major updates of the code. These following updates might require small changes to your code:

- Most noticeable is the change in the ``hydromt build`` CLI, where made the region argument optional and deprecated the resolution option. Futhermore, the user has to force existing folders to be overwritten when building new models.
- We also did a major overhaul of the ``GeoDataset`` and the associated ``.vector`` assessor to support any type of vector geometries (before only points).

More new features, including support for rotated grids, new cloud data catalogs and (caching of) tiled raster datasets and more details are listed below.


Changed
-------
- Removed resolution ('-r', '--res') from the hydromt build cli, made region (now '-r') an optional argument. PR #278
- If the model root already contains files when setting root, this will cause an error unless force overwrite (mode='w+' or --fo/--force-overwrite from command line). PR #278
- Revamped the GeoDataset (vector.py) to now work with geometry objects and wkt strings besides xy coordinates. PR #276
- GeoDataset can write to .nc that is compliant with ogr. PR #208
- Support for rotated grids in RasterDataset/Array, with new rotation and origin properties. PR #272
- Removed pygeos as an optional dependency, hydromt now relies entirely on shapely 2.0 PR #258
- Changed shapely to require version '2.0.0' or later. PR #228
- strict and consistent read/write mode policy PR #238
- do not automatically read hydromt_data.yml file in model root. PR #238
- RasterDataset zarr driver: possibility to read from several zarr stores. The datasets are then merged and ``preprocess`` can
  be applied similar to netcdf driver. PR #249

Added
-----
- New methods to compute PET in workflows.forcing.pet using Penman Monteith FAO-56 based on the `pyet` module. Available arguments are now method = ['debruin', 'makkink', 'penman-monteith_rh_simple', 'penman-monteith_tdew'] PR #266
- New get_region method in cli/api.py that returns a geojson representation of the parsed region. PR #209
- write raster (DataArray) to tiles in xyz structure with the RasterDataArray.to_xyz_tiles method. PR #262
- add zoom_level to DataCatalog.get_rasterdataset method. PR #262
- new write_vrt function in gis_utils to write '.vrt' using GDAL. PR #262
- new predefined catalog for cmip6 data stored on Google Cloud Storage ``cmip6_data``. Requires dependency gcsfs. PR #250
- new predefined catalog for public data stored on Amazon Web Services ``aws_data``. Requires dependency s3fs. PR #250
- new DataCatalog preprocess function ``harmonise_dims`` for manipulation and harmonization of array dimensions. PR #250
- experimental: support for remote data with a new yml data source ``filesystem`` attribute. Supported filesystems are [local, gcs, s3].
  Profile information can be passed in the data catalog ``kwargs`` under **storage_options**. PR #250
- experimental: new caching option for tiled rasterdatasets ('--cache' from command line). PR #286

Fixed
-----
- bug related to opening named raster files. PR #262
- All CRS objects are from pyproj library (instead of rasterio.crs submodule). PR #230
- fix reading lists and none with config. PR #246
- fix `DataCatalog.to_yml` and `DataCatalog.export()` with relative path and add meta section. PR #238

Deprecated
----------
- `x_dim`, `y_dim`, and `total_bounds` attributes of GeoDataset/GeoDataArray are renamed to `x_name`, `y_name` and `bounds`. PR #276
- Move pygeos to optional dependencies in favor of shapely 2.0. PR #228
- Resolution option in hydromt build cli. PR #278

Documentation
-------------
- Added **Working with GeoDatasets** python notebook. PR #276
- added **working_with_models** example notebook. PR #229
- added **export_data** example notebook. PR #222
- added **reading_point_data** example notebook. PR #216
- added **working_with_flow_directions** example notebook. PR #231
- added **prep_data_catalog** example notebook. PR #232
- added **reading_tabular_data** example notebook. PR #216


v0.6.0 (24 October 2022)
========================

In this release, we updated the ``Model API``  by renaming staticgeoms to geoms, adding a new maps object and removing abstract methods.
We also added new general subclasses to Model: ``GridModel``, ``LumpedModel``, ``MeshModel``, ``NetworkModel``.
These new subclasses have their own objects (e.g. grid for GridModel representing regular grids which replaces the old staticmaps object).
More details in the list below:

Added
-----
- ModelCatalog to discover generic and plugin model classes. `PR #202 <https://github.com/Deltares/hydromt/pull/202>`_
- Support for 2-dimensional tabular data through the new DataFrameAdapter. `PR #153 <https://github.com/Deltares/hydromt/pull/153>`_
- API calls to get info about model components and dataset for the dashboard. `PR #118 <https://github.com/Deltares/hydromt/pull/118>`_
- New submodel classes in hydromt: ``GridModel``, ``LumpedModel``, ``MeshModel``, ``NetworkModel``
- Added entrypoints for lumped_model, mesh_model, grid_model
- New mixin classes created for model specific object: ``GridMixin`` for self.grid, ``LumpedMixin`` for self.response_units, ``MeshMixin`` for self.mesh,
  ``MapsMixin`` for self.maps
- New high-level object: self.maps for storing regular rasters data (which can have resolution and / or projection).
- Maps generic setup methods: ``MapsMixin.setup_maps_from_raster`` and ``MapsMixin.setup_maps_from_rastermapping``
- Mesh generic setup methods: ``MeshModel.setup_mesh``, ``MeshMixin.setup_maps_from_raster`` and ``MeshMixin.setup_maps_from_rastermapping``

Changed
-------
- self.staticgeoms object and methods renamed to self.geoms
- self.staticmaps object and methods renamed to self.grid and moved into GridModel and GridMixin

Fixed
-----
- Bug in backward compatibility of staticgeoms (not read automatically). `Issue #190 <https://github.com/Deltares/hydromt/issues/190>`_
- Direct import of xarray.core.resample. `Issue #189 <https://github.com/Deltares/hydromt/issues/189>`_
- Bug in dim0 attribute of raster, removed instead of set to None if no dim0 `Issue #210 <https://github.com/Deltares/hydromt/issues/210>`_

Deprecated
----------
- self.staticgeoms and self.staticmaps are deprecated.

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
