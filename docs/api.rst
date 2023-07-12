.. currentmodule:: hydromt

.. _api_reference:

#############
API reference
#############

====
Data
====

.. _api_data_catalog:

Data catalog
============

General
-------

.. autosummary::
   :toctree: _generated

   data_catalog.DataCatalog
   data_catalog.DataCatalog.sources
   data_catalog.DataCatalog.keys
   data_catalog.DataCatalog.predefined_catalogs
   data_catalog.DataCatalog.to_dict
   data_catalog.DataCatalog.to_dataframe
   data_catalog.DataCatalog.to_yml
   data_catalog.DataCatalog.export_data

Add data sources
----------------

.. autosummary::
   :toctree: _generated

   data_catalog.DataCatalog.set_predefined_catalogs
   data_catalog.DataCatalog.from_predefined_catalogs
   data_catalog.DataCatalog.from_archive
   data_catalog.DataCatalog.from_yml
   data_catalog.DataCatalog.from_dict
   data_catalog.DataCatalog.update

.. _api_data_catalog_get:

Get data
--------

.. autosummary::
   :toctree: _generated

   data_catalog.DataCatalog.get_rasterdataset
   data_catalog.DataCatalog.get_geodataset
   data_catalog.DataCatalog.get_geodataframe



RasterDataset
=============

.. autosummary::
   :toctree: _generated

   data_adapter.RasterDatasetAdapter
   data_adapter.RasterDatasetAdapter.summary
   data_adapter.RasterDatasetAdapter.get_data
   data_adapter.RasterDatasetAdapter.to_dict
   data_adapter.RasterDatasetAdapter.to_file

GeoDataset
==========

.. autosummary::
   :toctree: _generated

   data_adapter.GeoDatasetAdapter
   data_adapter.GeoDatasetAdapter.summary
   data_adapter.GeoDatasetAdapter.get_data
   data_adapter.GeoDatasetAdapter.to_dict
   data_adapter.GeoDatasetAdapter.to_file

GeoDataFrame
============

.. autosummary::
   :toctree: _generated

   data_adapter.GeoDataFrameAdapter
   data_adapter.GeoDataFrameAdapter.summary
   data_adapter.GeoDataFrameAdapter.get_data
   data_adapter.GeoDataFrameAdapter.to_dict
   data_adapter.GeoDataFrameAdapter.to_file

DataFrame
=========

.. autosummary::
   :toctree: _generated

   data_adapter.DataFrameAdapter
   data_adapter.DataFrameAdapter.summary
   data_adapter.DataFrameAdapter.get_data
   data_adapter.DataFrameAdapter.to_dict
   data_adapter.DataFrameAdapter.to_file


======
Models
======

Discovery
=========

.. autosummary::
   :toctree: _generated

   ModelCatalog


.. _model_api:

Model
=====

Note that the base Model attributes and methods are available to all models.

.. autosummary::
   :toctree: _generated

   Model

High level methods
------------------

.. autosummary::
   :toctree: _generated

   Model.read
   Model.write
   Model.build
   Model.update
   Model.set_root
   Model.write_data_catalog

Model attributes
----------------

.. autosummary::
   :toctree: _generated

   Model.crs
   Model.region
   Model.api


Model components and attributes
-------------------------------

.. autosummary::
   :toctree: _generated

   Model.config
   Model.maps
   Model.geoms
   Model.forcing
   Model.states
   Model.results

General methods
---------------

.. autosummary::
   :toctree: _generated

   Model.get_config
   Model.set_config
   Model.read_config
   Model.write_config

   Model.set_maps
   Model.read_maps
   Model.write_maps

   Model.set_geoms
   Model.read_geoms
   Model.write_geoms

   Model.set_forcing
   Model.read_forcing
   Model.write_forcing

   Model.set_states
   Model.read_states
   Model.write_states

   Model.set_results
   Model.read_results

.. _setup_methods:

Setup methods
-------------

.. autosummary::
   :toctree: _generated

   Model.setup_config
   Model.setup_region
   Model.setup_maps_from_rasterdataset
   Model.setup_maps_from_raster_reclass


.. _grid_model_api:

GridModel
=========

.. autosummary::
   :toctree: _generated

   GridModel


Components and attributes
-------------------------

.. autosummary::
   :toctree: _generated

   GridModel.grid
   GridModel.crs
   GridModel.region


General methods
---------------

.. autosummary::
   :toctree: _generated

   GridModel.set_grid
   GridModel.read_grid
   GridModel.write_grid

Setup methods
-------------

.. autosummary::
   :toctree: _generated

   GridModel.setup_config
   GridModel.setup_region
   GridModel.setup_maps_from_rasterdataset
   GridModel.setup_maps_from_raster_reclass
   GridModel.setup_grid
   GridModel.setup_grid_from_constant
   GridModel.setup_grid_from_rasterdataset
   GridModel.setup_grid_from_raster_reclass
   GridModel.setup_grid_from_geodataframe


.. _lumped_model_api:

LumpedModel
===========

.. autosummary::
   :toctree: _generated

   LumpedModel


Components and attributes
-------------------------

.. autosummary::
   :toctree: _generated

   LumpedModel.response_units
   LumpedModel.crs
   LumpedModel.region

General methods
---------------

.. autosummary::
   :toctree: _generated

   LumpedModel.set_response_units
   LumpedModel.read_response_units
   LumpedModel.write_response_units

Setup methods
-------------

.. autosummary::
   :toctree: _generated

   LumpedModel.setup_config
   LumpedModel.setup_region
   LumpedModel.setup_maps_from_rasterdataset
   LumpedModel.setup_maps_from_raster_reclass


.. _mesh_model_api:

MeshModel
=========

.. autosummary::
   :toctree: _generated

   MeshModel


Components and attributes
-------------------------

.. autosummary::
   :toctree: _generated

   MeshModel.mesh
   MeshModel.crs
   MeshModel.region

General methods
---------------

.. autosummary::
   :toctree: _generated

   MeshModel.set_mesh
   MeshModel.read_mesh
   MeshModel.write_mesh


Setup methods
-------------

.. autosummary::
   :toctree: _generated

   MeshModel.setup_config
   MeshModel.setup_region
   MeshModel.setup_maps_from_rasterdataset
   MeshModel.setup_maps_from_raster_reclass
   MeshModel.setup_mesh_from_rasterdataset
   MeshModel.setup_mesh_from_raster_reclass


=========
Workflows
=========

Grid
====

.. autosummary::
   :toctree: _generated

   workflows.grid.grid_from_constant
   workflows.grid.grid_from_rasterdataset
   workflows.grid.grid_from_raster_reclass
   workflows.grid.grid_from_geodataframe


Basin mask
==========

.. autosummary::
   :toctree: _generated

   workflows.basin_mask.get_basin_geometry
   workflows.basin_mask.parse_region

River bathymetry
================

.. autosummary::
   :toctree: _generated

   workflows.rivers.river_width
   workflows.rivers.river_depth


Forcing
=======

Data handling
-------------

.. autosummary::
   :toctree: _generated

   workflows.forcing.precip
   workflows.forcing.temp
   workflows.forcing.press
   workflows.forcing.pet
   workflows.forcing.wind


Correction methods
------------------

.. autosummary::
   :toctree: _generated

   workflows.forcing.press_correction
   workflows.forcing.temp_correction

Time resampling methods
-----------------------

.. autosummary::
   :toctree: _generated

   workflows.forcing.resample_time
   workflows.forcing.delta_freq

Computation methods
-------------------

**PET**

.. autosummary::
   :toctree: _generated

   workflows.forcing.pet_debruin
   workflows.forcing.pet_makkink
   workflows.forcing.pm_fao56

=======================
Reading/writing methods
=======================

.. _open_methods:

Reading methods
===============

.. autosummary::
   :toctree: _generated

   io.open_raster
   io.open_mfraster
   io.open_raster_from_tindex
   io.open_vector
   io.open_vector_from_table
   io.open_geodataset
   io.open_timeseries_from_table

Raster writing methods
======================

.. autosummary::
   :template: autosummary/accessor_method.rst
   :toctree: _generated

   DataArray.raster.to_xyz_tiles
   DataArray.raster.to_osm_tiles
   DataArray.raster.to_raster
   Dataset.raster.to_mapstack


==============
Raster methods
==============

High level methods
==================

.. autosummary::
   :toctree: _generated

   merge.merge
   raster.full
   raster.full_like
   raster.full_from_transform

.. autosummary::
   :toctree: _generated
   :template: autosummary/accessor_method.rst

   DataArray.raster.from_numpy
   Dataset.raster.from_numpy

Attributes
==========

.. autosummary::
   :toctree: _generated
   :template: autosummary/accessor_attribute.rst

   DataArray.raster.attrs
   DataArray.raster.crs
   DataArray.raster.bounds
   DataArray.raster.transform
   DataArray.raster.res
   DataArray.raster.rotation
   DataArray.raster.origin
   DataArray.raster.nodata
   DataArray.raster.dims
   DataArray.raster.coords
   DataArray.raster.dim0
   DataArray.raster.y_dim
   DataArray.raster.x_dim
   DataArray.raster.xcoords
   DataArray.raster.ycoords
   DataArray.raster.shape
   DataArray.raster.size
   DataArray.raster.width
   DataArray.raster.height
   DataArray.raster.internal_bounds
   DataArray.raster.box


General methods
===============

.. autosummary::
   :template: autosummary/accessor_method.rst
   :toctree: _generated

   DataArray.raster.set_crs
   DataArray.raster.set_spatial_dims
   DataArray.raster.reset_spatial_dims_attrs
   DataArray.raster.identical_grid
   DataArray.raster.aligned_grid
   DataArray.raster.gdal_compliant
   DataArray.raster.idx_to_xy
   DataArray.raster.xy_to_idx
   DataArray.raster.rowcol
   DataArray.raster.xy
   DataArray.raster.flipud
   DataArray.raster.area_grid
   DataArray.raster.density_grid
   DataArray.raster.reclassify


Nodata handling and interpolation
=================================

.. autosummary::
   :template: autosummary/accessor_method.rst
   :toctree: _generated

   DataArray.raster.set_nodata
   DataArray.raster.mask_nodata
   DataArray.raster.interpolate_na


Clip
====

.. autosummary::
   :template: autosummary/accessor_method.rst
   :toctree: _generated

   DataArray.raster.clip_bbox
   DataArray.raster.clip_mask
   DataArray.raster.clip_geom

Reproject
=========

.. autosummary::
   :template: autosummary/accessor_method.rst
   :toctree: _generated

   DataArray.raster.reproject
   DataArray.raster.reindex2d
   DataArray.raster.reproject_like
   DataArray.raster.transform_bounds
   DataArray.raster.nearest_index

Transform
=========

.. autosummary::
   :template: autosummary/accessor_method.rst
   :toctree: _generated

   DataArray.raster.rasterize
   DataArray.raster.rasterize_geometry
   DataArray.raster.geometry_mask
   DataArray.raster.vectorize
   DataArray.raster.vector_grid

Sampling and zonal stats
========================

.. autosummary::
   :template: autosummary/accessor_method.rst
   :toctree: _generated

   DataArray.raster.sample
   DataArray.raster.zonal_stats


Low level methods
=================

.. autosummary::
   :toctree: _generated

   gis_utils.axes_attrs
   gis_utils.meridian_offset

==================
GeoDataset methods
==================

High level methods
==================

.. autosummary::
   :template: autosummary/accessor_method.rst
   :toctree: _generated

   DataArray.vector.from_gdf
   DataArray.vector.to_gdf
   DataArray.vector.from_netcdf
   DataArray.vector.to_netcdf
   Dataset.vector.from_gdf
   Dataset.vector.to_gdf
   Dataset.vector.from_netcdf
   Dataset.vector.to_netcdf

Attributes
==========

.. autosummary::
   :template: autosummary/accessor_attribute.rst
   :toctree: _generated

   DataArray.vector.attrs
   DataArray.vector.crs
   DataArray.vector.index_dim
   DataArray.vector.time_dim
   DataArray.vector.x_name
   DataArray.vector.y_name
   DataArray.vector.geom_name
   DataArray.vector.geom_type
   DataArray.vector.geom_format
   DataArray.vector.index
   DataArray.vector.bounds
   DataArray.vector.size
   DataArray.vector.sindex
   DataArray.vector.geometry
   Dataset.vector.attrs
   Dataset.vector.crs
   Dataset.vector.index_dim
   Dataset.vector.time_dim
   Dataset.vector.x_name
   Dataset.vector.y_name
   Dataset.vector.geom_name
   Dataset.vector.geom_type
   Dataset.vector.geom_format
   Dataset.vector.index
   Dataset.vector.bounds
   Dataset.vector.size
   Dataset.vector.sindex
   Dataset.vector.geometry

Conversion
==========

.. autosummary::
   :template: autosummary/accessor_method.rst
   :toctree: _generated

   DataArray.vector.ogr_compliant
   DataArray.vector.update_geometry
   DataArray.vector.to_geom
   DataArray.vector.to_xy
   DataArray.vector.to_wkt
   Dataset.vector.ogr_compliant
   Dataset.vector.update_geometry
   Dataset.vector.to_geom
   Dataset.vector.to_xy
   Dataset.vector.to_wkt

General methods
===============

.. autosummary::
   :template: autosummary/accessor_method.rst
   :toctree: _generated

   DataArray.vector.set_crs
   DataArray.vector.set_spatial_dims
   Dataset.vector.set_crs
   Dataset.vector.set_spatial_dims

Clip
====

.. autosummary::
   :template: autosummary/accessor_method.rst
   :toctree: _generated

   DataArray.vector.clip_bbox
   DataArray.vector.clip_geom
   Dataset.vector.clip_bbox
   Dataset.vector.clip_geom

Reproject
=========

.. autosummary::
   :template: autosummary/accessor_method.rst
   :toctree: _generated

   DataArray.vector.to_crs
   Dataset.vector.to_crs

======================
Flow direction methods
======================

These methods are based on the pyflwdir library. For more flow direction based methods
visit the `pyflwdir docs. <https://deltares.github.io/pyflwdir/latest/>`_

.. autosummary::
   :toctree: _generated

   flw.flwdir_from_da
   flw.d8_from_dem
   flw.reproject_hydrography_like
   flw.upscale_flwdir
   flw.stream_map
   flw.basin_map
   flw.gauge_map
   flw.outlet_map
   flw.clip_basins
   flw.dem_adjust


===================
General GIS methods
===================

Raster
=========

.. autosummary::
   :toctree: _generated

   gis_utils.create_vrt
   gis_utils.spread2d
   gis_utils.reggrid_area
   gis_utils.cellarea
   gis_utils.cellres

CRS and transform
=================

.. autosummary::
   :toctree: _generated

   gis_utils.parse_crs
   gis_utils.utm_crs
   gis_utils.affine_to_coords

Vector
======

.. autosummary::
   :toctree: _generated

   gis_utils.filter_gdf
   gis_utils.nearest
   gis_utils.nearest_merge


PCRaster I/O
============

.. autosummary::
   :toctree: _generated

   gis_utils.write_map
   gis_utils.write_clone


.. _statistics:

==========
Statistics
==========

Statistics and performance metrics
==================================

.. autosummary::
   :toctree: _generated

   stats.skills.bias
   stats.skills.percentual_bias
   stats.skills.nashsutcliffe
   stats.skills.lognashsutcliffe
   stats.skills.pearson_correlation
   stats.skills.spearman_rank_correlation
   stats.skills.kge
   stats.skills.kge_2012
   stats.skills.kge_non_parametric
   stats.skills.kge_non_parametric_flood
   stats.skills.rsquared
   stats.skills.mse
   stats.skills.rmse

=========
Utilities
=========

Configuration files
===================

.. autosummary::
   :toctree: _generated

   config.configread
   config.configwrite

Logging
=======

.. autosummary::
   :toctree: _generated

   log.setuplog
