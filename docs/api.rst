.. currentmodule:: hydromt

.. _api_reference:

#############
API reference
#############

======================
Command Line Interface
======================

.. toctree::
   :maxdepth: 2
   :hidden:

   api_cli/hydromt_cli.rst
   api_cli/hydromt_build.rst
   api_cli/hydromt_update.rst
   api_cli/hydromt_clip.rst
   api_cli/hydromt_export.rst
   api_cli/hydromt_check.rst


+------------------------------------------------+--------------------------------------------------------+
| `hydromt <api_cli/hydromt_cli.rst>`_           | Main command line interface of HydroMT.                |
+------------------------------------------------+--------------------------------------------------------+
| `hydromt build <api_cli/hydromt_build.rst>`_   | Build a model.                                         |
+------------------------------------------------+--------------------------------------------------------+
| `hydromt update <api_cli/hydromt_update.rst>`_ | Update an existing model.                              |
+------------------------------------------------+--------------------------------------------------------+
| `hydromt clip <api_cli/hydromt_clip.rst>`_     | Clip/Extract a submodel from an existing model.        |
+------------------------------------------------+--------------------------------------------------------+
| `hydromt export <api_cli/hydromt_export.rst>`_ | Export data extract from a data catalog.               |
+------------------------------------------------+--------------------------------------------------------+
| `hydromt check <api_cli/hydromt_check.rst>`_   | Check if data catalog or configuration file are valid. |
+------------------------------------------------+--------------------------------------------------------+

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
   data_catalog.DataCatalog.get_source
   data_catalog.DataCatalog.iter_sources
   data_catalog.DataCatalog.sources
   data_catalog.DataCatalog.keys
   data_catalog.DataCatalog.predefined_catalogs
   data_catalog.DataCatalog.to_dict
   data_catalog.DataCatalog.to_dataframe
   data_catalog.DataCatalog.to_yml
   data_catalog.DataCatalog.export_data
   data_catalog.DataCatalog.get_source_bbox
   data_catalog.DataCatalog.get_source_time_range

Add data sources
----------------

.. autosummary::
   :toctree: _generated

   data_catalog.DataCatalog.add_source
   data_catalog.DataCatalog.update
   data_catalog.DataCatalog.from_predefined_catalogs
   data_catalog.DataCatalog.from_yml
   data_catalog.DataCatalog.from_dict

.. _api_data_catalog_get:

Get data
--------

.. autosummary::
   :toctree: _generated

   data_catalog.DataCatalog.get_rasterdataset
   data_catalog.DataCatalog.get_geodataset
   data_catalog.DataCatalog.get_geodataframe
   data_catalog.DataCatalog.get_dataframe
   data_catalog.DataCatalog.get_dataset


Predefined data catalog
=======================

.. autosummary::
   :toctree: _generated

   predefined_catalog.PredefinedCatalog
   predefined_catalog.PredefinedCatalog.get_catalog_file
   predefined_catalog.create_registry_file



DataAdapter
===========

General
-------

.. autosummary::
   :toctree: _generated

   data_adapter.DataAdapter
   data_adapter.DataAdapter.summary
   data_adapter.DataAdapter.to_dict

RasterDataset
-------------

.. autosummary::
   :toctree: _generated

   data_adapter.RasterDatasetAdapter
   data_adapter.RasterDatasetAdapter.summary
   data_adapter.RasterDatasetAdapter.get_data
   data_adapter.RasterDatasetAdapter.to_dict
   data_adapter.RasterDatasetAdapter.to_file
   data_adapter.RasterDatasetAdapter.get_bbox
   data_adapter.RasterDatasetAdapter.get_time_range
   data_adapter.RasterDatasetAdapter.detect_bbox
   data_adapter.RasterDatasetAdapter.detect_time_range
   data_adapter.RasterDatasetAdapter.to_stac_catalog

GeoDataset
----------

.. autosummary::
   :toctree: _generated

   data_adapter.GeoDatasetAdapter
   data_adapter.GeoDatasetAdapter.summary
   data_adapter.GeoDatasetAdapter.get_data
   data_adapter.GeoDatasetAdapter.to_dict
   data_adapter.GeoDatasetAdapter.to_file
   data_adapter.GeoDatasetAdapter.get_bbox
   data_adapter.GeoDatasetAdapter.get_time_range
   data_adapter.GeoDatasetAdapter.detect_bbox
   data_adapter.GeoDatasetAdapter.detect_time_range
   data_adapter.GeoDatasetAdapter.to_stac_catalog

GeoDataFrame
------------

.. autosummary::
   :toctree: _generated

   data_adapter.GeoDataFrameAdapter
   data_adapter.GeoDataFrameAdapter.summary
   data_adapter.GeoDataFrameAdapter.get_data
   data_adapter.GeoDataFrameAdapter.to_dict
   data_adapter.GeoDataFrameAdapter.to_file
   data_adapter.GeoDataFrameAdapter.get_bbox
   data_adapter.GeoDataFrameAdapter.detect_bbox
   data_adapter.GeoDataFrameAdapter.to_stac_catalog

DataFrame
---------

.. autosummary::
   :toctree: _generated

   data_adapter.DataFrameAdapter
   data_adapter.DataFrameAdapter.summary
   data_adapter.DataFrameAdapter.get_data
   data_adapter.DataFrameAdapter.to_dict
   data_adapter.DataFrameAdapter.to_file
   data_adapter.DataFrameAdapter.to_stac_catalog

Dataset
-------

.. autosummary::
   :toctree: _generated

   data_adapter.DatasetAdapter
   data_adapter.DatasetAdapter.summary
   data_adapter.DatasetAdapter.get_data
   data_adapter.DatasetAdapter.to_dict
   data_adapter.DatasetAdapter.to_file
   data_adapter.DatasetAdapter.to_stac_catalog


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
   Model.root
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

   Model.read_nc
   Model.write_nc

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


.. _vector_model_api:

VectorModel
===========

.. autosummary::
   :toctree: _generated

   VectorModel


Components and attributes
-------------------------

.. autosummary::
   :toctree: _generated

   VectorModel.vector
   VectorModel.crs
   VectorModel.region

General methods
---------------

.. autosummary::
   :toctree: _generated

   VectorModel.set_vector
   VectorModel.read_vector
   VectorModel.write_vector

Setup methods
-------------

.. autosummary::
   :toctree: _generated

   VectorModel.setup_config
   VectorModel.setup_region
   VectorModel.setup_maps_from_rasterdataset
   VectorModel.setup_maps_from_raster_reclass


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
   MeshModel.mesh_names
   MeshModel.mesh_grids
   MeshModel.mesh_datasets
   MeshModel.mesh_gdf
   MeshModel.crs
   MeshModel.region
   MeshModel.bounds

General methods
---------------

.. autosummary::
   :toctree: _generated

   MeshModel.set_mesh
   MeshModel.get_mesh
   MeshModel.read_mesh
   MeshModel.write_mesh


Setup methods
-------------

.. autosummary::
   :toctree: _generated

   MeshModel.setup_config
   MeshModel.setup_region
   MeshModel.setup_mesh2d
   MeshModel.setup_mesh2d_from_rasterdataset
   MeshModel.setup_mesh2d_from_raster_reclass
   MeshModel.setup_maps_from_rasterdataset
   MeshModel.setup_maps_from_raster_reclass

.. _workflows_api:

=========
Workflows
=========

.. _workflows_grid_api:

Grid
====

.. autosummary::
   :toctree: _generated

   workflows.grid.grid_from_constant
   workflows.grid.grid_from_rasterdataset
   workflows.grid.grid_from_raster_reclass
   workflows.grid.grid_from_geodataframe

.. _workflows_mesh_api:

Mesh
====

.. autosummary::
   :toctree: _generated

   workflows.mesh.create_mesh2d
   workflows.mesh.mesh2d_from_rasterdataset
   workflows.mesh.mesh2d_from_raster_reclass

.. _workflows_basin_api:

Basin mask
==========

.. autosummary::
   :toctree: _generated

   workflows.basin_mask.get_basin_geometry
   workflows.basin_mask.parse_region

.. _workflows_rivers_api:

River bathymetry
================

.. autosummary::
   :toctree: _generated

   workflows.rivers.river_width
   workflows.rivers.river_depth

.. _workflows_forcing_api:

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
   DataArray.raster.to_slippy_tiles
   DataArray.raster.to_raster
   Dataset.raster.to_mapstack

.. _raster_api:

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

   DataArray.raster.clip
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

.. _geodataset_api:

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

.. _flw_api:

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

.. _gis_utils_api:

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


.. _statistics:

=====================================
Statistics and Extreme Value Analysis
=====================================

Statistics and performance metrics
==================================

.. autosummary::
   :toctree: _generated

   stats.skills.bias
   stats.skills.percentual_bias
   stats.skills.volumetric_error
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
   stats.skills.rsr

Extreme Value Analysis
=======================
.. autosummary::
   :toctree: _generated

   stats.extremes.get_peaks
   stats.extremes.fit_extremes
   stats.extremes.get_return_value
   stats.extremes.eva

=============
Design Events
=============
.. autosummary::
   :toctree: _generated

   stats.design_events.get_peak_hydrographs


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
