.. currentmodule:: hydromt

#############
Methods
#############

Reading/writing
===============

Reading methods
---------------

.. autosummary::
   :toctree: ../_generated

   io.open_raster
   io.open_mfraster
   io.open_raster_from_tindex
   io.open_vector
   io.open_vector_from_table
   io.open_geodataset
   io.open_timeseries_from_table

Raster writing methods
----------------------

.. autosummary::
   :toctree: ../_generated
   :template: autosummary/accessor_method.rst

   DataArray.raster.to_raster
   Dataset.raster.to_mapstack


Raster methods
==============

High level methods
------------------

.. autosummary::
   :toctree: ../_generated

   merge.merge
   raster.full
   raster.full_like
   raster.full_from_transform

.. autosummary::
   :toctree: ../_generated
   :template: autosummary/accessor_method.rst
   
   DataArray.raster.from_numpy
   Dataset.raster.from_numpy

Attributes
----------

.. autosummary::
   :toctree: ../_generated
   :template: autosummary/accessor_attribute.rst

   DataArray.raster.attrs
   DataArray.raster.crs
   DataArray.raster.bounds
   DataArray.raster.transform
   DataArray.raster.res
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
---------------

.. autosummary::
   :toctree: ../_generated
   :template: autosummary/accessor_method.rst

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


Nodata handling and interpolation
---------------------------------

.. autosummary::
   :toctree: ../_generated
   :template: autosummary/accessor_method.rst

   DataArray.raster.set_nodata
   DataArray.raster.mask_nodata
   DataArray.raster.interpolate_na


Clip
----

.. autosummary::
   :toctree: ../_generated
   :template: autosummary/accessor_method.rst

   DataArray.raster.clip_bbox
   DataArray.raster.clip_mask
   DataArray.raster.clip_geom

Reproject
---------

.. autosummary::
   :toctree: ../_generated
   :template: autosummary/accessor_method.rst

   DataArray.raster.reproject
   DataArray.raster.reindex2d
   DataArray.raster.reproject_like
   DataArray.raster.transform_bounds
   DataArray.raster.nearest_index

Transform
---------

.. autosummary::
   :toctree: ../_generated
   :template: autosummary/accessor_method.rst

   DataArray.raster.rasterize
   DataArray.raster.geometry_mask
   DataArray.raster.vectorize
   DataArray.raster.vector_grid

Sampling and zonal stats
------------------------

.. autosummary::
   :toctree: ../_generated
   :template: autosummary/accessor_method.rst

   DataArray.raster.sample
   DataArray.raster.zonal_stats


Low level methods
-----------------

.. autosummary::
   :toctree: ../_generated

   gis_utils.axes_attrs
   gis_utils.meridian_offset

Geospatial timeseries methods
=============================

High level methods
------------------

.. autosummary::
   :toctree: ../_generated
   :template: autosummary/accessor_method.rst

   DataArray.vector.from_gdf
   DataArray.vector.to_gdf
   Dataset.vector.from_gdf


Attributes
----------

.. autosummary::
   :toctree: ../_generated
   :template: autosummary/accessor_attribute.rst

   DataArray.vector.attrs
   DataArray.vector.crs
   DataArray.vector.index_dim
   DataArray.vector.time_dim
   DataArray.vector.y_dim
   DataArray.vector.x_dim
   DataArray.vector.xcoords
   DataArray.vector.ycoords
   DataArray.vector.index
   DataArray.vector.bounds
   DataArray.vector.total_bounds
   DataArray.vector.sindex
   DataArray.vector.geometry

General methods
---------------

.. autosummary::
   :toctree: ../_generated
   :template: autosummary/accessor_method.rst

   DataArray.vector.set_crs
   DataArray.vector.set_spatial_dims
   DataArray.vector.reset_spatial_dims_attrs


Clip
----

.. autosummary::
   :toctree: ../_generated
   :template: autosummary/accessor_method.rst

   DataArray.vector.clip_bbox
   DataArray.vector.clip_geom
   DataArray.vector.clip_mask

Reproject
---------

.. autosummary::
   :toctree: ../_generated
   :template: autosummary/accessor_method.rst

   DataArray.vector.to_crs

Low-level methods
-----------------

.. autosummary::
   :toctree: ../_generated

   gis_utils.filter_gdf


Flow direction methods
======================

These methods are based on the pyflwdir library. For more flow direction based methods
visit the `pyflwdir docs. <https://deltares.github.io/pyflwdir/latest/>`_

.. autosummary::
   :toctree: ../_generated

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


General GIS methods
===================

.. autosummary::
   :toctree: ../_generated
   
   gis_utils.spread2d
   gis_utils.parse_crs
   gis_utils.utm_crs
   gis_utils.affine_to_coords
   gis_utils.reggrid_area
   gis_utils.cellarea
   gis_utils.cellres
   gis_utils.write_map 
   gis_utils.write_clone
   


Statistics and performance metrics
==================================

.. autosummary::
   :toctree: ../_generated
   
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

