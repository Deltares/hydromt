.. currentmodule:: hydromt.gis


.. _raster_api:

===========
GIS methods
===========

Raster methods
==============

High level methods
------------------

.. autosummary::
   :toctree: ../_generated

   raster.full
   raster.full_like
   raster.full_from_transform
   raster_merge.merge

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
---------------

.. autosummary::
   :template: autosummary/accessor_method.rst
   :toctree: ../_generated

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
---------------------------------

.. autosummary::
   :template: autosummary/accessor_method.rst
   :toctree: ../_generated

   DataArray.raster.set_nodata
   DataArray.raster.mask_nodata
   DataArray.raster.interpolate_na


Clip
----

.. autosummary::
   :template: autosummary/accessor_method.rst
   :toctree: ../_generated

   DataArray.raster.clip
   DataArray.raster.clip_bbox
   DataArray.raster.clip_mask
   DataArray.raster.clip_geom

Reproject
---------

.. autosummary::
   :template: autosummary/accessor_method.rst
   :toctree: ../_generated

   DataArray.raster.reproject
   DataArray.raster.reindex2d
   DataArray.raster.reproject_like
   DataArray.raster.transform_bounds
   DataArray.raster.nearest_index

Transform
---------

.. autosummary::
   :template: autosummary/accessor_method.rst
   :toctree: ../_generated

   DataArray.raster.rasterize
   DataArray.raster.rasterize_geometry
   DataArray.raster.geometry_mask
   DataArray.raster.vectorize
   DataArray.raster.vector_grid

Sampling and zonal stats
------------------------

.. autosummary::
   :template: autosummary/accessor_method.rst
   :toctree: ../_generated

   DataArray.raster.sample
   DataArray.raster.zonal_stats


Writing methods
---------------

.. autosummary::
   :template: autosummary/accessor_method.rst
   :toctree: ../_generated

   DataArray.raster.to_xyz_tiles
   DataArray.raster.to_slippy_tiles
   DataArray.raster.to_raster
   Dataset.raster.to_mapstack

.. _geodataset_api:

GeoDataset methods
==================

High level methods
------------------

.. autosummary::
   :template: autosummary/accessor_method.rst
   :toctree: ../_generated

   DataArray.vector.from_gdf
   DataArray.vector.to_gdf
   DataArray.vector.from_netcdf
   DataArray.vector.to_netcdf
   Dataset.vector.from_gdf
   Dataset.vector.to_gdf
   Dataset.vector.from_netcdf
   Dataset.vector.to_netcdf

Attributes
----------

.. autosummary::
   :template: autosummary/accessor_attribute.rst
   :toctree: ../_generated

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
----------

.. autosummary::
   :template: autosummary/accessor_method.rst
   :toctree: ../_generated

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
---------------

.. autosummary::
   :template: autosummary/accessor_method.rst
   :toctree: ../_generated

   DataArray.vector.set_crs
   DataArray.vector.set_spatial_dims
   Dataset.vector.set_crs
   Dataset.vector.set_spatial_dims

Clip
----

.. autosummary::
   :template: autosummary/accessor_method.rst
   :toctree: ../_generated

   DataArray.vector.clip_bbox
   DataArray.vector.clip_geom
   Dataset.vector.clip_bbox
   Dataset.vector.clip_geom

Reproject
---------

.. autosummary::
   :template: autosummary/accessor_method.rst
   :toctree: ../_generated

   DataArray.vector.to_crs
   Dataset.vector.to_crs

.. _flw_api:

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

.. _gis_utils_api:

GIS utility methods
===================

Raster
------

.. autosummary::
   :toctree: ../_generated

   create_vrt.create_vrt
   raster_utils.spread2d
   raster_utils.reggrid_area
   raster_utils.cellarea
   raster_utils.cellres
   raster_utils.meridian_offset
   raster_utils.affine_to_coords
   raster_utils.affine_to_meshgrid

Vector
------

.. autosummary::
   :toctree: ../_generated

   vector_utils.filter_gdf
   vector_utils.nearest
   vector_utils.nearest_merge


General
-------

.. autosummary::
   :toctree: ../_generated

   gis_utils.parse_crs
   gis_utils.utm_crs
   gis_utils.bbox_from_file_and_filters
   gis_utils.parse_geom_bbox_buffer
   gis_utils.to_geographic_bbox
   gis_utils.axes_attrs
