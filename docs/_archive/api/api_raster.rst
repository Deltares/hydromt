.. currentmodule:: hydromt

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
   :template: autosummary/accessor_method.rst
   :toctree: ../_generated

   DataArray.raster.set_crs
   DataArray.raster.set_spatial_dims
   DataArray.raster.reset_spatial_dims_attrs
   DataArray.raster.identical_grid
   DataArray.raster.aligned_grid
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


Low level methods
-----------------

.. autosummary::
   :toctree: ../_generated

   gis_utils.axes_attrs
   gis_utils.meridian_offset

