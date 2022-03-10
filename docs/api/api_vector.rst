.. currentmodule:: hydromt

Geospatial timeseries methods
=============================

High level methods
------------------

.. autosummary::
   :template: autosummary/accessor_method.rst
   :toctree: ../_generated

   DataArray.vector.from_gdf
   DataArray.vector.to_gdf
   Dataset.vector.from_gdf


Attributes
----------

.. autosummary::
   :template: autosummary/accessor_attribute.rst
   :toctree: ../_generated

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
   :template: autosummary/accessor_method.rst
   :toctree: ../_generated

   DataArray.vector.set_crs
   DataArray.vector.set_spatial_dims
   DataArray.vector.reset_spatial_dims_attrs


Clip
----

.. autosummary::
   :template: autosummary/accessor_method.rst
   :toctree: ../_generated

   DataArray.vector.clip_bbox
   DataArray.vector.clip_geom

Reproject
---------

.. autosummary::
   :template: autosummary/accessor_method.rst
   :toctree: ../_generated

   DataArray.vector.to_crs

Low-level methods
-----------------

.. autosummary::
   :toctree: ../_generated

   gis_utils.filter_gdf