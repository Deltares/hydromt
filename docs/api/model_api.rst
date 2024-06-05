.. currentmodule:: hydromt.model

.. _model_api:

=====
Model
=====


Model class
===========

Note that the base Model attributes and methods are available to all models.

.. autosummary::
   :toctree: ../_generated

   Model

High-level methods
-----------

.. autosummary::
   :toctree: ../_generated

   Model.read
   Model.write
   Model.write_data_catalog

General methods
---------------

.. autosummary::
   :toctree: ../_generated

   Model.build
   Model.update
   Model.get_component
   Model.add_component
   Model.test_equal

Model attributes
----------------

.. autosummary::
   :toctree: ../_generated

   Model.data_catalog
   Model.crs
   Model.root
   Model.region
   Model.components

ModelRoot
=========

.. autosummary::
   :toctree: ../_generated

   ModelRoot

Attributes
----------

.. autosummary::
   :toctree: ../_generated

   ModelRoot.mode
   ModelRoot.is_writing_mode
   ModelRoot.is_reading_mode
   ModelRoot.is_override_mode

General Methods
---------------

.. autosummary::
   :toctree: ../_generated

   ModelRoot.set

Model components
================

ModelComponent
--------------

Note that the base ModelComponent attributes and methods are available to all model
components.

.. autosummary::
   :toctree: ../_generated

   components.ModelComponent
   components.ModelComponent.model
   components.ModelComponent.data_catalog
   components.ModelComponent.logger
   components.ModelComponent.root

SpatialModelComponent
---------------------

.. autosummary::
   :toctree: ../_generated

   components.SpatialModelComponent
   components.SpatialModelComponent.model
   components.SpatialModelComponent.data_catalog
   components.SpatialModelComponent.logger
   components.SpatialModelComponent.root
   components.SpatialModelComponent.crs
   components.SpatialModelComponent.bounds
   components.SpatialModelComponent.region

**Plugin developer methods**

.. autosummary::
   :toctree: ../_generated

   components.SpatialModelComponent._region_data
   components.SpatialModelComponent.write_region
   components.SpatialModelComponent.test_equal

ConfigComponent
---------------

.. autosummary::
   :toctree: ../_generated

   components.ConfigComponent
   components.ConfigComponent.model
   components.ConfigComponent.data_catalog
   components.ConfigComponent.logger
   components.ConfigComponent.root
   components.ConfigComponent.data
   components.ConfigComponent.write
   components.ConfigComponent.read
   components.ConfigComponent.create
   components.ConfigComponent.update
   components.ConfigComponent.set
   components.ConfigComponent.get_value
   components.ConfigComponent.test_equal

GeomsComponent
==============

.. autosummary::
   :toctree: ../_generated

   components.GeomsComponent
   components.GeomsComponent.model
   components.GeomsComponent.data_catalog
   components.GeomsComponent.logger
   components.GeomsComponent.root
   components.GeomsComponent.data
   components.GeomsComponent.region
   components.GeomsComponent.write
   components.GeomsComponent.read
   components.GeomsComponent.set
   components.GeomsComponent.test_equal

TablesComponent
===============

.. autosummary::
   :toctree: ../_generated

   components.TablesComponent
   components.TablesComponent.model
   components.TablesComponent.data_catalog
   components.TablesComponent.logger
   components.TablesComponent.root
   components.TablesComponent.data
   components.TablesComponent.write
   components.TablesComponent.read
   components.TablesComponent.set
   components.TablesComponent.test_equal

DatasetsComponent
=================

.. autosummary::
   :toctree: ../_generated

   components.DatasetsComponent
   components.DatasetsComponent.model
   components.DatasetsComponent.data_catalog
   components.DatasetsComponent.logger
   components.DatasetsComponent.root
   components.DatasetsComponent.data
   components.DatasetsComponent.write
   components.DatasetsComponent.read
   components.DatasetsComponent.set
   components.DatasetsComponent.test_equal

SpatialDatasetsComponent
========================

.. autosummary::
   :toctree: ../_generated

   components.SpatialDatasetsComponent
   components.SpatialDatasetsComponent.model
   components.SpatialDatasetsComponent.data_catalog
   components.SpatialDatasetsComponent.logger
   components.SpatialDatasetsComponent.root
   components.SpatialDatasetsComponent.data
   components.SpatialDatasetsComponent.region
   components.SpatialDatasetsComponent.write
   components.SpatialDatasetsComponent.read
   components.SpatialDatasetsComponent.add_raster_data_from_raster_reclass
   components.SpatialDatasetsComponent.add_raster_data_from_rasterdataset
   components.SpatialDatasetsComponent.set
   components.SpatialDatasetsComponent.test_equal

GridComponent
=============

.. autosummary::
   :toctree: ../_generated

   components.GridComponent
   components.GridComponent.model
   components.GridComponent.data_catalog
   components.GridComponent.logger
   components.GridComponent.root
   components.GridComponent.res
   components.GridComponent.transform
   components.GridComponent.crs
   components.GridComponent.bounds
   components.GridComponent.region
   components.GridComponent.data
   components.GridComponent.write
   components.GridComponent.read
   components.GridComponent.create_from_region
   components.GridComponent.add_data_from_constant
   components.GridComponent.add_data_from_rasterdataset
   components.GridComponent.add_data_from_raster_reclass
   components.GridComponent.add_data_from_geodataframe
   components.GridComponent.set
   components.GridComponent.test_equal

MeshComponent
=============

.. autosummary::
   :toctree: ../_generated

   components.MeshComponent
   components.MeshComponent.model
   components.MeshComponent.data_catalog
   components.MeshComponent.logger
   components.MeshComponent.root
   components.MeshComponent.data
   components.MeshComponent.crs
   components.MeshComponent.bounds
   components.MeshComponent.region
   components.MeshComponent.mesh_names
   components.MeshComponent.mesh_grids
   components.MeshComponent.mesh_datasets
   components.MeshComponent.mesh_gdf
   components.MeshComponent.write
   components.MeshComponent.read
   components.MeshComponent.create_2d_from_region
   components.MeshComponent.add_2d_data_from_rasterdataset
   components.MeshComponent.add_2d_data_from_raster_reclass
   components.MeshComponent.set
   components.MeshComponent.get_mesh

VectorComponent
===============

.. autosummary::
   :toctree: ../_generated

   components.VectorComponent
   components.VectorComponent.data
   components.VectorComponent.geometry
   components.VectorComponent.index_dim
   components.VectorComponent.crs
   components.VectorComponent.model
   components.VectorComponent.data_catalog
   components.VectorComponent.logger
   components.VectorComponent.root
   components.VectorComponent.read
   components.VectorComponent.write
   components.VectorComponent.set
   components.VectorComponent.test_equal


.. _workflows_api:

Model Processes
===============

.. _workflows_grid_api:

Grid
----

.. autosummary::
   :toctree: ../_generated

   processes.grid.create_grid_from_region
   processes.grid.create_rotated_grid_from_geom
   processes.grid.grid_from_constant
   processes.grid.grid_from_rasterdataset
   processes.grid.grid_from_raster_reclass
   processes.grid.grid_from_geodataframe
   processes.grid.rotated_grid

.. _workflows_mesh_api:

Mesh
----

.. autosummary::
   :toctree: ../_generated

   processes.mesh.create_mesh2d_from_region
   processes.mesh.create_mesh2d_from_mesh
   processes.mesh.create_mesh2d_from_geom
   processes.mesh.mesh2d_from_rasterdataset
   processes.mesh.mesh2d_from_raster_reclass

.. _workflows_region_api:

Region
------

.. autosummary::
   :toctree: ../_generated

   processes.region.parse_region_basin
   processes.region.parse_region_bbox
   processes.region.parse_region_geom
   processes.region.parse_region_grid
   processes.region.parse_region_other_model
   processes.region.parse_region_mesh

.. _workflows_basin_api:

Basin mask
----------

.. autosummary::
   :toctree: ../_generated

   processes.basin_mask.get_basin_geometry

.. _workflows_rivers_api:

River bathymetry
----------------

.. autosummary::
   :toctree: ../_generated

   processes.rivers.river_width
   processes.rivers.river_depth

.. _workflows_forcing_api:

Meteo
-----

.. autosummary::
   :toctree: ../_generated

   processes.meteo.precip
   processes.meteo.temp
   processes.meteo.press
   processes.meteo.pet
   processes.meteo.wind
   processes.meteo.press_correction
   processes.meteo.temp_correction
   processes.meteo.resample_time
   processes.meteo.delta_freq
   processes.meteo.pet_debruin
   processes.meteo.pet_makkink
   processes.meteo.pm_fao56
