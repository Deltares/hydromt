.. currentmodule:: hydromt.model

.. _model_components_api:

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
   components.ModelComponent.read
   components.ModelComponent.write
   components.ModelComponent.root

SpatialModelComponent
---------------------

.. autosummary::
   :toctree: ../_generated

   components.SpatialModelComponent
   components.SpatialModelComponent.model
   components.SpatialModelComponent.data_catalog

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

   components.GridComponent.root
   components.GridComponent.res
   components.GridComponent.transform
   components.GridComponent.crs
   components.GridComponent.bounds
   components.GridComponent.region
   components.GridComponent.data
   components.GridComponent.write
   components.GridComponent.read
   components.GridComponent.set
   components.GridComponent.test_equal

MeshComponent
=============

.. autosummary::
   :toctree: ../_generated

   components.MeshComponent
   components.MeshComponent.model
   components.MeshComponent.data_catalog
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

   components.VectorComponent.root
   components.VectorComponent.read
   components.VectorComponent.write
   components.VectorComponent.set
   components.VectorComponent.test_equal
