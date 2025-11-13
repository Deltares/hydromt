.. currentmodule:: hydromt.model

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
