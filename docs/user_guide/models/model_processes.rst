.. currentmodule:: hydromt.model.processes

.. _model_processes:

Model processes
===============
HydroMT uses **processes** functions to transform raw input data into model-ready
inputs and parameters. These processes can be as simple as reprojecting a raster dataset
to the model grid, or more complex operations like delineating a river network based on
a digital elevation model (DEM).

Processes are only available through the Python API and are usually the backbone of the model
and components ``setup_`` or ``add_data_`` methods. HydroMT proposes a collection of methods
that can be re-used in your python scripts or when developing your own plugins.

Grid
----
These methods allow to either create a grid or add data to an existing grid from different
data sources.

.. list-table::
   :widths: 30 70
   :header-rows: 1

   * - Processes
     - Description
   * - :py:meth:`~grid.create_grid_from_region`
     -  Create a 2D regular grid from a specified region.
   * - :py:meth:`~grid.create_rotated_grid_from_geom`
     - Create a rotated grid based on a geometry.
   * - :py:meth:`~grid.grid_from_constant`
     - Add data to a grid using a constant value.
   * - :py:meth:`~grid.grid_from_rasterdataset`
     - Add data to a grid by resampling a raster dataset.
   * - :py:meth:`~grid.grid_from_raster_reclass`
     - Reclassify raster data and add it to a grid.
   * - :py:meth:`~grid.grid_from_geodataframe`
     - Add data to a grid by rasterizing a GeoDataFrame.
   * - :py:meth:`~grid.rotated_grid`
     - Return the origin (x0, y0), shape (mmax, nmax) and rotation of the rotated grid.

Mesh
----
These methods allow to either create a mesh or add data to an existing mesh from different
data sources.

.. list-table::
   :widths: 30 70
   :header-rows: 1

   * - Processes
     - Description
   * - :py:meth:`~mesh.create_mesh2d_from_region`
     - Create a 2D mesh from a specified region according to UGRID conventions.
   * - :py:meth:`~mesh.create_mesh2d_from_mesh`
     - Create a 2D mesh based on an existing mesh.
   * - :py:meth:`~mesh.create_mesh2d_from_geom`
     - Create a regular 2D mesh from a boundary geometry.
   * - :py:meth:`~mesh.mesh2d_from_rasterdataset`
     - Add data to a mesh by resampling a raster dataset.
   * - :py:meth:`~mesh.mesh2d_from_raster_reclass`
     - Reclassify raster data and add it to a mesh.

Region
------
These methods allow to parse different region definitions (from the HydroMT region dictionary)
for setting up a model region.

.. list-table::
   :widths: 30 70
   :header-rows: 1

   * - Processes
     - Description
   * - :py:meth:`~region.parse_region_basin`
     - Parse a hydrographic basin region definition.
   * - :py:meth:`~region.parse_region_bbox`
     - Parse a bounding box region definition.
   * - :py:meth:`~region.parse_region_geom`
     - Parse a geometry file region definition.
   * - :py:meth:`~region.parse_region_grid`
     - Parse a raster grid file region definition.
   * - :py:meth:`~region.parse_region_other_model`
     - Parse a region definition based on another HydroMT model.
   * - :py:meth:`~region.parse_region_mesh`
     - Parse a mesh file region definition.

Basin mask
----------
This method allows to delineate hydrographic regions (basin, interbasin, subbasin) using
the region dictionary and a hydrography (DEM, flow direction) dataset.

.. list-table::
   :widths: 30 70
   :header-rows: 1

   * - Processes
     - Description
   * - :py:meth:`~basin_mask.get_basin_geometry`
     - Return a geometry of the (sub)(inter)basin(s).

River bathymetry
----------------
These methods allow to estimate river channel dimensions based on either direct cross-section
information or empirical relationships.

.. list-table::
   :widths: 30 70
   :header-rows: 1

   * - Processes
     - Description
   * - :py:meth:`~rivers.river_width`
     - Return average river width along a segment based on a river mask raster.
   * - :py:meth:`~rivers.river_depth`
     - Derive river depth estimates from data or based on bankfull discharge.

Meteo
-----
These methods allow to process meteorological data for use in gridded models. This includes
time resampling, downscaling, and calculation of potential evapotranspiration.

.. list-table::
   :widths: 30 70
   :header-rows: 1

   * - Processes
     - Description
   * - :py:meth:`~meteo.precip`
     - Process precipitation data.
   * - :py:meth:`~meteo.temp`
     - Process temperature data.
   * - :py:meth:`~meteo.press`
     - Process atmospheric pressure data.
   * - :py:meth:`~meteo.pet`
     - Determine reference evapotranspiration.
   * - :py:meth:`~meteo.wind`
     - Process wind speed data.
   * - :py:meth:`~meteo.press_correction`
     - Pressure correction based on elevation lapse_rate.
   * - :py:meth:`~meteo.temp_correction`
     - Temperature correction based on elevation data.
   * - :py:meth:`~meteo.resample_time`
     - Resample meteorological data to a different time frequency.
   * - :py:meth:`~meteo.pet_debruin`
     - Calculate potential evapotranspiration using De Bruin method.
   * - :py:meth:`~meteo.pet_makkink`
     - Calculate potential evapotranspiration using Makkink method.
   * - :py:meth:`~meteo.pm_fao56`
     - Calculate potential evapotranspiration using Penman-Monteith FAO56 method.
