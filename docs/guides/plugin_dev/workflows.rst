
.. _kernel_config:

Preparing a model workflow
==========================

A user can define a complete pipeline of methods and their options to :ref:`build a model <model_build>` or :ref:`updating an existing model <model_update>`
in the configuration `.yaml file <https://en.wikipedia.org/wiki/YAML>`_

**Steps in brief:**

1) Start with a template of the :ref:`HydroMT model plugin <plugins>` with which you want to build / update a model. Templates can be found in the online documentation of each model plugin.
2) Edit / add / remove sections (i.e. methods) based on which components you want to build or adapt. The arguments of specific methods can be found in the API chapter in the online documentation of each model plugin.
3) Save the configuration file and use it in combination with the HydroMT :ref:`build <model_build>` and :ref:`update <model_update>` methods.

.. NOTE::

    The HydroMT configuration file used to be in ini format, this has been deprecated and can no longer be used.
    The new format (supported from version 0.7.1) is a yaml file, which is more flexible and easier to read and write.

.. NOTE::

    The HydroMT model configuration (.yaml) file should not be confused with the simulation configuration file of the model kernel.
    While the first defines how HydroMT should build or update a model, the latter defines the simulation for the model kernel.
    The format of the latter differs with each plugin, but can be accessed in HydroMT trough the :py:meth:`~hydromt.Model.config` component.


Workflow (.yaml) file
--------------------------------

The YAML file serves as the configuration and workflow definition for building and updating models in HydroMT. It uses a simple key-value syntax with structured sections to define steps and arguments for each method. Each step corresponds to a model action (method), and arguments specify how that action is applied. The order of execution is determined by the sequence of steps listed in the YAML file.

- **Model Type**: The top level defines the `modeltype`, such as "model".
- **Global Components**: Under the `global` section, you can define reusable components
  that apply throughout the model workflow. Each component is named and has a specific
  type, like `GridComponent` or `ConfigComponent`.
- **Steps**: The main logic of the model is defined under `steps`. Each step is a method call, which typically includes the method name and its required arguments. The order of the steps is critical, as HydroMT executes each step sequentially.

Below is an example of the YAML format used in HydroMT:

```yaml
---
modeltype: model
global:
  components:
    grid:
      type: GridComponent
    config:
      type: ConfigComponent

steps:
  - config.update:
      data:
        header.settings: value
        timers.end: '2010-02-15'
        timers.start: '2010-02-05'

  - grid.create_from_region:
      region:
        bbox: [12.05, 45.30, 12.85, 45.65]
      res: 0.01
      crs: 4326
      basin_index_path: merit_hydro_index
      hydrography_path: merit_hydro

  - grid.add_data_from_constant:
      constant: 0.01
      name: c1
      dtype: float32
      nodata: -99.0

  - grid.add_data_from_rasterdataset:
      raster_data: merit_hydro_1k
      variables:
        - elevtn
        - basins
      reproject_method:
        - average
        - mode

  - grid.add_data_from_rasterdataset:
      raster_data: vito
      fill_method: nearest
      reproject_method: mode
      rename:
        vito: landuse

  - grid.add_data_from_raster_reclass:
      raster_data: vito
      reclass_table_data: vito_reclass
      reclass_variables:
        - manning
      reproject_method:
        - average

  - grid.add_data_from_geodataframe:
      vector_data: hydro_lakes
      variables:
        - waterbody_id
        - Depth_avg
      nodata:
        - -1
        - -999.0
      rasterize_method: value
      rename:
        waterbody_id: lake_id
        Depth_avg: lake_depth

  - grid.add_data_from_geodataframe:
      vector_data: hydro_lakes
      rasterize_method: fraction
      rename:
        hydro_lakes: water_frac

  - write:
      components:
        - config
        - grid
```

### Explanation of Key Methods

- **`config.update`**: Updates configuration settings. In the example, it sets parameters like `header.settings`, and start and end times for the model run.
- **`grid.create_from_region`**: Creates a grid based on a specified bounding box (bbox), resolution, and coordinate reference system (CRS). Additional options include setting basin and hydrography paths.
- **`grid.add_data_from_constant`**: Adds a constant value to the grid. Parameters like `name`, `dtype`, and `nodata` specify how the constant data is handled.
- **`grid.add_data_from_rasterdataset`**: Adds data from a raster dataset. It includes options to specify variables, reprojection methods, and renaming rules for variables.
- **`grid.add_data_from_raster_reclass`**: Reclassifies raster data based on a specified reclassification table and applies transformations to the grid.
- **`grid.add_data_from_geodataframe`**: Adds vector data to the grid, rasterizing specific attributes, handling nodata values, and renaming variables.
- **`write`**: Specifies which components of the model (e.g., `config`, `grid`) should be written to disk at the end of the workflow. By default, all files are written unless specified otherwise.

It should be noted that, by default, the HydroMT `build` and `update` commands write all output files at the end of the workflow using the `write` method. This behavior can be customized by explicitly specifying the `write` step in the YAML file, allowing more granular control over which files are written and when.
