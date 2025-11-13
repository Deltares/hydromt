.. _custom_model:

=================
Custom Models
=================

The main purpose of a HydroMT plugin is to define and manage model instances for your specific software.
HydroMT models are composed of components and rely on flexible setup methods to integrate input data.

The figure below illustrates the overall model building process:

.. figure:: /_static/model_building_process.png
   :align: center

   Model building process in HydroMT.

This section focuses on initializing your custom model class, managing its components, and implementing setup methods.
Data preparation and pre-processing is handled by HydroMT core.


Initialization
--------------

Most basic functionality is provided by the HydroMT core ``Model`` class.
By inheriting from it, you typically don't need to implement low-level methods unless you want to provide default behavior or custom components.

Example of adding default components in your model:

.. code-block:: python

    from hydromt.model import Model
    from hydromt.model.components import ConfigComponent, GridComponent

    class ExampleModel(Model):
        def __init__(
            self,
            root: str | None = None,
            config_filename: str | None = "settings.toml",
            mode: str = "w",
            data_libs: list[str] | str | None = None,
        ):
            """Initialize ExampleModel."""
            components = {
                "config": ConfigComponent(
                    self,
                    filename=str(config_filename),
                ),
                "grid": GridComponent(self),
            }

            super().__init__(
                root,
                components=components,
                region_component="grid",
                mode=mode,
                data_libs=data_libs,
            )

        ## Properties
        # Components
        @property
        def config(self) -> ConfigComponent:
            """Return the config component."""
            return self.components["config"]

        @property
        def grid(self) -> GridComponent:
            """Return the grid component."""
            return self.components["grid"]

.. Note::
    You can access a model's components directly as attributes using the name provided at initialization (e.g., ``model.grid``) for convenience.

Always call ``super().__init__`` first to ensure the core functionality is properly initialized.

.. note::

   **region_component**: Specify which component defines the model region.
   This is optional if you have only one spatial component, but required if multiple components exist
   (e.g. grid, forcing, states etc.). The region component provides the spatial extent and CRS for data processing.


Key Model Properties
--------------------

Some important properties available in your model:

- ``root`` - path to the model instance folder.
- ``crs`` - reference coordinate system (pyproj.CRS).
- ``data_catalog`` - access to datasets for populating model components.
- ``components`` - dictionary of all model components by name.
- ``<component_name>`` - direct access to a specific component (e.g., ``model.grid`` in the ``ExampleModel`` class).

For a complete list, see the :ref:`Model API documentation <model_api>`.


Setting Up Model Objects
------------------------

When building a model from scratch, inventory the components and their data requirements.
Most data processing logic resides in the components, but overarching functionality
(e.g. methods updating several components) can be implemented in the model itself.

Any method annotated with ``@hydromt_step`` is automatically exposed in the YAML workflow interface.

.. note::

   **Order of setup methods**: Typically, computational units (grids, meshes, vector layers) are set up first, followed by additional data layers.
   HydroMT does not enforce a strict order, but you can implement checks to ensure dependencies between setup methods are met.
   CLI workflows execute methods in YAML order; Python scripts can call setup functions in any order.

A step or setup method will typically follow this pattern:

1. Read data from the ``DataCatalog`` using appropriate methods (e.g., ``get_rasterdataset``, ``get_geodataframe``).
2. Process or transform the data as needed (e.g., resampling, reprojection) using HydroMT processes.
3. Map HydroMT standard variable names to model-specific conventions.
4. Add the processed data to the relevant model component using component methods (e.g., ``grid.set``, ``config.set``).

Example: adding a landuse grid from raster data (either in the ``Model`` or ``ModelComponent`` subclass):

.. code-block:: python

    @hydromt_step
    def setup_landuse(
        self,
        landuse: Union[str, Path, xr.DataArray],
    ):
        """Add landuse data to the grid component.

        Parameters
        ----------
        landuse: str, Path, xr.DataArray
            Data catalog key, file path, or xarray data object.

        Notes
        -----
        This method automatically reads the input, applies transformations, and
        stores the resulting data in the grid component.
        """
        self.logger.info(f"Preparing landuse data from {landuse}")
        # 1. Read data
        da_landuse = self.data_catalog.get_rasterdataset(
            landuse,
            geom=self.region,
            buffer=2,
            variables=["landuse"],
        )
        # 2. Transform or process
        ds_out = hydromt.model.processes.grid.grid_from_rasterdataset(
            grid_like=self.grid.data,
            ds=da_landuse,
            fill_method="nearest",
            reproject_method="mode",
        )
        # 3. Map to model conventions
        ds_out = ds_out.rename({"landuse": "landuse_class"})
        # 4. Add to grid component
        self.set_grid(ds_out)


.. note::

    **Input data types**

    Setup methods usually convert external datasets (raster/vector) into standardized HydroMT model components.
    Be explicit about supported input types in your docstrings.
    You may define multiple setup functions to handle different input types
    (e.g., ``setup_landuse_from_raster`` and ``setup_landuse_from_vector``) to ensure correct processing.


Processes
---------

For complex functionality, you can define reusable functions outside the model class, called **processes**.
Processes allow you to keep models lightweight and testable.

Best practices for defining processes:

- Avoid passing the full model instance; pass only the required arguments.

  .. code-block:: python

      def interpolate_grid(grid: xr.Dataset, crs: CRS):
          ...

  Not:

  .. code-block:: python

      def interpolate_grid(model: ExampleModel):
          grid = model.grid
          ...

- Use standard Python objects (``xarray.Dataset``, ``geopandas.GeoDataFrame``) in processes rather than model components.
- For GIS operations or statistics:

  - Raster: :ref:`raster <raster_api>`
  - Geodataset: :ref:`vector <geodataset_api>`
  - Mesh/Ugrid: see `xugrid <https://deltares.github.io/xugrid/>`_
  - Flow direction: :ref:`flwdir <flw_api>`
  - Statistical computations: :ref:`stats <statistics>`

- HydroMT includes common processes for:

  - Region handling: :ref:`region <workflows_region_api>`
  - Basin mask: :ref:`basin_mask <workflows_basin_api>`
  - Grid creation and data: :ref:`grid <workflows_grid_api>`
  - Mesh creation and data: :ref:`mesh <workflows_mesh_api>`
  - Meteorological data processing: :ref:`meteo <workflows_forcing_api>`


Workflow Integration
----------------------

Once setup methods and processes are defined, users can include them in YAML workflows:

.. code-block:: yaml

    steps:
      - grid.setup_landuse:
          landuse: "my_landuse.tif"

      - grid.setup_flow_directions:
          method: "d8"
