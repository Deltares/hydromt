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

    class AwesomeModel(Model):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            # Add default component
            self.add_component("grid", GridComponent)
            # Any additional custom initialization here

        def some_other_method(self):
            self.grid.do_something()

.. Note::
    You can access a model's components directly as attributes using the name provided at initialization (e.g., ``model.grid``) for convenience.

Always call ``super().__init__`` first to ensure the core functionality is properly initialized.


Key Model Properties
--------------------

Some important properties available in your model:

- ``root`` - path to the model instance folder.
- ``crs`` - reference coordinate system (pyproj.CRS).
- ``data_catalog`` - access to datasets for populating model components.
- ``components`` - dictionary of all model components by name.
- ``<component_name>`` - direct access to a specific component (e.g., ``model.grid`` in the ``AwesomeModel`` class).

For a complete list, see the :ref:`Model API documentation <model_api>`.


Setting Up Model Objects
------------------------

When building a model from scratch, inventory the components and their data requirements.
Most data processing logic resides in the components, but overarching functionality (e.g., perturbations for sensitivity analysis) can be implemented in the model itself.

Any method annotated with ``@hydromt_step`` is automatically exposed in the YAML workflow interface.

.. note::

   **Order of setup methods**: Typically, computational units (grids, meshes, vector layers) are set up first, followed by additional data layers.
   HydroMT does not enforce a strict order, but you can implement checks to ensure dependencies between setup methods are met.
   CLI workflows execute methods in YAML order; Python scripts can call setup functions in any order.


Setup Methods
--------------

HydroMT ``setup_<>`` methods generally perform four steps:

1. Read and parse data from the ``DataCatalog``.
2. Transform or process data (e.g., resampling, reprojection, filtering).
3. Rename or map HydroMT standard variable names to model-specific conventions.
4. Add the processed data to the appropriate model component.

Example: adding a landuse grid from raster data:

.. code-block:: python

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


Notes on Input Data Types
-------------------------

Setup methods usually convert external datasets (raster/vector) into standardized HydroMT model components.
Be explicit about supported input types in your docstrings.
You may define multiple setup functions to handle different input types (e.g., ``setup_landuse_from_raster`` and ``setup_landuse_from_vector``) to ensure correct processing.


Processes
---------

For complex functionality, define reusable functions outside the model class, called **processes**.
Processes allow you to keep models lightweight and testable.

Best practices for defining processes:

- Avoid passing the full model instance; pass only the required arguments.

  .. code-block:: python

      def interpolate_grid(grid: xr.Dataset, crs: CRS):
          ...

  Not:

  .. code-block:: python

      def interpolate_grid(model: AwesomeModel):
          grid = model.grid
          ...

- Use standard Python objects (``xarray.Dataset``, ``geopandas.GeoDataFrame``) in processes rather than model components.
- For GIS operations:
  - Raster: see :class:`xarray.Dataset`
  - Vector: see :class:`geopandas.GeoDataFrame`
  - Mesh/Ugrid: see `xugrid <https://deltares.github.io/xugrid/>`_

- HydroMT includes common workflows for:
  - Flow direction: `flwdir`
  - Basin masks: `basin_mask`
  - Statistical computations: `stats`


Workflow Integration
----------------------

Once setup methods and processes are defined, users can include them in YAML workflows:

.. code-block:: yaml

    steps:
      - landuse_component.setup_landuse:
          landuse: "my_landuse.tif"

      - grid_component.compute_flow_directions:
          method: "d8"
