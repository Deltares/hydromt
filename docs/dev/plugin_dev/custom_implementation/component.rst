.. _custom_component:

=================
Custom Components
=================

Components are the building blocks of a HydroMT model.
Each component is responsible for handling a specific aspect of the model, such as the grid, configuration, mesh, rivers, catchments, pipes, or any custom behavior your plugin might require.

Components encapsulate logic, data, and workflows in a reusable manner, and can interact with other components through the model instance.


Implementing a Component
^^^^^^^^^^^^^^^^^^^^^^^^^

Initialization
--------------

HydroMT defines two main types of components:

- `ModelComponent` - handles general model data and logic.
- `SpatialModelComponent` - extends `ModelComponent` to hold spatial data. A model **must** include at least one `SpatialModelComponent` for region-related functionality to work correctly.

All components receive a reference to the parent model at initialization, allowing access to other components as needed.

Example:

.. code-block:: python

    import pandas as pd

    from hydromt.model import Model
    from hydromt.model.components import ModelComponent

    class AwesomeRiverComponent(ModelComponent):
        def __init__(
            self,
            model: Model,
            filename: str = "component/{name}.csv",
        ):
            self._data: dict[str, pd.DataFrame | pd.Series] | None = None
            self._filename: str = filename
            super().__init__(model=model)

The `self._filename` property is essential because components are **lazy-loaded** by default.
Data is only loaded when needed, so this property defines where the component reads from or writes to.


Spatial Components
------------------

`SpatialModelComponent` requires additional information at initialization:

.. code-block:: python

    class SpatialExampleComponent(SpatialModelComponent):
        def __init__(
            self,
            model,
            *,
            region_component: str,
            filename: str = "spatial_component/{name}.nc",
            region_filename: str = "spatial_component/region.geojson",
        ):
            ...

- `region_component` indicates which component provides the reference region.
- `region_filename` defines where region data should be stored if applicable.
- Spatial components may derive their region from another spatial component rather than having their own.


Required Attributes and Functions
---------------------------------

Components are expected to define the following attributes:

- `data` - holds the actual component data (DataFrame, xarray.Dataset, etc.)
- `model` - reference to the parent model
- `data_catalog` - access to datasets through HydroMT's catalog system
- `root` - model root directory for storing files

Key functions include:

- `read()` - load the component's data
- `write()` - save the component's data

Optional functions for enhanced workflows:

- `set()` - overwrite the component's data programmatically. Not usually exposed in YAML workflows.
- `test_equal()` - verify that two components (including data) are identical. Useful for unit tests.

Functions annotated with `@hydromt_step` are **exposed to YAML workflows**, allowing users to call them as workflow steps.


Usage Tips
----------

- Always define `_filename` or `region_filename` for proper lazy loading.
- Keep components modular and self-contained.
- Use adapters or transformations when interacting with external datasets.
- Test spatial components carefully to ensure regions are consistent with the model structure.
- Annotate functions with `@hydromt_step` only if they are workflow-relevant.


Example Workflow Step
---------------------

.. code-block:: yaml

    steps:
      - river_component.load:
          filename: "data/rivers.csv"

      - grid_component.compute_flow_directions:
          method: "d8"
