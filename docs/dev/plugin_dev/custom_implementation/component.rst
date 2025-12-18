.. _custom_component:

=================
Custom Components
=================

Components are the building blocks of a HydroMT model.
Each component is responsible for handling a specific aspect of the model, such as the grid,
configuration, mesh, rivers, catchments, pipes, or any custom behavior your plugin might require.

Because a component has a read and write method, it is then typically associated
with a specific model file.

Components encapsulate logic, data, and workflows in a reusable manner, and can interact with other components through the model instance.


Implementing a Component
^^^^^^^^^^^^^^^^^^^^^^^^^

Initialization
--------------

HydroMT defines two main types of components:

- ``ModelComponent`` - handles general model data and logic.
- ``SpatialModelComponent`` - extends ``ModelComponent`` to hold spatial data. A model
  **must** include at least one ``SpatialModelComponent`` for region-related functionality to work correctly.

All components receive a reference to the parent model at initialization, allowing access to other components as needed.

Example:

.. code-block:: python

    import pandas as pd

    from hydromt.model import Model
    from hydromt.model.components import ModelComponent

    class ExampleConfigComponent(ModelComponent):
        def __init__(
            self,
            model: Model,
            filename: str = "config.toml",
        ):
            self._data: dict | None = None
            super().__init__(model=model, filename=filename)

The ``filename`` property is essential because components are **lazy-loaded** by default.
Data is only loaded when needed, so this property defines where the component reads from or writes to.


Spatial Components
------------------

``SpatialModelComponent`` requires additional information at initialization. This is because they
may derive their region from another spatial component rather than having their own.

.. code-block:: python

    class ExampleGridComponent(SpatialModelComponent):
        def __init__(
            self,
            model,
            *,
            region_component: str | None = None,
            filename: str = "grid.nc",
            region_filename: str | None = "grid_region.geojson",
        ):
            ...

- ``region_component`` indicates which component provides the reference region
  (eg forcing grid may use the definition of the static grid component). Not needed if this component itself defines the region.
- ``region_filename`` defines where region data should be stored if applicable.


Required Attributes and Functions
---------------------------------

Components will by default inherit the following from their base classes:

- ``model`` - reference to the parent model
- ``data_catalog`` - access to datasets through HydroMT's catalog system
- ``root`` - model root directory for storing files

Components are expected to define the following attributes:

- ``data`` - holds the actual component data (DataFrame, xarray.Dataset, etc.)

Key functions include:

- ``read()`` - reading the component and its data from disk.
- ``write()`` - save the component's data to disk

Optional functions for enhanced workflows:

- ``set()`` - update or overwrite the component's data programmatically. Not usually exposed in YAML workflows.
- ``_initialize()`` - initializing an empty component.
- ``test_equal()`` - verify that two components (including data) are identical. Useful for unit tests.

Finally, you can provide additional functionality by providing the following optional functions:

- ``create``: the ability to construct the schematization of the component from the provided arguments.
  e.g. computation units like grid cells, mesh1d or network lines, vector units for lumped model etc.
- ``add_data``: the ability to transform and add model data and parameters to the component once the
  schematization is well-defined (i.e. add land-use data to grid or mesh etc.).

Functions annotated with ``@hydromt_step`` are **exposed to YAML workflows**, allowing users to call them as workflow steps.

Additionally, we encourage some best practices to be aware of when implementing a components:

- Make sure that your component calls ``super().__init__(model=model)`` in the ``__init__`` function
  of your component. This will make sure that references such as ``self.root`` and ``self.region`` are
  registered properly so you can access them.
- Your component should take some variation of a ``filename`` argument in its ``__init__`` function that
  is either required or provides a default that is not `None`. This should be saved as an attribute
  and be used for reading and writing when the user does not provide a different path as an argument
  to the read or write functions. This allows developers, plugin developers and users alike to both
  provide sensible defaults as well as the opportunity to overwrite them when necessary.


Usage Tips
----------

- Always define ``_filename`` or ``region_filename`` for proper lazy loading.
- Keep components modular and self-contained.
- Use adapters or transformations when interacting with external datasets.
- Test spatial components carefully to ensure regions are consistent with the model structure.
- Annotate functions with ``@hydromt_step`` only if they are workflow-relevant.


Example Workflow Step
---------------------

.. code-block:: yaml

    steps:
      - config.update:
          starttime: "2025-01-01T00:00:00"

      - grid.setup_landuse:
          landuse: "my_landuse.tif"
