.. _model_components:

Model components (advanced)
===========================

.. note::

    This page is only about how to use components, not how to create a custom one. if
    you want to create a custom component or add default components to your custom model
    please refer to :ref:`custom_components` or :ref:`custom_model_builder`

Model components are how HydroMT specifies a lot of its behaviors. Basically For HydroMT,
a model is several by several components or files that you can then build and update step by step.

Each specific model and plugins will have its own set of components that it uses. Some common
components could be ``config``, ``grid``, ``forcing``, ``geoms``, ``states``, etc.
Always visit the documentation of the specific plugin you are using to see which components
it supports and what they do.

The anatomy of a components
---------------------------

Components in hydromt receive a reference back to the model they are a part of, so that
they can access more global properties of the model such as the root, the data catalog
and the write permissions as well as other components. This means that components can't
effectively exist outside of a model.

In general a component will have the following properties and functions:

- ``data``: the main data object of the component. This could be a dictionary,
  a xarray object, a geopandas dataframe, etc. depending on the component.
- ``read()``: function to read the component from disk into memory.
- ``write()``: function to write the component from memory to disk.
- ``set()``: function to add or update values in the component data.
- Other functions that are specific to the component and the model plugin.



Adding components to a model
----------------------------
If you are using a plugin, the components should be created automatically for you.
However if you are using the hydromt core standalone and its ``Model`` class, you
will need to create and add components yourself.

There are basically two ways to add a component to a model: using the workflow yaml and
using the python interface:

.. tab-set::

    .. tab-item:: Workflow Yaml (CLI or Python)

        In the ``global`` section of the workflow yaml, you can define which components
        the model should have, if they are spatial components and which components the
        model should use to define its region:

        .. code-block:: yaml

            modeltype: model
            global:
                components:
                    grid:
                        type: GridComponent
                    config:
                        type: ConfigComponent
                    forcing:
                        type: GridComponent
                        region_component: grid
                region_component: grid

    .. tab-item:: Python API


        Below is an example of how to construct a components
        and how to add it to the model:

        .. code-block:: python

            from hydromt.model.component import ConfigComponent, GridComponent
            from hydromt.model import Model

            # Prepare your components
            components = {
                "config": {
                    "type": ConfigComponent,
                    "filename": "config.yaml",
                },
                "grid": {
                    "type": GridComponent,
                },
                "forcing": {
                    "type": GridComponent,
                    "region_component": "grid",
                },
            }

            # Instantiate the model
            model = Model(
                root=str("tmp"),
                data_catalog=["artifact_data"],
                mode="w",
                components=components,
                region_component="grid",
            )


In the above examples, you can see that ``components`` should take a mapping where the keys are the name
the component will have (e.g. ``grid``). These must then again take a mapping that specifies at least
the type of component. The name of the component type should correspond to the python
class name (e.g. ``GridComponent``).

An additional point of note is that spatial components (such as ``forcing`` and
``grid``) in the examples above, can either define their own region (``grid``) or derive
their region from another component (``forcing``). This can be done by specifying the
``region_component`` key, and should refer to the name of the spatial component you wish
to use. You can also specify which spatial component the model should derive it's region
from.
