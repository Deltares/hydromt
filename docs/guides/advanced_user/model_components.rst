.. _model_components:

Model components
================

.. note::

    This page is only about how to use components, not how to create a custom one. if
    you want to create a custom component or add default components to your custom model
    please refer to :ref:`custom_components` or :ref:`custom_model_builder`

Model components are how HydroMT specifies a lot of it's behaviors. The plugins can
make specialized components that can then be used to achieve basically anything you
want that is within the scope of hydromt.

The anatomy of a components
---------------------------

Components in hydromt receive a reference back to the model they are a part of, so that
they can access more global properties of the model such as the root, the data catalog
and the write permissions as well as other components. This means that components can't
effectively exit outside of a model.

There are basically two ways to add a component to a model: using the workflow yaml and
using the python interface. For more detail on how components are created and accessed
from the workflow file see :ref:`this page <model_workflow>`

Adding components using the Python interface
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Below is an example of how to construct a component (in this case a ``ConfigComponent``)
and how to add it to the model. Below we'll give a more detailed explanation of the
steps.

.. code-block:: python

    from hydromt.model.component import ConfigComponent
    from hydromt.model import Model

    model = Model(mode="w", root=str("tmp"))

    config_component = ConfigComponent(model)
    model.add_component("config", config_component)
    model.config.set("test.data", "value")


As previously mentioned when creating a component, a model needs to be constructed first
so that the component can gain a reference to it, so this is the first step.

Secondly the component is created with the model being passed as an argument (the
reference will be created for you by the component itself).

Note that while the
component now has access to the model, the model does not yet have access to the
component because it has not been told about it. This is done by the ``add_component``
step. Here it takes a name (must be a valid python name) and a component as an argument.

After a component has been added to a model. it can be accessed as a property on the
model (this is why it must be a valid python name)
