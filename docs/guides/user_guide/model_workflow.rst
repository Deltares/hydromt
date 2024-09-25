.. _model_workflow:

Creating a Model Workflow
=========================

A model workflow (the ``.yaml`` file that tells hydromt what todo) consists of three
main sections:

1. ``modeltype``
2. ``global``
3. ``steps``

The model type tells hydromt which model to use. In the case of using hydromt core,
``model`` is the only option but if you have plugins installed those will probably
provide other options as well. You can discover which options you have installed with
the command ``hydromt --models``.

The global is where you provide any configuration that the model will need at
initialization. This is where you for example, can (and must in the case of core) add
components, and tell the model which spacial component should be used to figure out what
the model region is.

In a workflow file, all the components that a model needs to have (appart from any
default components your plugin has already provided) must be declared upfront. This is
done with the ``global`` keyword (typcially placed at the start of the file)

it should look like this

.. code-block:: yaml

    global:
        components:
            grid:
                type: GridComponent
            config:
                type: ConfigComponent
            forcing:
                type: SpatialDatasetsComponent
                region_component: grid
        region_component: grid


here you can see that ``components`` should take a mapping where the keys are the name
the component will have. These must then again take a mapping that specifies at least
the type of component. The name of the component type should correspond to the python
class name.

An additional point of note is that spacial components (such as ``forcing`` and
``grid``) in the example above, can either define their own region (``grid``) or derive
their region from another component (``forcing``). This can be done by specifying the
``region_component`` key, and should refer to the name of the spacial component you wish
to use. You can also specify which spacial component the model should derive it's region
from. If you have only one spacial component, this information may be omitted, but if
you have multiple (as in the example above) you must also specify which component the
model should derive it's region from.


Finally there is the ``steps`` part of the workflow. This should be a list, where each
list item should be a name of a function you want to run, followed by any arguments you
want to pass to that function. You can use the ``.`` syntax to call functions on
components, or omit this if the function you want to call is defined on the model.

For example, after you have specified the components that should be added to your model
in the ``global`` key, you can use them in the steps of your workflow like so:

.. code-block:: yaml

    steps:
        - grid.add_data_from_constant:
            constant: 0.01
            name: "c1"
            nodata: -99.0
        - ...

in the example above ``grid`` is the name of the component and
``add_data_from_constant`` is the function on it that you want to call. Please check the
specific component for what functions are available. Note that only functions that have
the ``@hydromt_step`` decorator are available to use from the yaml workflow.
