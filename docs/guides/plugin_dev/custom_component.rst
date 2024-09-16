.. _custom_components:

Custom components
=================

Components are a large part of how HydroMT defines its behavior.
A component is a part of the model architecture that is responsible
for handling a specific part of the data, such as the Grid, Config, or Mesh
but also potentially something like rivers, catchments, pipes or other custom behavior
your plugin might need.


Implementing a component
^^^^^^^^^^^^^^^^^^^^^^^^

Initialisation
--------------

There are generally two types of components you might want to implement:
`ModelComponent` and `SpatialModelComponent`. They are similar but the
`SpatialModelComponent` is meant to hold spacial data. A model MUST have at least one
`SpatialModelComponent` for it's region functionality to function properly.

Components of any kind, should take a reference to the model they are a part of at
initialization so that components can access other components through the model as is
necessary. Typically this is done like so:

.. doctest:: python

    class AwesomeRiverComponent(ModelComponent):
        def __init__(
            self,
            model: Model,
            filename: str = "component/{name}.csv",
        ):
            self._data: Optional[Dict[str, Union[pd.DataFrame, pd.Series]]] = None
            self._filename: str = filename
            super().__init__(model=model)


It is also important that the component has a ``self._filename`` property defined
because the data in components they are lazy loaded by default, meaning that
the data associated with them only get's loaded if necessary. Therefore it is important
to set a property on the component so that it can read from/write to the correct default
file location, without additional input.

``SpatialModelComponent`` s should take some additional information in their
initialisation:

.. code-block:: python

    def __init__(
        self,
        model: Model,
        *,
        region_component: str,
        filename: str = "spatial_component/{name}.nc",
        region_filename: str = "spatial_component/region.geojson",
    ):
        ...

The ``region_filename`` is a similar file location within the model root where region data
should be written if necessary. Note also that spatial components do not necessarily
have their own region. Sometimes they can derive their region from other spacial
components, such as a subgrid deriving its region from the main grid. If this is the
case then set the ``region_component`` to the name of the component from which you wish to
derive the region.

Required attributes
-------------------

Aside form initialisation the components are expected to have some other properties
(`data`, `model`, `data_catalog` and `root`) and functions (`read` and `write`).
Functions annotated with the ``@hydromt_step`` decorator will be available to users of a
workflow yaml. Depending on the context your component may also want to implement the
functions `set` (which is typically not annotaed with the ``@hydromt_step`` decorator
since it cannot be sued in a yaml workflow)`, and `test_equal`. `set` is typically used
by python users to overwrite data in the component after they have done something with
it outside of your component. `test_equal` is purely for testing purposes and should test
whether the component provided (including potential data) is equal to the component it
is being run on. This is very useful for testing that includes data.
