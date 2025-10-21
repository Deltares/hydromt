.. _register_plugins:

Implementing Custom behavior
============================

Due to the extremely wide range of data formats and conventions out in the world
HydroMT offers a lot of ways to define custom behaviour, which is where you will
probably spend most of your time as a plugin developer. We offer the possibility
of extending what is in hydromt via the use of entry points. These entry points
are a way of telling hydromt core about code you would like it to use to do specialised
such as reading and writing a custom file format. Currently core exposes the following
entry points to extend its functionalities:

* Model
* ModelComponent
* Resolvers
* Driver
* Catalog

We will elaborate on each of these entrypoints in more detail in their own section but
first we will cover the necessary information that is common to all: how to tell hydromt
to use your custom behaviour. As an example we will use the hypothetical hydromt plugin
for the (fictional) Advanced Water and Environmental Systems Optimization and Modeling
Engine or AWESOME for short. The package will be called `hydromt_awesome`.

EntryPoints
===========

Entrypoints are how you can tell Python (and by extention HydroMT) about your code that
you would like it to use. You can find more detailed information about them in the
`official Python documentation <https://packaging.python.org/en/latest/specifications/entry-points/>`
You first do this by specifying the entrypoint to tell HydroMT about in your
`pyproject.toml`. So for example in the `pyproject.toml` of `hydromt_awesome` we might
write:

.. code-block::toml

    [project.entry-points."hydromt.models"]
    awesome = "hydromt_awesome.awesome:AwesomeModel"

    [project.entry-points."hydromt.ModelComponent"]
    awesome_rivers = "hydromt_awesome.awesome.components:RiverComponent"
    awesome_lakes = "hydromt_awesome.awesome.components:LakesComponent"
    awesome_pipes = "hydromt_awesome.awesome.components:PipeComponent"

    [project.entry-points."hydromt.Resolver"]
    awesome_dem_resolver = "hydromt_awesome.awesome.resolvers:DEMResolver"

    [project.entry-points."hydromt.Driver"]
    awesome_kernel_driver = "hydromt_awesome.awesome.drivers:AwesomeKernelFileDriver"

    [project.entry-points."hydromt.Catalog"]
    awesome_default_data_catalog = "hydromt_awesome.awesome.data_catalog:AwesomeCatalog"


The structure of these should be very similar across all entrypoints. The header should
be of the form `project.entry-points.` followed by the path to the hydromt object you
want to provide your own implementation for. In most cases this should be accessible
from the root level of hydromt so one of the examples above should suffice.

The key under the header is just a name for the plugin that will be used to display
where a plugin is being loaded from (handy if you have multiple plugins loaded)
The value of the pair should be the path from the root necessary to import your object,
usually this will point to a submodule.

In the submodule you point to should define a global `__hydromt_eps__` list of names of
objects to import. Note that this should be a list of strings. For example, here is the
`__hydromt_eps__` list that core devines for it's model components (we use the same
mechanism as you will)

.. code-block:: python

    __hydromt_eps__ = [
        "ConfigComponent",
        "DatasetsComponent",
        "GeomsComponent",
        "GridComponent",
        "MeshComponent",
        "SpatialDatasetsComponent",
        "TablesComponent",
        "VectorComponent",
    ]

The objects you define in this list should of course be imported before you define them.
All the classes you define here should become available to access through the HydroMT
plugin functionality. You can verify that HydroMT can discover your plugins by running
the CLI with the relevant flag. For example if you want to see which model components
are available in the current installation you can print them by executing:

.. code-block:: shell

    $ hydromt --components
    Component plugins:
            - ConfigComponent (hydromt 1.0.0a0)
            - DatasetsComponent (hydromt 1.0.0a0)
            - GeomsComponent (hydromt 1.0.0a0)
            - GridComponent (hydromt 1.0.0a0)
            - MeshComponent (hydromt 1.0.0a0)
            - SpatialDatasetsComponent (hydromt 1.0.0a0)
            - TablesComponent (hydromt 1.0.0a0)
            - VectorComponent (hydromt 1.0.0a0)

This will show you all the components that HydroMT currently knows about, what package
they were imported from and which version.

After your plugin classes are properly detected, you can ask HydroMT to access them in
your scripts and yml workflow files. For more information on the specific entrypoints
and how they should be implemented, see each of the corresponding sections.
