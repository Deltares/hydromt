
.. _migration_plugin:

###############
Migrating to v1
###############

As part of the move to v1, several things have been changed that will need to be
adjusted in plugins or applications when moving from a pre-v1 version.

In this document, we focus on the changes for plugin developers. Changes that affect users
such as the :ref:`new YAML workflow file format <model_workflow_migration>` or the
:ref:`new DataCatalog format <data_catalog_migration>` are described in
the :ref:`migration guide <migration_guide>` section of the User Guide.

For an overview of all changes please refer to the :ref:`changelog <changelog>`.

Model and ModelComponent
========================

New ``ModelComponent`` class
----------------------------

**Rationale**

Prior to v1, the ``Model`` class was the only real place where developers could
modify the behavior of Core through either sub-classing it, or using various
``Mixin`` classes. All parts of a model were implemented as class properties
forcing every model to use the same terminology. While this was enough for
some users, it was too restrictive for others. For example, the SFINCS
plugin uses multiple grids for its computation, which was not possible in
the setup pre-v1. There was also a lot of code duplication for the use of
several parts of a model such as ``maps``, ``forcing`` and ``states``. To offer
users more modularity and flexibility, as well as improve maintainability, we
have decided to move the core to a component based architecture rather than
an inheritance based one.

**Changes required**

For users of HydroMT, the main change is that the ``Model`` class is now a composition of
``ModelComponent`` classes. The core ``Model`` class does not contain any components by default,
but these can be added by the user upon instantiation or through the yaml configuration file.
Plugins will define their implementation of the ``Model`` class with the components they need.

For a model which is instantiated with a ``GridComponent`` component called ``grid``, where previously you
would call ``model.setup_grid(...)`` or ``model.write_grid(...)``, you now call ``model.grid.create(...)``
or ``model.grid.write(...)``. To access the grid component data you call ``model.grid.data`` instead of ``model.grid``.

In the core of HydroMT, the available components are:

+-----------------------------------------------------------+----------------------------+-----------------------------------------------+
| v0.x                                                      | v1.x                       | Description                                   |
+===========================================================+============================+===============================================+
| Model.config                                              | ConfigComponent            | Component for managing model configuration    |
+-----------------------------------------------------------+----------------------------+-----------------------------------------------+
| Model.geoms                                               | GeomsComponent             | Component for managing 1D vector data         |
+-----------------------------------------------------------+----------------------------+-----------------------------------------------+
| Model.tables                                              | TablesComponent            | Component for managing non-geospatial data    |
+-----------------------------------------------------------+----------------------------+-----------------------------------------------+
| -                                                         | DatasetsComponent          | Component for managing non-geospatial data    |
+-----------------------------------------------------------+----------------------------+-----------------------------------------------+
| Model.maps / Model.forcing / Model.results / Model.states | SpatialDatasetsComponent   | Component for managing geospatial data        |
+-----------------------------------------------------------+----------------------------+-----------------------------------------------+
| GridModel.grid                                            | GridComponent              | Component for managing regular gridded data   |
+-----------------------------------------------------------+----------------------------+-----------------------------------------------+
| MeshModel.mesh                                            | MeshComponent              | Component for managing unstructured grids     |
+-----------------------------------------------------------+----------------------------+-----------------------------------------------+
| VectorModel.vector                                        | VectorComponent            | Component for managing geospatial vector data |
+-----------------------------------------------------------+----------------------------+-----------------------------------------------+

This change however means that you are now free to develop your own model components without having to re-use or adhere to the
terminology of the core model. You can now also only re-use the core components that you actually need!

Check the custom objects implementation guide to see how to:

- :ref:`add components to your model <custom_model>`
- :ref:`define your own custom components <custom_component>`

Changes to the `yaml` HydroMT configuration file format
-------------------------------------------------------
The format of the :ref:`YAML workflow file <model_workflow_migration>` has been updated
and support for **.ini** or **.toml** files has been removed.

Additionally, we have now restricted the ``Model`` and ``ModelComponent`` methods that users
can access via CLI and the YAML workflow file. Only methods decorated with the ``@hydromt_step``
decorator are now accessible. This was done to improve validation and avoid users calling
internal methods that were not intended for public use.

You will therefore need to add the ``@hydromt_step`` decorator to any method (setup, read, write) you want to
expose in your model or component. You can find an example in the :ref:`custom model guide <custom_model>`.


Model region and geo-spatial components
---------------------------------------

**Rationale**

The model region is a very integral part for the functioning of HydroMT. A users can define a geo-spatial
region for the model by specifying a bounding box, a polygon, or a hydrological (sub)basin. In the previous
version of HydroMT, a model could only have one region and it was "hidden" in the ``geoms`` property data.
Additionally, there was a lot of logic to handle the different ways of specifying a region through the code.

To simplify this, allow for component-specific regions rather than one single model region,
and consolidate a lot of functionality for easier maintenance, we decided to bring all this functionality
together in the ``SpatialModelComponent`` class. Some components inherit from this base component in order to
provide a ``region``, ``crs``, and ``bounds`` attribute. This class is not directly used by regular users, but
is used by the ``GridComponent``, ``VectorComponent``, ``MeshComponent`` and ``SpatialDatasetsComponent``.
Note that not all spatial components require their own region, but can also use the region of another
component. The model class itself may still have a ``region`` property, which points to the region of one of
the components, as defined by the user / plugin developer.

**Changes required**

The command line interface no longer supports a ``--region`` argument.
Instead, the region should be specified in the yaml file of the relevant component(s).

.. code-block:: yaml

	# Example of specifying the region component via grid.create_from_region
	global:
		region_component: grid
		components:
			grid:
				type: GridComponent
	steps:
		- grid.create_from_region:
			region:
				basin: [6.16, 51.84]

The Model region is no longer part of the ``geoms`` data. The default path the region is written to is no
longer ``/path/to/root/geoms/region.geojson`` but is now ``/path/to/root/region.geojson``. This behavior can
be modified both from the config file and the python API. Adjust your data and file calls as appropriate.

Another change to mention is that the region methods ``parse_region`` and ``parse_region_value`` are no
longer located in ``workflows.basin_mask`` but in ``model.region``. These functions are only relevant
for components that inherit from ``SpatialModelComponent``. See ``GridComponent`` and  ``model.processes.grid`` on how
to use these functions.

In HydroMT core, we let ``GridComponent`` inherit from ``SpatialModelComponent``. One can call ``model.grid.create_from_region``,
which will in turn call ``parse_region_x``, based on the kind of region it receives.

+--------------------------+-----------------------------------+
| v0.x                     | v1                                |
+==========================+===================================+
| model.setup_region(dict) | model.<component>.create_region() |
+--------------------------+-----------------------------------+
| model.write_geoms()      | model.<component>.write_region()  |
+--------------------------+-----------------------------------+
| model.read_geoms()       | model.<component>.read_region()   |
+--------------------------+-----------------------------------+
| model.set_region(...)    | -                                 |
+--------------------------+-----------------------------------+
| parse_region             | parse_region_basin                |
|                          | parse_region_geom                 |
|                          | parse_region_bbox                 |
|                          | parse_region_other_model          |
|                          | parse_region_grid                 |
|                          | parse_region_mesh                 |
+--------------------------+-----------------------------------+

Setup methods from core
-----------------------
Some of the setup methods that were previously part of the ``Model`` class or of the
``GridModel`` and ``MeshModel`` subclasses have been removed and were not migrated to the
new ``GridComponent`` or ``MeshComponent``. The rationale behind this method is that users
may not wish to always inherit all these methods when re-using core components.

However, the functionality of these methods has not been removed, and is still fully available in the
``hydromt.model.processes.grid`` and ``hydromt.model.processes.mesh`` submodules. So if you do
wish to keep the same functionality as before, you can simply add a new setup method in your
custom model or components, start by getting data from the data catalog, call the relevant
function from the processes submodule, and set the data to your component.


Creating default config files
-----------------------------

One main change is that the ``model.config`` used to be created by default from a template file which was
usually located in `Model._DATADIR//Model._NAME//Model._CONF`. To create a config from a template, users
now need to directly call the new ``config.create`` method, which is similar to how other components work.
Each plugin can still define a default config file template without sub-classing the ``ConfigComponent``
by providing a ``default_template_filename`` when initializing their ``ConfigComponent``.

Removed Model attributes
------------------------

Below you will find a summary of the functionalities, features, attributes and other things that were
removed from the ``Model`` class for v1 and how you can access their new equivalents.

- **api**: The ``api`` property and its associated attributes such as ``_API`` were previously provided to
  the plugins to enable additional validation. These have been superseded by the component architecture
  and have therefore been removed. Except in the case of equality checking (which will be covered separately
  below) plugins do not need to access any replacement functionality. All the type checking that was
  previously handled by the ``api`` property is now performed by the component architecture itself. If you use
  components as instructed they will take care of the rest for you.
- **_MAPS/_GEOMS/etc.**: As most aspects are now handled by the components, their model level attributes
  such as ``_GEOMS`` or ``_MAPS`` have been removed. The same functionality/ convention can still be used by
  setting these in the components.
- **_CONF** and **config_fn**: For the same reason, defining default config filename from the Model as been
  removed. To update the default config filename for your plugin/model, you can do so by setting the
  ``filename`` attribute of the ``ConfigComponent`` as followed. Similarly, if you would like to allow your user
  to easily update the model config file, you can re-add the **config_fn** in your model plugin:

.. code-block:: python

	class MyModel(Model):
	...
	def __init__(self, config_filename: Optional[str] = None):
		...
		# Add the config component
		if config_filename is None:
			config_filename = "my_plugin_default_config.toml"
		config_component = ConfigComponent(self, filename=config_filename)
		self.add_component("config", config_component)

- **_FOLDERS**: Since the components are now responsible for creating their folders when writing, we no
  longer have a ``_FOLDERS`` attribute and the ``Model`` will no longer create the folders during model init.
  This was done to provide more flexibility in which folders need to be created and which do not need to be.
  Components should make sure that they create the necessary folders themselves during writing.
- **_CLI_ARGS**: As region and resolution are removed from the command line arguments, this was not needed anymore.
- **deprecated attributes**: all grid related deprecated attributes have been removed (eg dims, coords, res etc.)


DataCatalog
===========

Changes to the data catalog file format
---------------------------------------
See the detailed section in the :ref:`data catalog migration guide <data_catalog_migration>`.


Removing dictionary-like features for the DataCatalog
-----------------------------------------------------

**Rationale**

To be able to support different version of the same data set (for example, data sets
that get re-released frequently with updated data) or to be able to take the same data
set from multiple data sources (e.g. local if you have it but AWS if you don't) the
data catalog has undergone some changes. Now since a catalog entry no longer uniquely
identifies one source, (since it can refer to any of the variants mentioned above) it
becomes insufficient to request a data source by string only. Since the dictionary
interface in python makes it impossible to add additional arguments when requesting a
data source, we created a more extensive API for this. In order to make sure users'
code remains working consistently and have a clear upgrade path when adding new
variants we have decided to remove the old dictionary like interface.

**Changes required**

Dictionary like features such as `catalog['source']`, `catalog['source'] = data`,
`source in catalog` etc. should be removed for v1. Equivalent interfaces have been
provided for each operation, so it should be fairly simple. Below is a small table
with their equivalent functions


..table:: Dictionary translation guide for v1
   :widths: auto

+--------------------------+--------------------------------------+
| v0.x                     | v1                                   |
+==========================+======================================+
| if 'name' in catalog:    | if catalog.contains_source('name'):  |
+--------------------------+--------------------------------------+
| catalog['name']          | catalog.get_source('name')           |
+--------------------------+--------------------------------------+
| for x in catalog.keys(): | for x in catalog.get_source_names(): |
+--------------------------+--------------------------------------+
| catalog['name'] = data   | catalog.set_source('name',data)      |
+--------------------------+--------------------------------------+


Split the responsibilities of the ``DataAdapter`` into separate classes
-----------------------------------------------------------------------

The previous version of the ``DataAdapter`` and its subclasses had a lot of
responsibilities:
- Validate the input from the ``DataCatalog`` entry.
- Find the right paths to the data based on a naming convention.
- Deserialize/read many different file formats into python objects.
- Merge these different python objects into one that represent that data source in the
model region.
- Homogenize the data based on the data catalog entry and HydroMT conventions.

In v1, this class has been split into three extendable components:

DataSource
^^^^^^^^^^

The ``DataSource`` is the python representation of a parsed entry in the ``DataCatalog``.
The ``DataSource`` is responsible for validating the ``DataCatalog`` entry. It also carries
the ``DataAdapter`` and ``DataDriver`` (more info below) and serves as an entrypoint to
the data.
Per HydroMT data type (e.g. ``RasterDataset``, ``GeoDataFrame``), HydroMT has one
``DataSource``, e.g. ``RasterDatasetSource``, ``GeoDataFrameSource``.

URIResolver
^^^^^^^^^^^

The ``URIResolver`` takes a single ``uri`` and the query parameters from the model,
such as the region, or the time range, and returns multiple absolute paths, or ``uri``s,
that can be read into a single python representation (e.g. ``xarray.Dataset``). This
functionality was previously covered in the ``resolve_paths`` function. However, there
are more ways than to resolve a single uri, so the ``URIResolver`` makes this
behavior extendable. Plugins or other code can subclass the Abstract ``URIResolver``
class to implement their own conventions for data discovery.
The ``URIResolver`` is injected into the ``Driver`` objects and can be used there.

Driver
^^^^^^

The ``Driver`` class is responsible for deserializing/reading a set of file types, like
a geojson or zarr file, into their python in-memory representations:
``geopandas.DataFrame`` or ``xarray.Dataset`` respectively. To find the relevant files based
on a single ``uri`` in the ``DataCatalog``, a ``URIResolver`` is used.
The driver has a ``read`` method. This method accepts a ``uri``, a
unique identifier for a single data source. It also accepts different query parameters,
such a the region, time range or zoom level of the query from the model.
This ``read`` method returns the python representation of the DataSource.
Because the merging of different files from different ``DataSource``s can be
non-trivial, the driver is responsible to merge the different python objects coming
from the driver to a single representation. This is then returned from the ``read``
method.
Because the query parameters vary per HydroMT data type, the is a different driver
interface per type, e.g. ``RasterDatasetDriver``, ``GeoDataFrameDriver``.

DataAdapter
^^^^^^^^^^^

The ``DataAdapter`` now has its previous responsibilities reduced to just homogenizing
the data coming from the ``Driver``. This means slicing the data to the right region,
renaming variables, changing units and more. The ``DataAdapter`` has a
``transform`` method that takes a HydroMT data type and returns this same type. This
method also accepts query parameters based on the data type, so there is a single
``DataAdapter`` per HydroMT data type.



Package API
===========

**Rationale**

As HydroMT contains many functions and new classes with v1, the hydromt folder structure
and the import statements have changed.

**Changes required**

The following changes are required in your code:

+--------------------------+--------------------------------------+
| v0.x                     | v1                                   |
+==========================+======================================+
| hydromt.config           | Removed                              |
+--------------------------+--------------------------------------+
| hydromt.log              | Removed (private: hydromt._utils.log)|
+--------------------------+--------------------------------------+
| hydromt.flw              | hydromt.gis.flw                      |
+--------------------------+--------------------------------------+
| hydromt.gis_utils        | hydromt.gis.utils                    |
+--------------------------+--------------------------------------+
| hydromt.raster           | hydromt.gis.raster                   |
+--------------------------+--------------------------------------+
| hydromt.vector           | hydromt.gis.vector                   |
+--------------------------+--------------------------------------+
| hydromt.gis_utils        | hydromt.gis.utils                    |
+--------------------------+--------------------------------------+
| hydromt.io               | hydromt.readers and hydromt.writers  |
+--------------------------+--------------------------------------+

Logging
=======

**Rationale**

Previous versions of HydroMT passed the logger around a lot. The Logging module is based on
singleton classes and log propagation using a naming convention. Due to some bugs in the
previous version of the code, the logger passing has been removed and the ``"hydromt"`` logger
now governs the logging output.

**Changes required**

Remove the ``logger`` keywords from your HydroMT functions, methods and classes. If you want to
influence HydroMT logging, change the ``"hydromt"`` logger: ``logger = logging.getLogger("hydromt")``.

Plugins
=======

Previously the ``Model`` class was the only entrypoint for providing core with custom behavior.
Now, there are four:

- ``Model``: This class is mostly responsible for dispatching function calls and otherwise delegating work to components.
- ``ModelComponent``. This class provides more specialized functionalities to do with a single part of a model such as a mesh or grid.
- ``Driver``. This class provides customizable loading of any data source.
- ``PredefinedCatalog``. This class provides a way to define a catalog of data sources that can be used in the model.

Each of these parts have entry points at their relevant submodules. For example, see how these
are specified in the ``pyproject.toml``

See the guide on :ref:`registering your custom objects <register_plugins>` for more information and examples.
