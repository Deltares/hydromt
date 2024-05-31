
.. _migration:

###############
Migrating to v1
###############

As part of the move to v1, several things have been changed that will need to be
adjusted in plugins or applications when moving from a pre-v1 version.
In this document, the changes that will impact downstream users and how to deal with
the changes will be listed. For an overview of all changes please refer to the
changelog.

Model and ModelComponent
========================

New `ModelComponent` class
--------------------------

**Rationale**

Prior to v1, the `Model` class was the only real place where developers could
modify the behavior of Core through either sub-classing it, or using various
`Mixin` classes. All parts of a model were implemented as class properties
forcing every model to use the same terminology. While this was enough for
some users, it was too restrictive for others. For example, the SFINCS
plugin uses multiple grids for its computation, which was not possible in
the setup pre-v1. There was also a lot of code duplication for the use of
several parts of a model such as `maps`, `forcing` and `states`. To offer
users more modularity and flexibility, as well as improve maintainability, we
have decided to move the core to a component based architecture rather than
an inheritance based one.

**Changes required**

For users of HydroMT, the main change is that the `Model` class is now a composition of
`ModelComponent` classes. The core `Model` class does not contain any components by default,
but these can be added by the user upon instantiation or through the yaml configuration file.
Plugins will define their implementation of the `Model` class with the components they need.

For a model which is instantiated with a `GridComponent` component called `grid`, where previously you
would call `model.setup_grid(...)`, you now call `model.grid.create(...)`.
To access the grid component data you call `model.grid.data` instead of `model.grid`.

In the core of HydroMT, the available components are:

+-----------------------+---------------------+--------------------------------------------------------+
| v0.x                  | v1.x                | Description                                            |
+=======================+=====================+========================================================+
| Model.config          | ConfigComponent     | Component for managing model configuration             |
+-----------------------+---------------------+--------------------------------------------------------+
| Model.geoms           | GeomsComponent      | Component for managing 1D vector data                  |
+-----------------------+---------------------+--------------------------------------------------------+
| Model.tables          | TablesComponent     | Component for managing non-geospatial data             |
+-----------------------+---------------------+--------------------------------------------------------+
| -                     | DatasetsComponent   | Component for managing non-geospatial data             |
+-----------------------+---------------------+--------------------------------------------------------+
| Model.maps / Model.forcing / Model.results / Model.states | SpatialDatasetsComponent | Component for managing geospatial data |
+-----------------------+---------------------+--------------------------------------------------------+
| GridModel.grid        | GridComponent       | Component for managing regular gridded data            |
+-----------------------+---------------------+--------------------------------------------------------+
| MeshModel.mesh        | MeshComponent       | Component for managing unstructured grids              |
+-----------------------+---------------------+--------------------------------------------------------+
| VectorModel.vector    | VectorComponent     | Component for managing geospatial vector data          |
+-----------------------+---------------------+--------------------------------------------------------+

Changes to the `yaml` HydroMT configuration file format
-------------------------------------------------------

**Rationale**

Given the changes to the `Model` class and new `ModelComponent` the `yaml` configuration file format has
also been updated. The new format is more flexible and allows configuration of the model object and its
components. Components can be added during instantiation by defining them in the yaml which is useful to
define a model on-the-fly based on existing components. Note that this is however not necessary for plugins
as the available components are defined in their implementation of the `Model` class.

**Changes required**

The first change to the YAML format is that now, at the root of the documents are three keys:

- `modeltype` (optional) details what kind of model is going to be used in the model.
This can currently also be provided only through the CLI, but given that YAML files are very
model specific we've decided to make this available through the YAML file as well.
- `global` is intended for any configuration for the model object itself, here you may override
any default configuration for the components provided by your implementation. Any options mentioned
here will be passed to the `Model.__init__` function
- `steps` should contain a list of function calls. In pre-v1 versions this used to be a dictionary,
but now it has become a list which removes the necessity for adding numbers to the end of function
calls of the same name. You may prefix a component name for the step in a dotted manner,
e.g. `<component>.<method>`, to indicate the function should be called on that component instead of the model.
In general any step listed here will correspond to a function on either the model or one of its components.
Any keys that are listed under a step will be provided to the function call as arguments.

An example of a fictional Wflow YAML file would be:

.. code-block:: yaml

	modeltype: wflow
	global:
		data_libs: deltares_data
		components:
			config:
				filename: wflow_sbm_calibrated.toml
	steps:
		- setup_basemaps:
			region: {'basin': [6.16, 51.84]}
			res: 0.008333
			hydrography_fn: merit_hydro
		- grid.add_data_from_geodataframe:
			vector_fn: administrative_areas
			variables: "id_level1"
		- grid.add_data_from_geodataframe:
			vector_fn: administrative_areas
			variables: "id_level3"
		- setup_reservoirs:
			reservoirs_fn: hydro_reservoirs
			min_area: 1.0
		- write:
			components:
				- grid
				- config
		- geoms.write:
			filename: geoms/*.gpkg
			driver: GPKG


Model region and geo-spatial components
---------------------------------------

**Rationale**

The model region is a very integral part for the functioning of HydroMT. A users can define a geo-spatial
region for the model by specifying a bounding box, a polygon, or a hydrological (sub)basin. In the previous
version of HydroMT, a model could only have one region and it was "hidden" in the `geoms` property data.
Additionally, there was a lot of logic to handle the different ways of specifying a region through the code.

To simplify this, allow for component-specific regions rather than one single model region,
and consolidate a lot of functionality for easier maintenance, we decided to bring all this functionality
together in the `SpatialModelComponent` class. Some components inherit from this base component in order to
provide a `region`, `crs`, and `bounds` attribute. This class is not directly used by regular users, but
is used by the `GridComponent`, `VectorComponent`, `MeshComponent` and `SpatialDatasetsComponent`.
Note that not all spatial components require their own region, but can also use the region of another
component. The model class itself may still have a `region` property, which points to the region of one of
the components, as defined by the user / plugin developer.

**Changes required**

The command line interface no longer supports a `--region` argument.
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

The Model region is no longer part of the `geoms` data. The default path the region is written to is no
longer `/path/to/root/geoms/region.geojson` but is now `/path/to/root/region.geojson`. This behavior can
be modified both from the config file and the python API. Adjust your data and file calls as appropriate.

Another change to mention is that the region methods ``parse_region`` and ``parse_region_value`` are no
longer located in ``workflows.basin_mask`` but in `model.region`. These functions are only relevant
for components that inherit from `SpatialModelComponent`. See `GridComponent` and  `model.processes.grid` on how
to use these functions.

In HydroMT core, we let `GridComponent` inherit from `SpatialModelComponent`. One can call `model.grid.create_from_region`,
which will in turn call `parse_region_x`, based on the kind of region it receives.

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

Removing support for `ini` and `toml` HydroMT configuration files
-----------------------------------------------------------------

**Rationale**
To keep a consistent experience for our users we believe it is best to offer a single
format for configuring HydroMT, as well as reducing the maintenance burden on our side.
We have decided that YAML suits this use case the best. Therefore we have decided to
deprecate other config formats for configuring HydroMT. Writing model config files
to other formats will still be supported, but HydroMT won't be able to read them
subsequently. From this point on YAML is the only supported format to configure HydroMT.

**Changes required**

Convert any model config files that are still in `ini` or `toml` format to their
equivalent YAML files. This can be done with manually or any converter, or by reading
and writing it through the standard Python interfaces.

Implementing Model Components (for developers)
----------------------------------------------

Here we will describe the specific changes needed to use a `Model` object.
The changes necessary to have core recognize your plugins are described below.
Now a `Model` is made up of several `ModelComponent` classes to which it can delegate work.
While it should still be responsible for workloads that span multiple components
it should delegate work to components whenever possible. For specific changes needed
for appropriate components see their entry in this migration guide, but general
changes will be described here.

Components are objects that the `Model` class can delegate work to. Typically,
they are associated with one object such as a grid, forcing or tables.
To be able to work within a `Model` class properly a `ModelComponent` must implement
the following methods:

- `read`: reading the component and its data from disk.
- `write`: write the component in its current state to disk in the provided root.

Additionally, it is highly recommended to also provide the following methods to ensure
HydroMT can properly handle your objects:

- `set`: add or overwrite data in the component.
- `_initialize`: initializing an empty component.

Finally, you can provide additional functionality by providing the following optional functions:

- `create`: the ability to construct the schematization of the component from the provided arguments.
  e.g. computation units like grid cells, mesh1d or network lines, vector units for lumped model etc.
- `add_data`: the ability to transform and add model data and parameters to the component once the
  schematization is well-defined (i.e. add land-use data to grid or mesh etc.).

Additionally, we encourage some best practices to be aware of when implementing a components:

- Make sure that your component calls `super().__init__(model=model)` in the `__init__` function
  of your component. This will make sure that references such as `self.logger` and `self.root` are
  registered properly so you can access them.
- Your component should take some variation of a `filename` argument in its `__init__` function that
  is either required or provides a default that is not `None`. This should be saved as an attribute
  and be used for reading and writing when the user does not provide a different path as an argument
  to the read or write functions. This allows developers, plugin developers and users alike to both
  provide sensible defaults as well as the opportunity to overwrite them when necessary.

It may additionally implement any necessary functionality. Any implemented functionality should be
available to the user when the plugin is loaded, both from the Python interpreter as well as the
`yaml` file interface. However, to add some validation, functions that are intended to be called from
the yaml interface need to be decorated with the `@hydromt_step` decorator like below.
This decorator can be imported from the root of core.

.. code-block:: python
	@hydromt_step
	def write(self, ...) -> None:
		pass

When implementing a component, you should inherit from the `ModelComponent` class. When you do this,
not only will it provide some additional validation that you have implemented the correct functions,
but your components will also gain access to the following attributes:

+----------------+---------------------------------------------------------------------------------------------------+------------------------------------------+
| Attribute name | Description                                                                                       | Example                                  |
+================+===================================================================================================+==========================================+
| model          | A reference to the model containing the component which can be used to retrieve other components  | self.model.get_component(...)            |
+----------------+---------------------------------------------------------------------------------------------------+------------------------------------------+
| data_catalog   | A reference to the model's data catalog which can be used to retrieve data                        | self.data_catalog.get_rasterdataset(...) |
+----------------+---------------------------------------------------------------------------------------------------+------------------------------------------+
| logger         | A reference to the logger of the model                                                            | self.logger.info(....)                   |
+----------------+---------------------------------------------------------------------------------------------------+------------------------------------------+
| root           | A reference to the model root which can be used for permissions checking and determining IO paths | self.root.path                           |
+----------------+---------------------------------------------------------------------------------------------------+------------------------------------------+

As briefly mentioned in the table above, your component will be able to retrieve other components
in the model through the reference it receives. Note that this makes it impractical if not impossible
to use components outside of the model they are assigned to.

Adding Components to a Model
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Components can be added to a `Model` object by using the `model.add_component` function. This function
takes the name of the component, and the TYPE (not an instance) of the component as argument. When these
components are added, they are uninitialized (i.e. empty). You can populate them by calling functions such
as `create` or `read` from the yaml interface or any other means through the interactive Python API.

Once a component has been added, any component (or other object or scope that has access to the model class)
can retrieve necessary components by using the `model.get_component` function which takes the name of the
desired component you wish to retrieve. At this point you can do with it as you please.

A developer can defined its own new component either by inheriting from the base `ModelComponent` or from
another one (e.g, `class SubgridComponent(GridComponent)`). The new components can be accessed and discovered
through the `PLUGINS` architecture of HydroMT similar to Model plugins. See the related paragraph for more details.

The `Model.__init__` function can be used to add default components by plugins like so:

.. code-block:: python

	class ExampleModel(Model):
		def __init__(self):
			super().__init__(...)
			self.add_component("grid", GridComponent(self))

	# or

	class ExampleModel(Model):
		def __init__(self):
			super().__init__(..., components={"grid": GridComponent(self}))


If you want to allow your plugin user to modify the root and update or add new component during instantiation
then you can use:

.. code-block:: python

	class ExampleEditModel(Model):
		def __init__(
			self,
			components: Optional[Dict[str, Any]] = None,
			root: Optional[str] = None,
		):
			# Recursively update the components with any defaults that are missing in
			# the components provided by the user.
			components = components or {}
			default_components = {
				"grid": {"type": "GridComponent"},
			}
			components = hydromt.utils.deep_merge.deep_merge(
				default_components, components
			)

			# Now instantiate the Model
			super().__init__(
				root = root,
				components = components,
			)


**SpatialModelComponent**

The region of a `SpatialModelComponent` can either be derived directly from its own component or based on
another referenced component (e.g. a forcing component for which the reference region can be taken from the
grid component). For `SpatialModelComponent` that can derive their own region, it is up to the developer
of the subclass to define how to derive the region from the component `data` by implementing the
`_region_data` property.

The `Model` also contains a property for `region`. That property only works if there is a
`SpatialModelComponent` in the model. If there is only one `SpatialModelComponent`, that component
is automatically detected as the `region`. If there are more than one, the `region_component` can be
specified in the `global` section of the yaml file. If there are no `SpatialModelComponent`s in the model,
the `region` property will error. You can specify this in the configuration as follows:

.. code-block:: yaml

	global:
		region_component: grid
		components:
			grid:
				type: GridComponent  # or any other component that inherits from SpatialModelComponent

The alternative is to specify the region component reference in python, which is useful for plugin developers:

.. code-block:: python

	class ExampleModel(Model):
		def __init__(self):
			super().__init__(region_component="grid", components={"grid": {"type": "GridComponent"}})



**GridComponent**

The `GridMixin` and `GridModel` have been restructured into one `GridComponent` with only
a weak reference to one general `Model` instance. The `set_grid`, `write_grid`,
`read_grid`, and `setup_grid` have been changed to the more generically named `set`,
`write`, `read`, and `create` methods respectively. Also, the `setup_grid_from_*`
methods have been changed to `add_data_from_*`. The functionality of the GridComponent
has not been changed compared to the GridModel.

+------------------------------+-------------------------------------------+
| v0.x                         | v1                                        |
+==============================+===========================================+
| model.set_grid(...)          | model.grid.set(...)                       |
+------------------------------+-------------------------------------------+
| model.read_grid(...)         | model.grid.read(...)                      |
+------------------------------+-------------------------------------------+
| model.write_grid(...)        | model.grid.write(...)                     |
+------------------------------+-------------------------------------------+
| model.setup_grid(...)        | model.grid.create_from_region(...)        |
+------------------------------+-------------------------------------------+
| model.setup_grid_from_*(...) | model.grid.add_data_from_*(...)           |
+------------------------------+-------------------------------------------+

**VectorComponent**

The `VectorMixin` and `VectorModel` have been restructured into one `VectorComponent` with only
a weak reference to one general `Model` instance. The `set_vector`, `write_vector`,
and `read_vector` have been changed to the more generically named `set`,
`write`, and `read` methods respectively. Also, the `setup_vector_from_*`
methods have been changed to `add_data_from_*`. The functionality of the VectorComponent
has not been changed compared to the VectorModel.

+------------------------------+-------------------------------------------+
| v0.x                         | v1                                        |
+==============================+===========================================+
| model.set_vector(...)        | model.vector.set(...)                    |
+------------------------------+-------------------------------------------+
| model.read_vector(...)       | model.vector.read(...)                    |
+------------------------------+-------------------------------------------+
| model.write_vector(...)      | model.vector.write(...)                   |
+------------------------------+-------------------------------------------+

**MeshComponent**

The MeshModel has just like the `GridModel` been replaced with its implementation
of the `ModelComponent`: `MeshComponent`. The restructuring of `MeshModel` follows the same pattern
as the `GridComponent`.

+--------------------------------+-------------------------------------------+
| v0.x                           | v1                                        |
+================================+===========================================+
| model.set_mesh(...)            | model.mesh.set(...)                       |
+--------------------------------+-------------------------------------------+
| model.read_mesh(...)           | model.mesh.read(...)                      |
+--------------------------------+-------------------------------------------+
| model.write_mesh(...)          | model.mesh.write(...)                     |
+--------------------------------+-------------------------------------------+
| model.setup_mesh(...)          | model.mesh.create_2d_from_region(...)     |
+--------------------------------+-------------------------------------------+
| model.setup_mesh2d_from_*(...) | model.mesh.add_2d_data_from_*(...)        |
+--------------------------------+-------------------------------------------+

**TablesComponent**

The previous `Model.tables` is now replaces by a `TablesComponent` that can used to store several
non-geospatial tabular data into a dictionary of pandas DataFrames. The `TablesComponent` for now
only contains the basic methods such as `read`, `write` and `set`.

**GeomsComponent**

The previous `Model.geoms` is now replaced by a `GeomsComponent` that can be used to store several
geospatial geometry based data into a dictionary of geopandas GeoDataFrames. The `GeomsComponent`
for now only contains the basic methods such as `read`, `write` and `set`.

**DatasetsComponent and SpatialDatasetsComponent**

The previous `Model` attributes `forcing`, `states`, `results` and `maps` are now replaced by
a `DatasetsComponent` and a `SpatialDatasetsComponent` that can be used to store several xarray datasets
into a dictionary. If your component should have a region property (in reference to another component),
the component should inherit from `SpatialModelComponent`.

The `DatasetsComponent` for now only contains the basic methods such as `read`, `write` and `set`.
The `SpatialModelComponent` contains additional methods to ``add_raster_data_from`` rasterdataset
and rasterdataset reclassification.

**ConfigComponent**

What was previously called `model.config` as well as some other class variables such as `Model._CONF`
is now located in `ConfigComponent`. Otherwise it still works mostly identically, meaning that it will
parse dotted keys like `a.b.c` into nested dictionaries such as `{'a':{'b':{'c': value}}}`. By default
the data will be read from and written to `<root>/config.yml` which can be overwritten either by providing
different arguments or by sub-classing the component and providing a different default value.

One main change is that the `model.config` used to be created by default from a template file which was
usually located in `Model._DATADIR//Model._NAME//Model._CONF`. To create a config from a template, users
now need to directly call the new `config.create` method, which is similar to how other components work.
Each plugin can still define a default config file template without sub-classing the `ConfigComponent`
by providing a `default_template_filename` when initializing their `ConfigComponent`.

Removed Model attributes
^^^^^^^^^^^^^^^^^^^^^^^^

Below you will find a summary of the functionalities, features, attributes and other things that were
removed from the `Model` class for v1 and how you can access their new equivalents.

- **api**: The `api` property and its associated attributes such as `_API` were previously provided to
  the plugins to enable additional validation. These have been superseded by the component architecture
  and have therefore been removed. Except in the case of equality checking (which will be covered separately
  below) plugins do not need to access any replacement functionality. All the type checking that was
  previously handled by the `api` property is now performed by the component architecture itself. If you use
  components as instructed they will take care of the rest for you.
- **_MAPS/_GEOMS/etc.**: As most aspects are now handled by the components, their model level attributes
  such as `_GEOMS` or `_MAPS` have been removed. The same functionality/ convention can still be used by
  setting these in the components.
- **_CONF** and **config_fn**: For the same reason, defining default config filename from the Model as been
  removed. To update the default config filename for your plugin/model, you can do so by setting the
  `filename` attribute of the `ConfigComponent` as followed. Similarly, if you would like to allow your user
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
  longer have a `_FOLDERS` attribute and the `Model` will no longer create the folders during model init.
  This was done to provide more flexibility in which folders need to be created and which do not need to be.
  Components should make sure that they create the necessary folders themselves during writing.
- **_CLI_ARGS**: As region and resolution are removed from the command line arguments, this was not needed anymore.
- **deprecated attributes**: all grid related deprecated attributes have been removed (eg dims, coords, res etc.)


DataCatalog
===========

Changes to the data catalog `yaml` file format
----------------------------------------------

With the addition of new classes responsible for different stages of the data
reading phase, see below, the data catalog yaml file is updated accordingly:

.. code-block:: yaml

	mysource:
		data_type: RasterDataset
		uri: meteo/era5_daily/nc_merged/era5_{year}*_daily.nc
		metadata:
			category: meteo
			notes: Extracted from Copernicus Climate Data Store; resampled by Deltares to daily frequency
			crs: 4326
			nodata: -9999
			...
		driver:
			name: netcdf
			filesystem: local
			metadata_resolver: convention
			options:
				chunks:
					latitude: 250
					longitude: 240
					time: 30
				combine: by_coords
		data_adapter:
			rename:
				d2m: temp_dew
				msl: press_msl
				...
			unit_add:
				temp: -273.15
				temp_dew: -273.15
				...
			unit_mult:
				kin: 0.000277778
				kout: 0.000277778
				...

Where there are a few changes from the previous versions:

- `path` is renamed to `uri`
- `driver` is it's own class and can be specified:
	- by string, implying default arguments
	- using a YAML object, with a mandatory `name` plus kwargs.
- `metadata_resolver` hangs under driver and can be specified:
	- by string, implying default arguments
	- using a YAML object, with a mandatory `name` plus kwargs.
- `filesystem` is moved to driver, and can be specified:
	- by string, implying default arguments
	- using a YAML object, with a mandatory `protocol` plus kwargs.
- `unit_add`, `unit_mult`, `rename`, `attrs`, `meta` are moved to `data_adapter`

There is also a script available for migrating your data catalog, available at
`scripts/migrate_catalog_to_v1.py'.


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


Split the responsibilities of the `DataAdapter` into separate classes
---------------------------------------------------------------------

The previous version of the `DataAdapter` and its subclasses had a lot of
responsibilities:
- Validate the input from the `DataCatalog` entry.
- Find the right paths to the data based on a naming convention.
- Deserialize/read many different file formats into python objects.
- Merge these different python objects into one that represent that data source in the
model region.
- Homogenize the data based on the data catalog entry and HydroMT conventions.

In v1, this class has been split into three extendable components:

DataSource
^^^^^^^^^^

The `DataSource` is the python representation of a parsed entry in the `DataCatalog`.
The `DataSource` is responsible for validating the `DataCatalog` entry. It also carries
the `DataAdapter` and `DataDriver` (more info below) and serves as an entrypoint to
the data.
Per HydroMT data type (e.g. `RasterDataset`, `GeoDataFrame`), HydroMT has one
`DataSource`, e.g. `RasterDatasetSource`, `GeoDataFrameSource`.

MetaDataResolver
^^^^^^^^^^^^^^^^

The `MetaDataResolver` takes a single `uri` and the query parameters from the model,
such as the region, or the time range, and returns multiple absolute paths, or `uri`s,
that can be read into a single python representation (e.g. `xarray.Dataset`). This
functionality was previously covered in the `resolve_paths` function. However, there
are more ways than to resolve a single uri, so the `MetaDataResolver` makes this
behavior extendable. Plugins or other code can subclass the Abstract `MetaDataResolver`
class to implement their own conventions for data discovery.
The `MetaDataResolver` is injected into the `Driver` objects and can be used there.

Driver
^^^^^^

The `Driver` class is responsible for deserializing/reading a set of file types, like
a geojson or zarr file, into their python in-memory representations:
`geopandas.DataFrame` or `xarray.Dataset` respectively. To find the relevant files based
on a single `uri` in the `DataCatalog`, a `MetaDataResolver` is used.
The driver has a `read` method. This method accepts a `uri`, a
unique identifier for a single data source. It also accepts different query parameters,
such a the region, time range or zoom level of the query from the model.
This `read` method returns the python representation of the DataSource.
Because the merging of different files from different `DataSource`s can be
non-trivial, the driver is responsible to merge the different python objects coming
from the driver to a single representation. This is then returned from the `read`
method.
Because the query parameters vary per HydroMT data type, the is a different driver
interface per type, e.g. `RasterDatasetDriver`, `GeoDataFrameDriver`.

DataAdapter
^^^^^^^^^^

The `DataAdapter` now has its previous responsibilities reduced to just homogenizing
the data coming from the `Driver`. This means slicing the data to the right region,
renaming variables, changing units, regridding and more. The `DataAdapter` has a
`transform` method that takes a HydroMT data type and returns this same type. This
method also accepts query parameters based on the data type, so there is a single
`DataAdapter` per HydroMT data type.



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

Plugins
=======

Previously the `Model` class was the only entrypoint for providing core with custom behavior.
Now, there are four:

- `Model`: This class is mostly responsible for dispatching function calls and otherwise
   delegating work to components.
- `ModelComponent`. This class provides more specialized functionalities to do with a single
   part of a model such as a mesh or grid.
- `Driver`. This class provides customizable loading of any data source.
- `PredefinedCatalog`. This class provides a way to define a catalog of data sources that
   can be used in the model.

Each of these parts have entry points at their relevant submodules. For example, see how these
are specified in the `pyproject.toml`

.. code-block:: toml
	[project.entry-points."hydromt.components"]
	core = "hydromt.components"

	[project.entry-points."hydromt.models"]
	core = "hydromt.models"

	[project.entry-points."hydromt.drivers"]
	core = "hydromt.drivers"

To have post v1 core recognize there are a few new requirements:
1. The entrypoint exposes a submodule or script which must specify a `__hydromt_eps__` attribute.
2. All objects listed in the `__hydromt_eps__` attribute will be made available as plugins in the relevant category.
   These can only be subclasses of the relevant base classes in core for each category (e.g. `ModelComponent` for components)
