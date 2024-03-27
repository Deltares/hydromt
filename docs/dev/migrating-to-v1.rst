
.. _migration:

Migrating to v1
===============

As part of the move to v1, several things have been changed that will need to be
adjusted in plugins or applications when moving from a pre-v1 version.
In this document, the changes that will impact downstream users and how to deal with
the changes will be listed. For an overview of all changes please refer to the
changelog.

Command line and general API users
----------------------------------

Removing support for `ini` and `toml` model configuration files
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

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

Changed import and hydromt folder structures
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

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


Changes to the format of the yaml interface
-------------------------------------------

The first change to the YAML format is that now, at the root of the documents are three keys:
`modeltype`, `global` and `steps`.
- `modeltype` (optional) details what kind of model is going to be used in the model. This can currently also be provided only through the CLI,
but given that YAML files are very model specific we've decided to make this available through the YAML file as well.
- `global` is intended for any configuration for the model object itself, here you may override any default
configuration for the components provided by your implementation. Any options mentioned here will be passed to the `Model.__init__` function
- `steps` should contain a list of function calls. In pre-v1 versions this used to be a dictionary, but now it has become a list
which removes the necessity for adding numbers to the end of function calls of the same name. You may prefix a component name
for the step in a dotted manner to indicate the function should be called on that component instead of the model. In general any step
listed here will correspond to a function on either the model or one of its components. Any keys that are listed under a step will be
provided to the function call as arguments.

An example of a fictional Wflow YAML file would be:

```yaml
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
```

Data catalog
------------

Removing dictionary-like features for the data catalog
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

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

Model
-----

Moving from an inheritance to composition structure for the Model class
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

**Rationale**

Prior to v1, the `Model` class was the only real place where developers could
modify the behavior of Core through either subclassing it, or using various
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

Here we will describe the specific changes needed to use a `Model` object.
The changes necessary to have core recognize your plugins are described below.
Now a `Model` is made up of several `Component` classes to which it can delegate work.
While it should still be responsible for workloads that span multiple components
it should delegate work to components whenever possible. For specific changes needed
for appropriate components see their entry in this migration guide, but general
changes will be described here.

Implementing Model Components
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Components are objects that the `Model` class can delegate work to. Typically, they are associated with one object such as a grid,
forcing or tables. To be able to work within a `Model` class properly a component must implement the following methods:

- `read`: reading the component and its data from disk.
- `write`: write the component in its current state to disk in the provided root.

Additionally, it is highly recommended to also provide the following methods to ensure HydroMT can properly handle your objects:

- `set`: provide the ability to override the current data in the component.
- `_initialize`: provide the ability to override the current data in the component.

Finally, you can provide additional functionality by providing the following optional functions:

- `create`: the ability to construct the schematization of the component (computation units like grid cells, `mesh1d` or network lines, vector units for lumped model etc.) from the provided arguments.
- `add_data`: the ability to add model data and parameters to the component once the schematization is well-defined (i.e. add land-use data to grid or mesh etc.).

It may additionally implement any necessary functionality. Any implemented functionality should be available to the user when the plugin is loaded, both from the Python interpreter as well as the `yaml` file interface. However, to add some validation, functions that are intended to be called from the yaml interface need to be decorated with the `@hydromt_step` decorator like so:

```python
@hydromt_step
def write(self, ...) -> None:
	pass
```

This decorator can be imported from the root of core. When implementing a component, you should inherit from the core provided class called
`ModelComponent`. When you do this, not only will it provide some additional validation that you have implemented the correct functions,
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

As briefly mentioned in the table above, your component will be able to retrieve other components in the model through the reference it receives. Note that this makes it impractical if not impossible to use components outside of the model they are assigned to.

**Manipulating Components**

Components can be added to a `Model` object by using the `model.add_component` function. This function takes the name of the component, and the TYPE (not an instance) of the component as argument. When these components
are added, they are uninitialized (i.e. empty). You can populate them by calling functions such as `create` or `read` from the yaml interface or any other means through the interactive Python API.

Once a component has been added, any component (or other object or scope that has access to the model class) can retrieve necessary components by using the
`model.get_component` function which takes the name of the desired component and the TYPE of the component you wish to retrieve. At this point you can do
with it as you please.

In the core of HydroMT, the available components are (list or maybe table):
  - `GridComponent` for data on a regular grid
  - etc.

 A user can defined its own new component either by inheriting from the base ``ModelComponent`` or from another one (eg SubgridComponent(GridComponent)). The new components can be accessed and discovered through the `PLUGINS` architecture of HydroMT similar to Model plugins. See the related paragraph for more details.

The `Model.__init__` function can be used to add default components by plugins like so:

```python

class ExampleModel(Model):
	def __init__(self):
        self.root: ModelRoot = ModelRoot(".")
		self.add_component("region", ModelRegionComponent)
		self.add_component("grid", GridComponent)
		...

```

If you want to allow your plugin user to modify the root and update or add new component during instantiation then you can use:

``` python

class ExampleEditModel(Model):
    def __init__(
        self,
        components: Optional[dict[str, dict[str, Any]]] = None,
        root: Optional[str] = None,
    ):        
        # Recursively update the components with any defaults that are missing in the components provided by the user.
        components = components or {}
        default_components = {
            "region": {"type": "ModelRegionComponent"},
            "grid": {"type": GridComponent},
        }
        components = hydromt._utils.deep_merge.deep_merge(
            default_components, components
        )
        
        # Now instantiate the Model
        super().__init__(
            root = root,
            components = components,
        )

Making the model region its own component
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

**Rationale**

The model region is a very integral part for the functioning of HydroMT. Additionally
there was a lot of logic to handle the different ways of specifying a region
through the code. To simplify this, highlight the importance of the model region,
make this part of the code easier to customise and consolidate a lot of functionality
for easier maintenance, we decided to bring all this functionality together in
the `ModelRegionComponent` class. This is a required component for a HydroMT model,
and should contain all functionality necessary to deal with it.


**Changes required**

The Model Region is no longer part of the `geoms` data, which means that you will
need a separate write function in your config file. You can use `region.write` for this.
Additionally the default path the region is written to is no longer
`/path/to/root/geoms/region.geojson` but is now `/path/to/root/region.geojson`.
This behaviour can be modified both from the config file and the python API.
Adjust your data and file calls as appropriate.

Another change to mention is that the region methods ``parse_region`` and
``parse_region_value`` are no longer located in ``workflows.basin_mask`` but in
``model.components.region``. The methods stays however the same, only the import changes.

As alluded to above, since region is no longer part of the `geoms` family, it has
received its own object with appropriate functions to use. These are `region.create`,
`region.read`, `region.write` and `region.set`. These work as expected and similar to
the other components. (which will be described more in detail in this migration
guide later.) For convenience a table with the previous function calls that were
removed and their new equivalent is provided below:


+--------------------------+---------------------------+
| v0.x                     | v1                        |
+==========================+===========================+
| model.setup_region(dict) | model.region.create(dict) |
+--------------------------+---------------------------+
| model.write_geoms()      | model.region.write()      |
+--------------------------+---------------------------+
| model.read_geoms()       | model.region.read()       |
+--------------------------+---------------------------+
| model.set_region(...)    | model.region.set(...)     |
+--------------------------+---------------------------+

GridComponent
^^^^^^^^^^^^^

**Rationale**

In v1 the `GridModel` will no longer exist. Instead we created a `GridComponent`,
which is an implementation of the `ModelComponent` class. The idea is that this gives
users more flexibility with adding components to their model class, for instance multiple
grids. In addition, the `ModelComponent`s improve maintainability of the code and
terminology of the components and their methods.

**Changes**

The `GridMixin` and `GridModel` have been restructured into one `GridComponent` with only
a weak reference to one general `Model` instance. The `set_grid`, `write_grid`,
`read_grid`, and `setup_grid` have been changed to the more generically named `set`,
`write`, `read`, and `create` methods respectively. Also, the `setup_grid_from_*`
methods have been changed to `add_data_from_*`. The functionality of the GridComponent
has not been changed compared to the GridModel.

+------------------------------+-------------------------------------------+
| v0.x                         | v1                                        |
+==============================+===========================================+
| model.set_grid(...)          | model.grid_component.set(...)             |
+------------------------------+-------------------------------------------+
| model.read_grid(...)         | model.grid_component.read(...)            |
+------------------------------+-------------------------------------------+
| model.write_grid(...)        | model.grid_component.write(...)           |
+------------------------------+-------------------------------------------+
| model.setup_grid(...)        | model.grid_component.create(...)          |
+------------------------------+-------------------------------------------+
| model.setup_grid_from_*(...) | model.grid_component.add_data_from_*(...) |
+------------------------------+-------------------------------------------+


Plugins
-------

Previously the `Model` class was the only entrypoint for providing core with custom behavior.
Now, there are three:

- `Model`: This class is mostly responsible for dispatching function calls and otherwise delegating work to components.
- `ModelComponent`. This class provides more specialized functionalities to do with a single part of a model such as a mesh or grid.
- `Driver`. TBC

Each of these parts have entry points at their relevant submodules. For example, see how these are specified in the `pyproject.toml`

```toml
[project.entry-points."hydromt.components"]
core = "hydromt.components"

[project.entry-points."hydromt.models"]
core = "hydromt.models"
```

To have post v1 core recognize there are a few new requirements:
1. There must be a dedicated separate submodule (i.e. a folder with a `__init__.py` file that you can import from) for each of the plugins you want to implement (i.e. components, models and drivers need their own submodule)
2. These submodules must have an `__init__.py` and this file must specify a `__all__` attribute.
3. All objects listed in the `__all__` attribute will be made available as plugins in the relevant category. This means these submodules should not re-export anything that is not a plugin.
4. Though this cannot be enforced in Python, there is a base class for each of the plugin categories in core, which your objects should inherit from, this makes sure that you implement all the relevant functionality.

When you have specified the plugins you wish to make available to core in your `pyproject.toml`, all objects should be made available through a global static object called `PLUGINS`. This object has attributes
for each of the corresponding plugin categories.


DataAdapter
-----------

The previous version of the `DataAdapter` and its subclasses had a lot of
responsabilities:
- Validate the input from the `DataCatalog` entry.
- Find the right paths to the data based on a naming convention.
- Deserialize/read many different file formats into python objects.
- Merge these different python objects into one that represent that data source in the
model region.
- Homogenize the data based on the data catalog entry and HydroMT conventions.

In v1, this class has been split into three extentable components:

DataSource
^^^^^^^^^^

The `DataSource` is the python representation of a parsed entry in the `DataCatalog`.
The `DataSource` is responsable for validating the `DataCatalog` entry. It also carries
the `DataAdapter` and `DataDriver` (more info below) and serves as an entrypoint to
the data.
Per HydroMT data type (e.g. `RasterDataset`, `GeoDataFrame`), HydroMT has one
`DataSource`, e.g. `RasterDatasetSource`, `GeoDataFrameSource`.

MetaDataResolver
^^^^^^^^^^^^^^^^

The `MetaDataResolver` takes a single `uri` and the query parameters from the model,
such as the region, or the timerange, and returns multiple absolute paths, or `uri`s,
that can be read into a single python representation (e.g. `xarray.Dataset`). This
functionality was previously covered in the `resolve_paths` function. However, there
are more ways than to resolve a single uri, so the `MetaDataResolver` makes this
behaviour extendable. Plugins or other code can subclass the Abstract `MetaDataResolver`
class to implement their own conventions for data discovery.
The `MetaDataResolver` is injected into the `Driver` objects and can be used there.

Driver
^^^^^^

The `Driver` class is responsable for deserializing/reading a set of file types, like
a geojson or zarr file, into their python in-memory representations:
`geopandas.DataFrame` or `xarray.Dataset` respectively. To find the relevant files based
on a single `uri` in the `DataCatalog`, a `MetaDataResolver` is used.
The driver has a `read` method. This method accepts a `uri`, a
unique identifier for a single datasource. It also accepts different query parameters,
such a the region, timerange or zoom level of the query from the model.
This `read` method returns the python representation of the DataSource.
Because the merging of different files from different `DataSource`s can be
non-trivial, the driver is responsable to merge the different python objects coming
from the driver to a single representation. This is then returned from the `read`
method.
Because the query parameters vary per HydroMT data type, the is a different driver
interface per type, e.g. `RasterDatasetDriver`, `GeoDataFrameDriver`.

DataAdapter
^^^^^^^^^^^

The `DataAdapter` now has its previous responsabilities reduced to just homogenizing
the data coming from the `Driver`. This means slicing the data to the right region,
renaming variables, changing units, regridding and more. The `DataAdapter` has a
`transform` method that takes a HydroMT data type and returns this same type. This
method also accepts query parameters based on the data type, so there is a single
`DataAdapter` per HydroMT data type.
