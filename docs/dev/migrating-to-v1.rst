
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
| hydromt.raster           | hydromt.gis.raster                   |
+--------------------------+--------------------------------------+


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
data source, we created a more extensive API for this. In order to make sure users's
code remains working consistently and have a clear upgrade path when adding new
variants we have decided to remove the old dictionary like interface.

**Changes required**

Dictionary like features such as `catalog['source']`, `catalog['source'] = data`,
`source in catalog` etc should be removed for v1. Equivalent interfaces have been
provided for each operation so it should be fairly simple. Below is a small table
with their equivalent functions


.. table:: Dictionary translation guide for v1
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

Making the model region it's own component
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

add a chagne
