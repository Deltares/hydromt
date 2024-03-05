
.. _migration:

Migrating to v1
---------------

As part of the move to v1, several things have been changed that will need to be adjusted in plugins or applications when moving from a pre-v1 version.
In this documents the changes that will impact downstream consumers and how to deal with the changes will be listed. For an overview of all changes
please reference the changelog.


Removing dictionary-like features for the data catalog
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Rationale
=========

To be able to support different version of the same data set (for example, data sets that get re-released frequently with updated data) or to be able to take the same data set from multiple data sources (e.g. local if you have it but AWS if you don't) the data catalog has undergone some changes. Now since a catalog entry no longer uniquely identifies one source, (since it can refer to any of the variants mentioned above) it becomes insufficient to be request a data source by string only. Since the dictionary interface in python makes it impossible to add additional arguments when requesting a data source, we created a more extensive API for this. In order to make sure users's code remains working consistently and have a clear upgrade path when adding new variants we have decided to remove the old dictionary like interface.

Changes required
================

Dictionary like features such as `catalog['source']`, `catalog['source'] = data`, `source in catalog` etc should be removed for v1. Equivalent interfaces have been provided for ech opperation so it should be farily simple. Below is a small table with their equivalent functions


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

Removing support for `ini` and `toml` model configuration files
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Rationale
=========
To keep a consistent experience for our users we beleive it is best to offer a single format for configuring HydroMT, as well as reducing the maintenence burden on our side. We have decided that YAML suits this use case the best. Therefore we have decided to deprecate other config formats for configuring HydroMT. Writing simulation config files to other formats will still be supported, but HydroMT won't be able to read them subsequently. From this point on YAML is the only supported format to configure HydroMT

Changes required
================

Convert any model config files that are still in `ini` or `toml` format to their equivalent YAML files. This can be done with any converter, or by reading and writing it through the standard Python interfaces.


Making the model region it's own component
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^


Rationale
=========

The model region is a very integral part for the functioning of HydroMT. Additionally there was a lot of logic to handle the differnt ways of specifying a region throught the code. To simplify this, hilight the importance of the model region, make this part of the code easier to customise and consolidate a lot of functionality for easier maintenence, we decided to bring all this functionality together in the `ModelRegionComponent` class. This is a required component for a HydroMT model, and should contain all functionality necessary to deal with it.


Changes required
================

The Model Region is no longer part of the `geoms` data, which means that you will need a seperate write function in your config file. You can use `region.write` for this.
Additionally the default path the region is written to is no longer `/path/to/root/geoms/region.format` but is now `/path/to/root/region.geojson`. This behaviour can be moddified both from the config file and the python API. We have further more restricted the file format of the model region to `GeoJSON` and the data type to `GeoDataFrame`. `GeoSeries` are allso acceptable but will be converted internally. Other data types or formats are nolonger allowed. Adjust your data and file calls as appropriate.

As aluded to above, since region is no longer part of the `geoms` family, it has recieved it's own object with appropriate functions to use. These are `region.create`, `region.read`, `region.write` and `region.set`. These work as expected and similar to the ther comopnents. (which will be described more indetail in this migration guide later.) For convinience a table with the previous function calls that were removed and their new equivalent is provided below:


+--------------------------+---------------------------+
| v0.x                     | v1                        |
+==========================+===========================+
| model.setup_region(dict) | model.region.create(dict) |
+--------------------------+---------------------------+
| model.write_geoms()      | model.region.write()      |
+--------------------------+---------------------------+
| model.set_region(...)    | model.region.set(...)     |
+--------------------------+---------------------------+
