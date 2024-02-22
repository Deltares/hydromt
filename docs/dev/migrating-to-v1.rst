.. _migrating:

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
| Dictionary               | v1                                   |
+==========================+======================================+
| if 'name' in catalog:    | if catalog.contains_source('name'):  |
+--------------------------+--------------------------------------+
| catalog['name']          | catalog.get_source('name')           |
+--------------------------+--------------------------------------+
| for x in catalog.keys(): | for x in catalog.get_source_names(): |
+--------------------------+--------------------------------------+
| catalog['name'] = data   | catalog.set_source('name',data)      |
+--------------------------+--------------------------------------+
