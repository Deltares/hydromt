.. _python_updates_v1:

Updates for python users
========================
In HydroMT v1, the internal data structure and API were redesigned to improve consistency and maintainability.
Most changes affect how model components (such as ``config`` and ``grid``) are accessed and how model data is read and written.

Component Class
---------------

Rationale
^^^^^^^^^
Prior to v1, the ``Model`` class was the only real place where developers could
modify the behavior of Core. All parts of a model were implemented as class properties
forcing every model to use the same terminology. While this was enough for
some users, it was too restrictive for others. For example, the SFINCS
plugin uses multiple grids for its computation, which was not possible in
the setup pre-v1. There was also a lot of code duplication for the use of
several parts of a model such as ``maps``, ``forcing`` and ``states``. To offer
users more modularity and flexibility, as well as improve maintainability, we
have decided to move the core to a component based architecture rather than
an inheritance based one.

The model components are now **dedicated classes** rather than raw data objects (e.g., ``xarray``, ``dict``, or ``geopandas``).
Each component can be accessed via the ``Model`` instance and exposes its underlying data through the ``.data`` property.

For users of HydroMT, the main change is that the ``Model`` class is now a composition of
``ModelComponent`` classes. The core ``Model`` class does not contain any components by default,
but these can be added by the user upon instantiation or through the yaml configuration file.
Plugins will define their implementation of the ``Model`` class with the components they need.

For a model which is instantiated with a ``GridComponent`` component called ``grid``, where previously you
would call ``model.setup_grid(...)``, you now call ``model.grid.create(...)``.
To access the grid component data you call ``model.grid.data`` instead of ``model.grid``.

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

Example: Accessing Component Data
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Each component provides structured access to its data via the ``.data`` property.

.. code-block:: python

    from hydromt import ExampleModel

    model = ExampleModel(root="path/to/model", mode="r")

    # Access xarray.Dataset of static grid
    grid = model.grid.data

    # Access configuration dictionary
    config = model.config.data

Example: Writing Components
^^^^^^^^^^^^^^^^^^^^^^^^^^^

Read and write operations are now handled at the **component level**.

.. code-block:: python

    # Write configuration file
    model.config.write()

    # Write updated grid to disk
    model.grid.write()


These changes provide a clearer and more modular interface, making it easier to manipulate model components independently.

DataCatalog
-----------

Rationale
^^^^^^^^^
The data catalog structure has been refactored to introduce a more modular design and
clearer separation of responsibilities across several new classes (``DataSource``, ``Driver``, ``URIResolver``, and ``DataAdapter``):

- ``URIResolver`` is in charge of parsing the path or URI of the file (e.g if you are using some keywords like
  ``{year}`` or ``{month}`` in your paths or if you want to read tiled raster)
- ``Driver`` is in charge of reading the data from the source (e.g reading a netcdf file from a local disk or from cloud)
- ``DataAdapter`` is in charge of harmonizing the data to standard HydroMT data structures (e.g. renaming variables, setting attributes, units conversion, etc.)
- ``DataSource`` is the main class that ties everything together and is used by the ``DataCatalog`` to load data.


Additionally, to be able to support different version of the same data set (for example, data sets
that get re-released frequently with updated data) or to be able to take the same data
set from multiple data sources (e.g. local if you have it but AWS if you don't) the
data catalog has undergone some changes. Now since a catalog entry no longer uniquely
identifies one source, (since it can refer to any of the variants mentioned above) it
becomes insufficient to request a data source by string only. Since the dictionary
interface in python makes it impossible to add additional arguments when requesting a
data source, we created a more extensive API for this. In order to make sure users'
code remains working consistently and have a clear upgrade path when adding new
variants we have decided to remove the old dictionary like interface.
Dictionary like features such as `catalog['source']`, `catalog['source'] = data`,
`source in catalog` etc. should be removed for v1. Equivalent interfaces have been
provided for each operation, so it should be fairly simple. Below is a small table
with their equivalent functions.

How to upgrade
^^^^^^^^^^^^^^
The high levels functions of ``DataCatalog`` and the different ``get_data`` methods have not changed
apart that the ``Driver`` and ``DataAdapter`` options have to be specified differently in the catalog yaml file or
explicitly under ``source_kwargs`` when calling the ``get_data`` methods.
However, the dictionary-like interface has been removed.

.. code-block:: python

    import hydromt
    catalog = hydromt.DataCatalog('path_to_your_catalog.yml')

    # Old v0.x way (removed)
    gdf = catalog.get_geodataframe('path/to/locations.csv', driver_kwargs={'sep': ';'})

    # New v1.x way
    gdf = catalog.get_geodataframe(
        'path/to/locations.csv',
        source_kwargs={
          'driver': {'name': 'pandas', 'options': {'sep': ';'}}
        }
    )


Below is a table with the equivalent functions for the removed dictionary-like interface:

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


API and supporting functions
----------------------------
HydroMT provides a series of supporting functions either for GIS operations, statistical methods,
or file specific I/O operations. These functions have been moved and/or grouped under specific submodules
to improve discoverability and maintainability.

Main changes includes:

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
