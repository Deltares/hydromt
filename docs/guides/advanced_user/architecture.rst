.. _architecture:

Architecture
============

HydroMT supports a large variety of models, which all require different types of data.
It is therefore important that the API that HydroMT exposes is extendable. HydroMT is
composed of a small set of key classes that support extension. In this section we walk
through these classes and describe their main responsibilities and where they interact.

Model
-----

The :Class:`Model` is the main representation of the model that is being built. A model
is built step by step by adding :Class:`ModelComponent`s to the Model. :ref:`Plugins`
can define steps which act on these components to implement complex interactions between
different components. The area of interest for the model can be defined by the
:Class:`ModelRegion`. The complete model building workflow can be encoded in a
:ref:`model_yaml_setup` file.

ModelComponent
--------------

A :Class:`Model` can be populated with many different :Class:`ModelComponent`s. A
component can represent any type of data you have on your area of interest. This
component can have many properties, but always has a py:method:`ModelComponent.read` and
py:method:`ModelComponent.write` component to read in and write out data.
A :Class:`Model` must have at least one :Class:`ModelComponent`.

ModelRegion
-----------

The :Class:`ModelRegion` defines the area of interest for a certain
:Class:`ModelComponent`. Users can define a geo-spatial region for one or more
:Class:`ModelComponent`s. The model can also define a generic region by referring to a
:Class:`ModelRegion`.

DataCatalog
-----------

:Class:`Model`s need data. Where the data should be found and how it should be loaded is
defined in the :Class:`DataCatalog`. Each item in the catalog is a :Class:`DataSource`.
Users can create their own catalogs, using a `yaml` format, or they can share their
:Class:`PredefinedCatalog` using the :ref:`plugins` system.

DataSource
----------

The :class:`DataSource` is the python representation of a parsed entry in the
:class:`DataCatalog`. The source is responsible for validating the catalog entry. It
also carries the :class:`DataAdapter`, :class:`URIResolver` and :class:`Driver` and
serves as an entrypoint to the data. Per HydroMT data type (e.g. `RasterDataset`,
`GeoDataFrame`), HydroMT has one :Class:`DataSource`, e.g. :Class:`RasterDatasetSource`,
:Class:`GeoDataFrameSource`. The py:method:`DataSource.read` method governs the full
process of discovery with the :Class:`URIResolver`, reading data with the
:Class:`Driver`, and transforming the data to a HydroMT standard with a
:Class:`DataAdapter`.

URIResolver
-----------

Finding the right address where the requested data is stored is not always
straightforward. Searching for data differs between finding data in a web-service,
database, a catalog or when dealing with a certain naming convention. Exploring where
the right data can be found is implemented in the :Class:`URIResolver`. The
:Class:`URIResolver` takes a single `uri` from the data catalog, and the query
parameters from the model, such as the region, or the time range, and returns multiple
absolute paths, or `uri`s, that can be read into a single python representation (e.g.
`xarray.Dataset`). The :Class:`URIResolver` is extendable, so :ref:`Plugins` or other
code can subclass the Abstract :Class:`URIResolver` class to implement their own
conventions for data discovery.

Driver
------

The :Class:`Driver` class is responsible for reading a set of file types, like a
`geojson` or `zarr`` file, into their python in-memory representations:
`geopandas.DataFrame` or `xarray.Dataset` respectively. This class can also be extended
using the :ref:`plugins`. Because the merging of different files from different
:Class:`DataSource`s can be non-trivial, the driver is responsible to merge the
different python objects coming from the driver to a single representation. This is then
returned from the `read` method. The query parameters vary per HydroMT data type, so
there is is a different driver interface per type, e.g. :Class:`RasterDatasetDriver`,
:Class:`GeoDataFrameDriver`. To help with different filesystems, the driver class is
handed a `fsspec.Filesystem`.

DataAdapter
-----------

The :Class:`DataAdapter` homogenizes the data coming from the :Class:`Driver`. This
means slicing the data to the right region, renaming variables, changing units,
regridding and more. The adapter has a `transform` method that takes a python object and
returns the same type, e.g. an `xr.Dataset`. This method also accepts query parameters
based on the data type, so there is a single :Class:`DataAdapter` per HydroMT data type.

Architecture Diagram
====================

The above is summarized in the following architecture diagram. Only the aforementioned
methods and properties are used.

.. image:: ../../drawio/exported/HydroMT-Architecture-OverArching.drawio.png
    :width: 800
    :alt: HydroMT main classes
