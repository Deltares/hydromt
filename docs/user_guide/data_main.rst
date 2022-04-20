.. _get_data:

Working with data in HydroMT  
============================

The best way to provide data to HydroMT is by using a **data catalog**. The goal of this 
data catalog is to provide simple and standardized access to (large) datasets. 
It supports many drivers to read different data formats and contains several pre-processing steps to unify the datasets. 
A data catalog can be initialized from one or more **yaml file(s)**, which contain all required information to read and pre-process a dataset, 
as well as meta data for reproducibility.

You can :ref:`explore and make use of pre-defined data catalogs <existing_catalog>` (primarily global data), 
:ref:`prepare your own Data Catalog <own_catalog>` (e.g. to include local data) or use a combination of both. 

.. _SupportedDataset: 

Supported data types
^^^^^^^^^^^^^^^^^^^^

HydroMT currently supports the following data types:

- :ref:`RasterDataset <RasterDataset>`: static and dynamic raster (or gridded) data 
- :ref:`GeoDataFrame <GeoDataFrame>`: static vector data 
- :ref:`GeoDataset <GeoDataset>`: dynamic point location data

Internally the RasterDataset and GeoDataset are represented by :py:class:`xarray.Dataset` objects 
and GeoDataFrame by :py:class:`geopandas.GeoDataFrame`. We use drivers, typically from third-party
packages and sometimes wrapped in HydroMT functions, to parse many different file formats to this 
standardized internal data representation. 

An overview of the supported data formats and associated drivers and arguments are shown in the 
:ref:`Preparing a Data Catalog <own_catalog>` section.

.. note::

    Tabulated data without a spatial component such as mapping tables are planned to be added. 
    Please contact us through the issue list if you would like to add other drivers.

.. _get_data_python: 

Python usage 
^^^^^^^^^^^^

To read a dataset in Python using the HydroMT requires two steps:

 - Initialize a :py:class:`~hydromt.data_adapter.DataCatalog` with references to user- or pre-defined data catalog yaml files
 - Use :ref:`one of the get_* methods <api_data_catalog_get>` to access (a temporal or spatial region of) the data.

For example to retrieve a raster dataset:

.. code-block:: python

    import hydromt
    data_cat = hydromt.DataCatalog(data_libs=r'/path/to/data-catalog.yml')
    ds = data_cat.get_rasterdataset('source_name', bbox=[xmin, ymin, xmax, ymax])  # returns xarray.dataset

More detail about working with `raster data  <../_examples/read_raster_data.ipynb>`_, vector data or geospatial time-series data is provided in the linked examples.

.. _get_data_cli: 

Command line usage 
^^^^^^^^^^^^^^^^^^

When using the HydroMT command line interface, one can provide data catalog by specifying the
path to the yaml file with the ``-d`` or ``--data`` option. 
Multiple yaml files can be added by reusing the ``-d`` option.

For example when using the build method:

.. code-block:: console

    hydromt build MODEL REGION -d /path/to/data_catalog1.yml -d /path/to/data_catalog1.yml

A special exception is made for the Deltares data catalog which can be accessed with the 
``--dd`` or ``--deltares-data`` flag (requires access to the Deltares P-drive).

.. code-block:: console

    hydromt build MODEL REGION -dd


.. hint::

    If no yaml file is provided to the CLI build or update methods or to :py:class:`~hydromt.data_adapter.DataCatalog`, 
    HydroMT will use the data stored in the `hydromt-artifacts <https://github.com/DirkEilander/hydromt-artifacts>`_ 
    which contains an extract of global data for a small region around the Piave river in Northern Italy.

Related API references
^^^^^^^^^^^^^^^^^^^^^^

For related functions see:

 - :ref:`DataCatalog API <api_data_catalog>`
 - :ref:`get_* methods <api_data_catalog_get>`
 - :ref:`data reading-methods <open_methods>`


.. toctree::
    :hidden:

    data_existing_cat.rst
    data_prepare_cat.rst
    data_conventions.rst
    Example: Reading raster data <../_examples/read_raster_data.ipynb>