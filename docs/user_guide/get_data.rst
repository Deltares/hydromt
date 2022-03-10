.. _get_data:

Working with data in hydroMT  
============================

The best way to provide data to hydroMT is by using a **data catalog**. The goal of this 
data catalog is to provide simple and standardized access to (large) datasets which are 
parsed to convenient Python objects. It supports many drivers to read different data formats and 
contains several pre-processing steps to unify the datasets. A data catalog can be build from one 
or more yaml files, which contain all required information to read and pre-process a dataset, 
as well as meta data to improve reproducibility.

You can :ref:`explore and make use of pre-defined data catalogs <existing_catalog>` (primarily global data), 
:ref:`prepare your own Data Catalog <own_catalog>` (e.g. to include local data) or use a combination of both. 

.. note::

    Not all data catalogs are openly accessible, but may require you to be connected to a specific network.

.. note::

    If the data catalog is initiating without a reference to a user- or pre-defined data catalog, hydroMT
    will default to the *hydromt-artifacts* data catalog where a small spatial subset of several datasets is
    stored for testing purposes.

.. _SupportedDataset: 

Supported data types
^^^^^^^^^^^^^^^^^^^^

HydroMT currently supports the following data types:

- :ref:`*RasterDataset* <RasterDataset>`: static and dynamic raster (or gridded) data 
- :ref:`*GeoDataFrame* <GeoDataFrame>`: static vector data 
- :ref:`*GeoDataset* <GeoDataset>`: dynamic point location data

Internally the RasterDataset and GeoDataset are represented by :py:class:`xarray.Dataset` objects 
and GeoDataFrame by :py:class:`geopandas.GeoDataFrame`. We use drivers, typically from third-party
packages and sometimes wrapped in hydroMT functions, to parse many different file formats to this 
standardized internal data representation. 

An overview of the supported data formats and associated drivers and arguments are shown in the 
:ref:`Preparing a Data Catalog <own_catalog>` section.

.. note::

    Tabulated data without a spatial component such as mapping tables are planned to be added. 
    Please contact us through the issue list if you would like to add other drivers.

.. _get_data_cli: 

Command line usage 
^^^^^^^^^^^^^^^^^^

When using the hydroMT command line, one can link to the data catalog by specifying the
name of an existing data catalog file or the path to where the yaml file is located with 
the ``-d`` or ``--data`` option. Multiple yaml files can be added by reusing the ``-d`` option.

For example when using the build method:

.. code-block:: console

    hydromt build MODEL REGION -d /path/to/data_catalog1.yml -d /path/to/data_catalog1.yml

A special exception is made for the deltares_data catalog which can be accessed with the 
``--dd`` or ``--deltares-data`` flag.

.. code-block:: console

    hydromt build MODEL REGION -dd

.. note::

    - If no yml file is selected (e.g. for testing purposes), HydroMT will use the data stored in the `hydromt-artifacts <https://github.com/DirkEilander/hydromt-artifacts>`_ which contains an extract of global data for a small region around the Piave river in Northern Italy.
    - For Deltares users is to select the deltares-data library (requires access to the Deltares P-drive). In the command lines examples below, this is done by adding either **-dd** or **--deltares-data** (no path required) to the build / update command line.
    - In all other cases refer to a local yml file by adding -d /path/to/data_catalog.yml in the command line.

.. _get_data_python: 

Python usage 
^^^^^^^^^^^^

With a `DataCatalog` in place **hydroMT** can be used to read the data with the `DataCatalog.getrasterdataset` method.
The use of a `DataCatalog` allows for minimal pre-processing in order to get uniform variable names and units.
See `Reading raster data examples <https://deltares.github.io/hydromt/latest/examples/examples/read_raster_data.html#Reading-raster-data>`_ which highlights 
various commonly used options to read single or multiple file raster datasets into an `xarray.Dataset` or `xarray.DataArray` object with geospatial attributes.

Basic usage to read a dataset in python using the hydroMT data catalog requires two steps:
 - Initialize a DataCatalog with references to user- or pre-defined data catalog files
 - Use one of the get_* methods to access the data.

Example usage to retrieve a raster dataset

.. code-block:: python

    import hydromt
    data_cat = hydromt.DataCatalog(data_libs=r'/path/to/data-catalog.yml')
    ds = data_cat.get_rasterdataset('source_name', bbox=[xmin, ymin, xmax, ymax])  # returns xarray.dataset

First import the necessary libraries and initialize the logger:

.. ipython:: python

    import numpy as np
    import xarray as xr
    from pprint import pprint
    import glob
    import os
    import hydromt
    from hydromt.log import setuplog
    logger = setuplog("read raster data", log_level=10)

Next, read data from the artifacts:

.. ipython:: python

    data_catalog = hydromt.DataCatalog(logger=logger)
    data_catalog.from_artifacts()

Next check the merit_hydro dataset available in and read from the artifacts:

.. ipython:: python

    path = os.path.join(os.path.dirname(data_catalog["merit_hydro"].path), "*.tif")
    fns = glob.glob(path)
    fns

Finally open one of the rasters and check the print statement: 
 
.. ipython:: python

    da = hydromt.open_raster(fns[0], chunks={"x": 1000, "y": 1000})
    print(da)

.. note::
    
    For the `hydromt_delwaq plugin <https://deltares.github.io/hydromt_delwaq/latest/index.html#hydromt-plugin-delwaq>`_ three data types exist, see `emissions <https://deltares.github.io/hydromt_delwaq/latest/api/api_workflows.html#emissions>`_: 
    raster, vector and admin. The admin type is an administrative raster to which related parameters can be mapped (e.g. country or region boundaries can be used to delign different values of sewage connection percentage). 
    A mapping table is required to link the related parameters to the administrative raster.

Related API references
^^^^^^^^^^^^^^^^^^^^^^

For all available functions see:

 - `API reading-methods <https://deltares.github.io/hydromt/latest/api/api_methods.html#reading-methods>`_
 - `API get data <https://deltares.github.io/hydromt/latest/api/api_data_adapter.html#get-data>`_
 - `API Data Catalog <https://deltares.github.io/hydromt/latest/api/api_data_adapter.html#data-catalog>`_
 - `API Data adapter <https://deltares.github.io/hydromt/latest/api/api_data_adapter.html#data-adapter>`_



Visualizing a dataset
^^^^^^^^^^^^^^^^^^^^^

For visualization purposes one can use the `geopandas explore function <https://geopandas.org/en/stable/docs/reference/api/geopandas.GeoDataFrame.explore.html#geopandas-geodataframe-explore>`_.


.. toctree::
    :hidden:

    existing_datacatalogs.rst
    prepare_data.rst
    data_conventions.rst