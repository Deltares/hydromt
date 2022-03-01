.. currentmodule:: hydromt

.. _get_data:

Getting data from the Data Catalog  
==================================

.. note::
    
    TODO: 
    1) integrate some basic examples using ipython snippets (such as below)?;
    2) describe/include additional examples to cover sunctions such as getrasterdataset, print metadata, open_raster, visualize raster ...

With a `DataCatalog` in place **hydroMT** can be used to read the data with the `DataCatalog.getrasterdataset` method.
The use of a `DataCatalog` allows for minimal pre-processing in order to get uniform variable names and units.
See `examples <https://deltares.github.io/hydromt/latest/examples/examples/read_raster_data.html#Reading-raster-data>`_ which highlights 
various commonly used options to read single or multiple file raster datasets into an `xarray.Dataset` or `xarray.DataArray` object with geospatial attributes.

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


Related API references
^^^^^^^^^^^^^^^^^^^^^^

For all available functions see:

 - `API reading-methods <https://deltares.github.io/hydromt/latest/api/api_methods.html#reading-methods>`_
 - `API get data <https://deltares.github.io/hydromt/latest/api/api_data_adapter.html#get-data>`_
 - `API Data Catalog <https://deltares.github.io/hydromt/latest/api/api_data_adapter.html#data-catalog>`_
 - `API Data adapter <https://deltares.github.io/hydromt/latest/api/api_data_adapter.html#data-adapter>`_

Usages with Model plugins 
^^^^^^^^^^^^^^^^^^^^^^^^^

When using hydromt in combination with one of the model plugins one can link to the data catalog by specifying the path where the yml file is located:

::

    -d, --data PATH     File path to yml data sources file.

Providing a data catalog for the CLI ``hydromt build`` and ``hydromt update`` methods is done with 
``-d /path/to/data_catalog.yml``. Entries from the data_catalog can then be used 
in the *options.ini*. Multiple yml files can be added by reusing the ``-d`` option.

.. code-block:: console

    hydromt build MODEL REGION -i options.ini -d /path/to/data_catalog.yml

Basic usage to read a raster dataset

.. code-block:: python

    import hydromt
    data_cat = hydromt.DataCatalog(data_libs=r'/path/to/data-catalog.yml')
    ds = data_cat.get_rasterdataset('merit_hydro', bbox=[xmin, ymin, xmax, ymax])  # returns xarray.dataset


For the hydromt_delwaq plugin three data types exist, see `emissions <https://deltares.github.io/hydromt_delwaq/latest/api/api_workflows.html#emissions>`_: 
raster, vector and admin. The admin type is an administrative raster to which related parameters can be mapped (e.g. country or region boundaries can be used to delign different values of sewage connection percentage). 
A mapping table is required to link the related parameters to the administrative raster.

Visualizing a dataset
^^^^^^^^^^^^^^^^^^^^^

For visualization purposes one can use the `geopandas explore function <https://geopandas.org/en/stable/docs/reference/api/geopandas.GeoDataFrame.explore.html#geopandas-geodataframe-explore>`_.