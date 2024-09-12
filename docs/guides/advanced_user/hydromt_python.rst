.. _hydromt_python:

Using HydroMT in Python
=======================

As HydroMT's architecture is modular, it is possible to use HydroMT as a python library
without using the command line interface (CLI). Using the libary, you actually have
access to many internal functionalities. This can be useful if you want to for example:

- build / update / clip models, check configurations or export data in Python instead of
  the CLI
- analyse model inputs or results
- analyse and compare input data without connecting to a specific model
- process input data for another reason than building a model
- create your own HydroMT plugin

So first, let's go deeper into the API of HydroMT. You have all available functions
and their documentation in the `API reference <../api.rst>`_.

HydroMT is here to read and harmonize **input data**, and to process it via its
**methods and workflows** in order to prepare ready to run **models**. So HydroMT's
methods are organized around these main objects.

Input data
----------

.. currentmodule:: hydromt.data_catalog

Most classes around the finding, reading and transforming input data have
implementations for the five different data_types in HydroMT.
The main objects to work with input data are:

* The :class:`~data_catalog.DataCatalog` is the most high-level class and leverages the
    next few classes to find, read and transform the data coming from its configuration.
    The methods used to do this are called  ``get_<data_type>``, for example
    :func:`~data_catalog.DataCatalog.get_rasterdataset`.
* The :class:`~uri_resolvers.uri_resolver.URIResolver` is responsible for finding the
    data based on a single uri. This class is generic for all data_types. An
    implementation that finds data based on naming conventions is the
    :class:`~uri_resolvers.convention_resolver.ConventionResolver`.
* The :class:`Driver <drivers.base_driver.BaseDriver>` has different subclasses based on
    the data_type, for example
    :class:`~drivers.raster.raster_dataset_driver.RasterDatasetDriver`, which then has
    different implementations, for example a driver for reading raster data using
    rasterio: :class:`~drivers.raster.rasterio_driver.RasterioDriver`. which reads
    raster data.
* The :class:`DataAdapter <adapters.data_adapter_base.DataAdapterBase>` has subclasses
    that transform data, like renaming, reprojecting etc. These subclasses are for
    example: :class:`~adapters.rasterdataset.RasterDatasetAdapter`.

So let's say you would like to read data in HydroMT, you can do this by creating a
DataCatalog instance and using the ``get_<data_type>`` methods to read the data.
This is a short example:

.. code-block:: python

    import hydromt
    # create a data catalog from a local data_catalog file
    cat = hydromt.DataCatalog("data_catalog.yml")
    # read a raster dataset ("dem" source in the data catalog)
    dem = cat.get_rasterdataset("dem")
    # read a vector dataset ("catchments" source in the data catalog)
    catchments = cat.get_geodataframe("catchments")
    # read a geodataset with some time and space slicing
    qobs = cat.get_geodataset(
      "qobs",
      time_range = ("2000-01-01", "2010-12-31"),
      bbox = [5.0, 50.0, 6.0, 51.0]
    )

You can find more detailed examples on using the DataCatalog and DataAdapter in Python in:

* `Reading raster data <../_examples/reading_raster_data.ipynb>`_
* `Reading vector data <../_examples/reading_vector_data.ipynb>`_
* `Reading geospatial point time-series data <../_examples/reading_point_data.ipynb>`_
* `Preparing a data catalog <../_examples/prep_data_catalog.ipynb>`_
* `Exporting data <../_examples/export_data.ipynb>`_

.. _xarray_accessors:

Xarray Accessors
----------------
Some powerful functionality that HydroMT uses is exposed in the ``gis`` module. In this
module `xarray accessors
<https://docs.xarray.dev/en/stable/internals/extending-xarray.html>`_ are located. These
allow for powerful new methods on top of xarray ``Dataset`` and ``DataArray`` classes.
There is the :ref:`raster API <raster_api>`, such as functionality to repoject,
resample, transform, interpolate nodata or zonal statistics. There is also the
:ref:`GeoDataset API <geodataset_api>` to work with geodataset data (N-dim
point/line/polygon geometry). For example, reprojecting, transform, update geometry or
convert to geopandas.GeoDataFrame to access further GIS methods.

.. _flowpy_wrappers:

FlowPy Wrappers:
----------------

`PyFlwDir <https://deltares.github.io/pyflwdir/latest/index.html>` contains a series of
methods to work with gridded DEM and flow direction datasets. The ``gis.flw`` module
builds on top of this and provides hydrological methods for raster DEM data. For
example, calculate flow direction, flow accumulation, stream network, catchments, or
reproject hydrography.

.. _stats:

Stats:
------

The ``stats`` module has statistical methods including ``skills`` to compute skill
scores of models (e.g. NSE, KGE, bias and many more) and ``extremes`` to analyse extreme
events (extract peaks or compute return values).

.. _processes:

.. currentmodule:: hydromt.model

The ``model`` module has a ``processes`` submodule. This module contains some functions
to work with different kinds of model in- and ouput.

* :ref:`grid <workflows_grid_api>`: generic workflows to prepare regular gridded data.
    Used with the :class:`~grid.GridComponent`. For example to prepare regular grid data from constant,
    from RasterDataset (with or without reclassification) or from GeoDataFrame.
* :ref:`mesh <workflows_mesh_api>`: generic workflows to prepare unstructured mesh
    data. Used with the :class:`~mesh.MeshComponent`. For example to create a mesh grid or prepare
    unstructured mesh data from RasterDataset.
* :ref:`basin_mask <workflows_basin_api>`: workflows to prepare a basin mask based on
    different region definitions (bounding box, point coordinates, polygon etc.)
* :ref:`rivers <workflows_rivers_api>`: workflows to prepare river profile data like
    width and depth.
* :ref:`temp <workflows_forcing_api>`: workflows to prepare meteorological forcing
    data. For example to prepare precipitation, temperature, or compute
    evapotranspiration data. Advanced downscaling methods are also available within
    these workflows.

You can find a couple of detailed examples of how to use HydroMT methods and workflows in Python:

* `Working with raster data <../_examples/working_with_raster.ipynb>`_
* `Working with flow direction data <../_examples/working_with_flow_directions.ipynb>`_
* `Define hydrological model regions <../_examples/delineate_basin.ipynb>`_
* `Extreme Value Analysis <../_examples/doing_extreme_value_analysis.ipynb>`_

Models
------

As well as with the CLI, you can also :ref:`build <python_build>`, :ref:`update
<python_update>` or :ref:`clip <python_clip>` models in Python. If you want to develop
you own plugin you can find detailed information in the :ref:`plugin development guide
<plugin_quickstart>`.

But you can also use HydroMT and its :class:`~model.Model` and
:class:`~components.base.ModelComponent` classes to do some analysis on your model
inputs or results. HydroMT views a model as a combination of different components to
represent the different type of inputs of a model, like ``config`` for the model run
configuration file, ``forcing`` for the dynamic forcing data of the model etc. For each
component, there are methods to ``set_<component>`` (update or add a new data layer),
``read_<component>`` and ``write_<component>``. In the :ref:`model API
<model_interface>` you can find all available components.

Here is a small example of how to use the :class:`~model.Model` class in python to plot
or analyse your model:

.. code-block:: python

    import hydromt
    # create a GridModel instance for an existing grid model saved in "grid_model" folder
    model = hydromt.GridModel(root="grid_model", mode="r")
    # read/get the grid data
    forcing = model.grid
    # plot the DEM
    grid["dem"].plot()

    # read the model results
    results = model.results
    # Get the simulated discharge
    qsim = results["qsim"]
    # Read observations using a local data catalog
    cat = hydromt.DataCatalog("data_catalog.yml")
    qobs = cat.get_geodataset("qobs")
    # Compute some skill scores like NSE
    nse = hydromt.stats.nashsutcliffe(qobs, qsim)

You can find more detailed examples on using the Model class in Python in:

* `Working with (grid) models in python <../_examples/working_with_models.ipynb>`_
* `Working with mesh models <../_examples/working_with_meshmodel.ipynb>`_

And feel free to visit some of the :ref:`plugins <plugins>` documentation to find even more examples!
