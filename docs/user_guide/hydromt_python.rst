.. _hydromt_python:

Using HydroMT in Python
=======================

As HydroMT's architecture is very modular, it is possible to use HydroMT in Python
without using the command line interface (CLI). With Python, you actually have access to
a lot of internal functionalities. This can be useful if you want to for example:

- build / update / clip models, check configurations or export data in Python instead of the CLI
- analyse model inputs or results
- analyse and compare input data without connecting to a specific model
- process input data for another reason than building a model
- create your own HydroMT plugin

So first, let's go deeper into the API of HydroMT. You have all available functions
and their documentation in the `API reference <../api.rst>`_.

HydroMT is here to read and harmonize **input data**, and to process it via its **methods and
workflows** in order to prepare ready to run **models**. So HydroMT's methods are organized
around these main objects.

Input data
----------
The main objects to work with input data are the `DataCatalog <../_generated/hydromt.data_catalog.DataCatalog.rst>`_
and the `DataAdapter <../_generated/hydromt.data_adapter.DataAdapter.rst>`_ classes of HydroMT.
The ``DataCatalog`` is what is used to tell HydroMT where the data can be found and how it can be read,
as well as what maintains the administration of exactly what data was used to maintain reproducibility.
The ``DataAdapter`` are what does the actual reading of the data and get instructed and instantiated by
the DataCatalog. Currently, five different types of input data are supported by the Adapters and represented by a specific Python data
object:

- `RasterDatasetAdapter <../_generated/hydromt.data_adapter.RasterDatasetAdapter.rst>`_ :
  gridded datasets such as DEMs or gridded spatially distributed rainfall datasets (represented
  by :ref:`RasterDataset <RasterDataset>` objects, a raster-specific type of Xarray Datasets)
- `DataFrameAdapter <../_generated/hydromt.data_adapter.DataFrameAdapter.rst>`_ :
  tables that can be used to, for example, convert land classes to roughness values (represented by
  Pandas :ref:`DataFrame <DataFrame>` objects)
- `GeoDataFrameAdapter <../_generated/hydromt.data_adapter.GeoDataFrameAdapter.rst>`_ :
  vector datasets such as administrative units or river center lines (represented by Geopandas :ref:`GeoDataFrame <GeoDataFrame>` objects)
- `GeoDatasetAdapter <../_generated/hydromt.data_adapter.GeoDatasetAdapter.rst>`_ :
  time series with associated geo-locations such as observations of discharge (represented by :ref:`GeoDataset <GeoDataset>`
  objects, a geo-specific type of Xarray Datasets)
- `DatasetAdapter <../_generated/hydromt.data_adapter.DatasetAdapter.rst>`_ :
  non-spatial N-dimension data (represented by Xarray :ref:`Dataset <Dataset>` objects).

So let's say you would like to read data in HydroMT, you can do this by creating a
DataCatalog instance and using the ``get_<data_type>`` methods to read the data. These are:

- `get_rasterdataset <../_generated/hydromt.data_catalog.DataCatalog.get_rasterdataset.rst>`_
- `get_dataframe <../_generated/hydromt.data_catalog.DataCatalog.get_dataframe.rst>`_
- `get_geodataframe <../_generated/hydromt.data_catalog.DataCatalog.get_geodataframe.rst>`_
- `get_geodataset <../_generated/hydromt.data_catalog.DataCatalog.get_geodataset.rst>`_
- `get_dataset <../_generated/hydromt.data_catalog.DataCatalog.get_dataset.rst>`_

Here is a short example of how to read data in Python using HydroMT:

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

Methods and workflows
---------------------
Most of the heavy work in HydroMT is done by :ref:`Methods and workflows <methods_workflows>`.
``Methods`` provide the low-level functionality such as GIS rasterization, reprojection, or zonal statistics.
``Workflows`` combine several methods to transform input data to a model layer (e.g. interpolate nodata,
then reproject). Examples of workflows include the delineation of hydrological basins (watersheds), conversion
of landuse-landcover data to model parameter maps, and preparation of meteorological data.

The available methods in HydroMT are:

- :ref:`raster <raster_api>`: GIS methods to work with raster / regular grid data. For example,
  reprojecting, resampling, transform, interpolate nodata or zonal statistics etc.
- :ref:`vector <geodataset_api>`: GIS methods to work with geodataset data (N-dim point/line/polygon geometry).
  For example, reprojecting, transform, update geometry or convert to geopandas.GeoDataFrame to access
  further GIS methods.
- :ref:`flw <flw_api>`: Hydrological methods for raster DEM data. For example, calculate flow direction,
  flow accumulation, stream network, catchments, or reproject hydrography.
- :ref:`gis_utils <gis_utils_api>`: other general GIS methods. For example to compute the area of
  a grid cell or to perform a merge of nearest geodataframe.
- :ref:`stats <statistics>`: Statistical methods including ``skills`` to compute skill scores
  of models (e.g. NSE, KGE, bias and many more) and ``extremes`` to analyse extreme events
  (extract peaks or compute return values).

The available workflows in HydroMT are:

- :ref:`grid <workflows_grid_api>`: generic workflows to prepare regular gridded data. Used
  with the ``GridModel``. For example to prepare regular grid data from constant, from RasterDataset (with or
  without reclassification) or from GeoDataFrame.
- :ref:`mesh <workflows_mesh_api>`: generic workflows to prepare unstructured mesh data. Used
  with the ``MeshModel``. For example to create a mesh grid or prepare unstructured mesh data from RasterDataset.
- :ref:`basin_mask <workflows_basin_api>`: workflows to prepare a basin mask based on different region
  definitions (bounding box, point coordinates, polygon etc.)
- :ref:`rivers <workflows_rivers_api>`: workflows to prepare river profile data like width and depth.
- :ref:`forcing <workflows_forcing_api>`: workflows to prepare meteorological forcing data. For example to
  prepare precipitation, temperature, or compute evapotranspiration data.	Advanced downscaling methods
  are also available within these workflows.

You can find a couple of detailed examples of how to use HydroMT methods and workflows in Python:

* `Working with raster data <../_examples/working_with_raster.ipynb>`_
* `Working with flow direction data <../_examples/working_with_flow_directions.ipynb>`_
* `Define hydrological model regions <../_examples/delineate_basin.ipynb>`_
* `Extreme Value Analysis <../_examples/doing_extreme_value_analysis.ipynb>`_

Models
------
As well as with the CLI, you can also :ref:`build <python_build>`, :ref:`update <python_update>`
or :ref:`clip <python_clip>` models in Python. If you want to develop you own plugin you can find detailed information in the
:ref:`plugin development guide <plugin_quickstart>`.

But you can also use HydroMT and its ``Model`` class to do some analysis on your model inputs or results.
HydroMT views a model as a combination of different components to represent the different type of inputs
of a model, like ``config`` for the model run configuration file, ``forcing`` for the dynamic forcing
data of the model etc. For each component, there are methods to ``set_<component>`` (update or add a new
data layer), ``read_<component>`` and ``write_<component>``.
In the :ref:`model API <model_interface>` you can find all available components.

Here is a small example of how to use the ``Model`` class in python to plot or analyse your model:

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
