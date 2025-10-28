.. _hydromt_python:

Using HydroMT in Python
=======================

As HydroMT's architecture is modular, it is possible to use HydroMT as a python library
without using the command line interface (CLI). Using the library, you have more of
HydroMT's functionalities at your disposal. This can be useful if you want to for
example:

- build / update models, check configurations or export data in Python instead of
  the CLI
- analyze model inputs or results
- analyze and compare input data without connecting to a specific model
- process input data for another reason than building a model
- create your own HydroMT plugin

So first, let's go deeper into the API of HydroMT. You have all available functions
and their documentation in the `API reference <../api.rst>`_.

HydroMT is here to read and harmonize **input data**, and to process it via its
**methods and (GIS) processes** in order to prepare ready to run **models**. So HydroMT's
methods are organized around these main objects.

.. dropdown:: **Model functions**

   - :ref:`build a model <hydromt_build_python>`
   - :ref:`update a model <hydromt_update_python>`
   - :ref:`loading a model <hydromt_load_python>`

.. dropdown:: **Data catalog functions**

   - :ref:`export data <hydromt_export_python>`
   - :ref:`reading data <hydromt_data_read_python>`

.. dropdown:: **Methods and processes**

   - :ref:`methods <hydromt_methods_python>`
   - :ref:`processes <hydromt_processes_python>`

Model functions
---------------

You can use HydroMT to build or update a model in Python instead of the CLI. Additionally
with Python, you can also load an existing model to read or analyze its inputs or results.

.. _hydromt_build_python:
Building a model
^^^^^^^^^^^^^^^^

The ``build`` function is used to build models from scratch. In Python, you can also
use the build function in combination with the build workflow file to build a model. Here
is a small example of how to use the build function in Python:

.. code-block:: python

    from hydromt import ExampleModel
    from hydromt.io import read_workflow_yaml

    # Instantiate model
    model = ExampleModel(
        root="./path/to/example_model",
        data_catalog=["./path/to/data_catalog.yml"],
    )
    # Read build options from yaml
    _, _, build_options = read_workflow_yaml(
        "./path/to/build_options.yaml"
    )
    # Build model
    model.build(steps=build_options)

Additionally, in Python, you can also build the model step-by step by calling each of the
model steps as methods, instead of using a workflow file. For example:

.. code-block:: python

    from hydromt import ExampleModel

    # Instantiate model
    model = ExampleModel(
        root="./path/to/example_model",
        data_catalog=["./path/to/data_catalog.yml"],
    )
    # Build model step by step
    # Step 1: populate the config with some values
    model.config.update(
        data = {'starttime': '2000-01-01', 'endtime': '2010-12-31'}
    )
    # Step 2: define the model grid
    model.grid.create_from_region(
        region={"subbasin": [12.2051, 45.8331], "uparea": 50},
        res=1000,
        crs="utm",
        hydrography_fn="merit_hydro_1k",
        basin_index_fn="merit_hydro_index",
    )
    # Step 3: add DEM data to the model grid
    model.grid.add_data_from_rasterdataset(
        raster_fn="merit_hydro_1k",
        variables="elevtn",
        fill_method=None,
        reproject_method="bilinear",
        rename={"elevtn": "DEM"},
    )
    # Write the model to disk
    model.write()

.. _hydromt_update_python:
Updating a model
^^^^^^^^^^^^^^^^
The ``update`` function is used to update an existing model. In Python, you can also
use the update function in combination with the workflow file to update a model. Here
is a small example of how to use the update function in Python:

.. code-block:: python

    from hydromt import ExampleModel
    from hydromt.io import read_workflow_yaml

    # Instantiate model
    model = ExampleModel(
        root="./path/to/example_model_to_update",
        data_catalog=["./path/to/data_catalog.yml"],
        mode = "r+", # open model in read and write mode
    )
    # Read update options from yaml
    _, _, update_options = read_workflow_yaml(
        "./path/to/update_options.yaml"
    )
    # If you want to save the model in a different folder
    model.read()
    model.root.set("./path/to/updated_example_model", mode="w")
    # Update model
    model.update(steps=update_options)

Similarly to build, you can also update the model step by step by calling each of the
model steps as methods, instead of using a workflow file. For example:

.. code-block:: python

    from hydromt import ExampleModel

    # Instantiate model
    model = ExampleModel(
        root="./path/to/example_model_to_update",
        data_catalog=["./path/to/data_catalog.yml"],
        mode = "r+", # open model in read and write mode
    )
    # If you want to save the model in a different folder
    model.read()
    model.root.set("./path/to/updated_example_model", mode="w")
    # Update model step by step
    # Step 1: update the config with new values
    model.config.update(
        data = {'starttime': '2010-01-01', 'endtime': '2020-12-31'}
    )
    # Step 2: add landuse data in the model grid
    model.grid.add_data_from_rasterdataset(
        raster_fn="vito",
        reproject_method="mode",
        rename={"vito": "landuse"},
    )
    # Write the updated model to disk
    model.write()

.. _hydromt_load_python:
Loading and analyzing a model
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
You can also use HydroMT and its :class:`~model.Model` and
:class:`~components.base.ModelComponent` classes to do some analysis on your model
inputs or results. HydroMT views a model as a combination of different components to
represent the different type of inputs of a model, like ``config`` for the model run
configuration file, ``forcing`` for the dynamic forcing data of the model etc. For each
component, there are methods to ``<component>.set`` (update or add a new data layer),
``<component>.read`` and ``<component>.write``.

Here is a small example of how to use the :class:`~model.Model` class in python to plot
or analyze your model:

.. code-block:: python

    from hydromt import ExampleModel
    # create a ExampleModel instance for an existing model saved in "example_model" folder
    model = hydromt.ExampleModel(root="example_model", mode="r")
    # read/get the grid data
    grid = model.grid.data
    # plot the DEM
    grid["DEM"].plot()

You can find more detailed examples on using the Model class in Python in:

* `Working with models in python <../_examples/working_with_models.ipynb>`_

And feel free to visit some of the :ref:`plugins <plugins>` documentation to find even more examples!

Data catalog functions
----------------------

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

.. warning::
    Remember to close any open datasets when you are finished working with them.
    Leaving datasets open may cause xarray to lock the files, which can prevent
    access by other processes until your Python session ends. For example, if you
    open a NetCDF file used by a model and do not close it before calling
    ``Model.update()``, HydroMT will raise a ``PermissionError`` when attempting
    to write to the model.

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
    Used with the :class:`~grid.GridComponent`. For example to prepare regular grid data
    from constant, from RasterDataset (with or without reclassification) or from GeoDataFrame.
* :ref:`mesh <workflows_mesh_api>`: generic workflows to prepare unstructured mesh
    data. Used with :class:`~mesh.MeshComponent`. For example to create a mesh grid or
    prepare unstructured mesh data from RasterDataset.
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
