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

   - :ref:`reading data <hydromt_data_read_python>`
   - :ref:`export data <hydromt_export_python>`

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
``<component>.read`` and ``<component>.write``. The underlying data of each component
is accessible via the ``<component>.data`` attribute (e.g dict, xarray or geopandas objects,
etc.).

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

* `Working with models in python <../../_examples/working_with_models.ipynb>`_

And feel free to visit some of the :ref:`plugins <plugins>` documentation to find even more examples!

Data catalog functions
----------------------
The DataCatalog is a core part of HydroMT to find, read, harmonize and transform input data.
It is usually prepared from a yaml file that defines different data sources and
their properties. You can find more information on the DataCatalog in the
:ref:`Data Catalog documentation <get_data>`.

.. currentmodule:: hydromt.data_catalog

.. _hydromt_data_read_python:

Reading data
^^^^^^^^^^^^
HydroMT supports reading five different data types:

- **rasterdataset**: raster (regular grid) data (e.g DEM, landuse, soil type, model grid etc.)
- **geodataframe**: vector data (e.g shapefiles, geojson, etc.)
- **geodataset**: time varying vector data (e.g point time-series, station data, etc.)
- **dataframe**: non-spatial tabular data (e.g csv, excel, etc.)
- **dataset**: non-spatial multi-dimensional data (e.g netcdf, hdf5, etc.)

All input data is accessible and be read through the :class:`~data_catalog.DataCatalog` class
via its ``get_<data_type>`` methods, for example: ``get_rasterdataset``,
``get_geodataframe``, ``get_geodataset``, ``get_dataframe``, ``get_dataset``.

Here is a small example of how to use the DataCatalog to read data in Python:

.. code-block:: python

    from hydromt import DataCatalog

    # create a data catalog from a local data_catalog file
    cat = DataCatalog("data_catalog.yml")
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

You can find more detailed examples on using the DataCatalog in Python in:

* `Reading raster data <../../_examples/reading_raster_data.ipynb>`_
* `Reading vector data <../../_examples/reading_vector_data.ipynb>`_
* `Reading geospatial point time-series data <../../_examples/reading_point_data.ipynb>`_
* `Reading tabular data <../../_examples/reading_tabular_data.ipynb>`_

In short, what happens in the bachkground of the ``get_<data_type>`` methods is:

1. The DataCatalog finds the data source in the data catalog that matches the
   requested name and solves the data source path (e.g local file, remote url, database, etc.)
2. The DataCatalog uses the appropriate data reader or ``driver`` to read the data from the source
   depending on its type and file format (e.g. tif, netcdf, shapefile, csv, etc.)
3. The DataCatalog harmonizes (renaming, unit conversion) and slices (variables, time, space) the
   data according to the data source properties using the appropriate ``data_adapter``.

HydroMT is flexible enough that you can add your own data types or readers if needed. You
can find more information on how the DataCatalog works and how to implement your own data
readers in the :ref:`Developer documentation <intro_developer_guide>`.


.. _hydromt_export_python:

Exporting data
^^^^^^^^^^^^^^
HydroMT also supports exporting data from a DataCatalog for a specific region or time range.
This can be useful to extract and export input data for a specific model or region of interest.
For this, you can use the :py:meth:`~data_catalog.DataCatalog.export_data` method of the
:class:`~data_catalog.DataCatalog` class. Here is a small example of how to use the export_data method
in Python:

.. code-block:: python

    from hydromt import DataCatalog

    # create a data catalog from a local data_catalog file
    cat = DataCatalog("data_catalog.yml")
    # export data for a specific region and time range
    cat.export_data(
        new_root = "./exported_data",
        bbox = [5.0, 50.0, 6.0, 51.0],
        time_range = ("2000-01-01", "2010-12-31"),
        source_names = ["dem", "landuse", "qobs"]
    )


Methods and processes
----------------------
HydroMT provides a set of methods and (GIS) processes to process input data in order to
prepare ready to run models. You can find more information on the available methods and
processes in the :ref:`supporting functionnalities <methods_processes>`, :ref:`model processes <model_processes>` and in
the :ref:`API reference <api_reference>`.

These methods and processes are only available via the Python API. You can use them
directly in your Python scripts or Jupyter notebooks to process data for your models
or for other purposes.

.. _hydromt_methods_python:

Methods
^^^^^^^
Methods provide the low-level functionality to do the required processing of common data types
such as grid and vector data.

HydroMT provides methods in different modules including:

- ``gis.raster``: methods to work with raster (regular grid) data including reprojection,
  resampling, transforming, interpolating nodata or zonal statistics.
- ``gis.vector``: methods to work with geodataset data (N-dimensional vector data). For example, reprojecting,
  transforming, updating geometry or converting to geopandas.GeoDataFrame to access further GIS methods.
- ``gis.flw``: hydrological methods for raster DEM data. For example, calculate flow direction,
  flow accumulation, stream network, catchments, or reproject hydrography.
- ``stats.skills``: statistical methods to compute skill scores of models (e.g. NSE, KGE, bias and many more).
- ``stats.extremes``: methods to analyse extreme events (extract peaks or compute return values).

.. _hydromt_processes_python:

Processes
^^^^^^^^^
Processes combine several methods to go from raw input data to a model component. Examples of
processes include the delineation of hydrological basins (watersheds), conversion of
landuse-landcover to model parameter maps, etc.

HydroMT provides processes in the ``model.processes`` module including:

- ``model.processes.basin_mask``: processes to delineate (sub-)basins and create basin masks.
- ``model.processes.grid``: processes to prepare regular gridded data from different data types.
- ``model.processes.mesh``: processes to prepare unstructured mesh data from different data types.
- ``model.processes.rivers``: processes to prepare river network data for hydrological or 1D hydraulic models.
- ``model.processes.meteo``: processes to prepare gridded meteorological forcing data including downscaling methods.
