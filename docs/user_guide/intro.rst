.. _intro_user_guide:

==========
User guide
==========

**HydroMT** (Hydro Model Tools) is an open-source Python package that facilitates the process of
building and analysing spatial geoscientific models with a focus on water system models.
It does so by automating the workflow to go from raw data to a complete model instance which
is ready to run and to analyse model results once the simulation has finished.
As such it is an interface between *user*, *data* and *models*.

.. figure:: ../_static/hydromt_using.jpg

  A sequence describing how to prepare ready-to-run models using HydroMT

In short the most common usage of hydromt is to build/update models via the following
steps (see also :ref:`the quick overview on how to use HydroMT <quick_overview>`):

1. Collect and catalog raw input data (e.g. DEM, land use, soil, climate, etc.)
2. Prepare a HydroMT configuration file to let HydroMT know which part of your model you wish to prepare, how and using which data (e.g. DEM from SRTM, MERIT Hydro, Copernicus or other source)
3. Run HydroMT to do all the data reading and processing for you either via the command line interface or via the Python API.

In this user guide, we will go through the different steps of the workflow and the
different functionalities of HydroMT. We will also provide examples and tutorials to
help you get started with HydroMT.

The user guide is organized as follows:

- :ref:`Introduction to HydroMT <detailed_intro>`: A brief introduction to HydroMT and its main functionalities.
- :ref:`Quick overview <quick_overview>`: A quick overview on how to use HydroMT.
- :ref:`HydroMT command line interface <hydromt_cli>`: A detailed description of the command line interface.
- :ref:`HydroMT Python API <hydromt_python>`: A detailed description of the Python API (advanced users).
- :ref:`Working with data in HydroMT <get_data>`: A detailed section to go through how HydroMT interacts
  with raw input data through its DataCatalog.
- :ref:`Working with models in HydroMT <model_main>`: A detailed section to go through how to prepare and interact with
  models using HydroMT.
- :ref:`Methods and workflows <methods_workflows>`: A detailed section to go through the different methods and workflows
  available in HydroMT (advanced or python users).

One final remark: because each model is unique at least in its input file format (e.g.
netCDF, text, binary, etc.), HydroMT is usually used in combination with a **plugin**.
These plugins are here to define model specific file formats and readers/writers. You can
find the list of available plugins in the :ref:`plugins section <plugins>`.

.. toctree::
   :maxdepth: 2
   :hidden:
   :caption: Working with HydroMT

   hydromt_intro.rst
   ../getting_started/quick_overview.rst
   hydromt_cli.rst
   hydromt_python.rst
   terminology.rst

.. toctree::
   :maxdepth: 2
   :hidden:
   :caption: Working with data in HydroMT

   data_overview.rst
   data_prepare_cat.rst
   data_types.rst
   data_existing_cat.rst
   data_conventions.rst
   ../_examples/prep_data_catalog.ipynb
   ../_examples/export_data.ipynb
   ../_examples/reading_raster_data.ipynb
   ../_examples/reading_vector_data.ipynb
   ../_examples/reading_point_data.ipynb
   ../_examples/reading_tabular_data.ipynb
   ../_examples/working_with_tiled_raster_data.ipynb

.. toctree::
   :maxdepth: 2
   :hidden:
   :caption: Working with models in HydroMT

   model_overview.rst
   model_build.rst
   model_update.rst
   model_clip.rst
   model_config.rst
   model_region.rst
   ../_examples/working_with_models_basics.ipynb
   ../_examples/working_with_models.ipynb
   ../_examples/working_with_meshmodel.ipynb
   ../_examples/delineate_basin.ipynb

.. toctree::
   :maxdepth: 2
   :hidden:
   :caption: Methods and workflows

   methods_main.rst
   methods_stats.rst
   ../_examples/working_with_raster.ipynb
   ../_examples/working_with_geodatasets.ipynb
   ../_examples/working_with_flow_directions.ipynb
   ../_examples/doing_extreme_value_analysis.ipynb
