.. _intro_user_guide:

User Guide
==========

HydroMT is a Python package that aims to facilitate the process of building models and analyzing model results
by automating the process to go from raw data to model data. It is an interface between *data*, *user* and hydro
*models*. In the user guide you can find:

- how to work with data in HydroMT
- how to develop and process models with HydroMT

From the user side, HydroMT is organised in the following way:

| **Command Line Interface (CLI)**
| The CLI is a high-level interface to HydroMT. It is used to run HydroMT methods such as **build**, **update** or **clip** for
  all model plugins, such as Wflow, Delwaq, SFINCS etc.

| **Configuration**
| The complete building or updating process of a model can be configured in a single configuration *.ini* file. 
  This file describes the full pipeline of model setup components and their arguments. The components vary for the 
  different model plugins and are documented for each at their respective documentation websites.

| **Data Catalogue**
| HydroMT can make use of various types of data sources such as vector data, GDAL rasters or NetCDF files.
  The path and attributes of each of these dataset are listed in a data catalogue *.yml* file. HydroMT provides 
  several pre-defined data catalogues with mostly global datasets that can be used as is, but note that not all data is 
  openly accessible. Local or other datasets can also be included by extending or using a user defined yaml file.

| **Python Interface**
| Most common functionalities can be called through the CLI. From the Python interface, however, much more
  lower level functionalities are available.

This user guide concentrates on the core functions of HydroMT. For more specific information each
:ref:`plugin <plugins>` contains an additional user guide.

Content
-------

.. toctree::
   :maxdepth: 2
   :hidden:

   get_data.rst
   existing_datacatalogs.rst
   prepare_data.rst
   model_main.rst
   terminology.rst
   data_conventions.rst