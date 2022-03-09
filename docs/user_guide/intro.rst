.. _intro_user_guide:

User Guide
==========

HydroMT is a Python package that aims to facilitate the process of building models and analyzing model results
by automating the process to go from raw data to model data. It is a interface between *data*, *user* and hydro
*models*. The user guide introduces the the interface components from a user's perspective:

- how to work with data in HydroMT
- how to develop and process models with HydroMT

From the user side, HydroMT is organised in the following way:

| **Command Line Interface (CLI)**
| The CLI is a high-level interface to HydroMT. It is used to run HydroMT methods such as build, update or clip for
  a specific model supported by the package, such as Wflow, Delwaq, SFINCS etc.

| **Configuration**
| When using the CLI, specific options such as which data sources to use, which components to include etc.
  are provided in a *.ini* file. These options, organised in sections, vary for the different models and are documented
  in the model components.

| **Data Catalogue**
| HydroMT can make use of various types of data sources such as vector data, GDAL rasters or NetCDF files.
  The path and attributes of each of these dataset are listed in a *.yml* file. HydroMT already contains a list of default
  global datasets that can be used as is. Local or other datasets can also be included by extending or using another local yaml file.

| **Python Interface**
| Most common functionalities can be called through the CLI. From the Python interface, however, much more
  functionalities are available.

This user guide concentrates on the core functions of HydroMT. For more specific information each
:ref:`plugin <plugins>` contains an additional user guide.

Content
-------

.. toctree::
   :maxdepth: 2

   data.rst
   model_build.rst
   model_update.rst
   model_clip.rst
   model_post.rst
   terminology.rst