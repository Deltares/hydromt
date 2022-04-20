.. _intro_user_guide:

User guide
==========

HydroMT is a Python package that aims to facilitate the process of building models and analyzing model results
by automating the process to go from raw data to model data. It is an interface between *user*, *data* and hydro
*models*. 

From the user side, HydroMT is organized in the following way:

| **Data Catalog**
| HydroMT is data-agnostic through the *Data adapter* that reads a wide range of data formats 
  and unifies the input data. HydroMT currently supports vector (*GeoDataFrame*), raster (*RasterDataset*)
  and time-series data (*GeoDataset*) types. Datasets are listed in and passed to HydroMT in a user defined data catalog 
  :ref:`yaml file <data_yaml>`. HydroMT also provides several pre-defined data catalogs with mostly global datasets that can be used as is, 
  but note that not all data is openly accessible. 

| **Configuration**
| The complete building or updating process of a model can be configured in a single configuration *.ini* file. 
  This file describes the full pipeline of model setup components and their arguments. The components vary per 
  model :ref:`plugin <plugins>` and are documented for each at their respective documentation websites.

| **Command Line Interface (CLI)**
| The CLI is a high-level interface to HydroMT. It is used to run HydroMT methods such as 
  :ref:`*build* <cli_build>`, :ref:`*update* <cli_update>` or :ref:`*clip* <cli_clip>`.

| **Python Interface**
| Most common functionalities can be called through the CLI. From the Python interface, however, the user
  can interact directly with a model through the :ref:`Model API <model_interface>` that provides a general interface 
  to the model schematization (*staticgeoms* and *staticmaps*), model *forcing*, model *states*, model *results* 
  and model configuration (*config*). Furthermore, many methods for raster and vector GIS, 
  hydrography and statists are available.

.. image:: ../_static/Architecture_model_data_input.png

.. toctree::
   :maxdepth: 2
   :hidden:
   :caption: Table of Contents

   data_main.rst
   model_main.rst
   gis.rst
   statistics.rst
   terminology.rst  