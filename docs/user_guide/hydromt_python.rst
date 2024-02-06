.. _hydromt_python:

Using HydroMT in Python
=======================

The functionality of HydroMT can be broken down into five components, which are around input data,
model instances, and methods and workflows. Users can interact with HydroMT through a high-level
command line interface (CLI) to build model instances from scratch, update existing model instances
or analyze model results. Furthermore, a Python interface is available that exposes
all functionality for experienced users. An overview of the package components and the concepts they are concerned with can be seen in the table below.

+------------------+------------------+--------------+------------------+------------+-------------+--------------------+-------------+
| Component        | Reproducibility  | Data Access  | Data Processing  | Input Data | Output Data | Plugin Connection  | Provided by |
+==================+==================+==============+==================+============+=============+====================+=============+
| Data Adapter     |                  | x            | x                | x          |             |                    | Core        |
+------------------+------------------+--------------+------------------+------------+-------------+--------------------+-------------+
| Data Catalog     | x                | x            |                  | x          |             |                    | User        |
+------------------+------------------+--------------+------------------+------------+-------------+--------------------+-------------+
| Workflow         |                  |              | x                |            | x           |                    | Core/Plugin |
+------------------+------------------+--------------+------------------+------------+-------------+--------------------+-------------+
| Model (object)   |                  |              |                  |            | x           | x                  | Core/Plugin |
+------------------+------------------+--------------+------------------+------------+-------------+--------------------+-------------+
| Model (config)   | x                |              |                  |            | x           | x                  | User        |
+------------------+------------------+--------------+------------------+------------+-------------+--------------------+-------------+



In short, there are three important concepts in HydroMT core that are important to cover:

- ``DataCatalog``: This is what is used to tell HydroMT where the data can be found and how it can be read, as well as what maintains the administration of exactly what data was used to maintain reproducibility. As a user or plugin developer you do not need to worry about the code for this, but you will need to provide a correct catalog for HydroMT to use.
- ``DataAdapters``: These are what do the actual reading of the data and get instructed and instantiated by the DataCatalog.  While a lot of the work in HydroMT happens here, plugins or users shouldn't really need to know about these beyond using the proper ``data_type`` in their configuration.
- ``Workflows``: These are functions that transform input data and can call a set of methods to for example, resample, fill nodata, reproject, derive
  other variables etc. The core has some of these workflows but you may need new ones for your plugin.
- ``Model`` (object): This is where the magic happens (as far as the plugin is concerned). We have provided some generic models that you can
  override to get basic/generic functionality, but using the model functionality is where it will be at for you. The scheme below lists the current
  relationship between the HydroMT ``Model`` and generic sub-Model classes and the know plugins.
- ``Model`` (config): While the object model is what actually performs the operations and what plugin developers are generally concerned with, the config file of the model is what instructs HydroMT and the plugins on what to do. This is the second file a user will need to provide in order to perform a HydroMT run from the command line.

A more detailed overview of how HydroMT functions internally along with a more in depth explanation are pictured below:

.. _arch:

.. figure:: ../_static/hydromt_run.jpg

  A schematic overview of the sequence of steps that are involved in a HydroMT run.
