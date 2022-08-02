
.. _model_config:

Preparing a model configuration
===============================

A user can define a complete pipeline of methods and their options to :ref:`build a model <model_build>` or :ref:`updating an existing model <model_update>`
in the configuration `.ini file <https://en.wikipedia.org/wiki/INI_file>`_ 

**Steps in brief:**

1) Start with a template of the :ref:`HydroMT model plugin <plugins>` with which you want to build / update a model. Templates can be found in the online documentation of each model plugin.
2) Edit / add / remove sections (i.e. methods) based on which components you want to build or adapt. The arguments of specific methods can be found in the API chapter in the online documentation of each model plugin.
3) Save the configuration file and use it in combination with the HydroMT :ref:`build <model_build>` and :ref:`update <model_update>` methods.

.. NOTE::

    The HydroMT model configuration (.ini) file should not be confused with the simulation configuration file of the model kernel.
    While the first defines how HydroMT should build or update a model, the latter defines the simulation for the model kernel. 
    The format of the latter differs with each plugin, but can be accessed in HydroMT trough the :py:meth:`hydromt.Model.config` component.

Model configuration (.ini) file
------------------------------- 

The .ini file has a simple syntax with sections and key-value pairs. In HydroMT sections corresponds with model methods
and the key-value pair with the arguments of each method. For available methods and their arguments of a specific model, 
please visit the :ref:`plugin documentation pages <plugins>`.
When passed to the build or update CLI methods, HydroMT executes all methods in order as provided in the .ini file. 
As such the .ini file, in combination with a data catalog yaml file 
define a **reproducible** model.

HydroMT configuration file specifications and conventions:

- HydroMT will execute each method (i.e. section) in the order it is provided in the .ini file.
- Methods can be re-used by enumerating the methods by adding a number to the end (without underscore or space!).
  Although this is not enforced in the code, by convention we start enumerating the second call of each method with a number 2, the third call with a number 3 etc.
- Arguments ending with ``_fn`` (short for filename) are by convention used to set a data source from the data catalog based on its source name, see :ref:`Working with data in HydroMT <get_data>`.

An example .ini file is shown below. Note that this .ini file does not apply to any supported model plugin.

.. code-block:: ini

    [setup_basemaps]
    topography_fn = merit   # source name of topography data
    crs = 4326              # CRS EPSG code 
    res = 100               # resolution [m]

    [setup_manning_roughness]
    lulc_fn = globcover             # source name of landuse-landcover data
    mapping_fn = globcover_mapping  # source name of mapping table converting lulc classes to N values