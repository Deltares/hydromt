
.. _model_config:

Model configuration
-------------------

A user can define a complete pipeline of methods and their options to :ref:`build a model <model_build>` or :ref:`updating an existing model <model_update>`
in the configuration `.ini file <https://en.wikipedia.org/wiki/INI_file>`_ 

The .ini file has a simple syntax with sections and key-value pairs. In HydroMT sections corresponds with model methods
and the key-value pair with the options of each method. When passed to the build or update CLI methods, HydroMT
executes all methods in order as provided in the .ini file. As such the .ini file, in combination with a data catalog yaml file 
define a **reproducible** model.



An example ini file is shown below.
