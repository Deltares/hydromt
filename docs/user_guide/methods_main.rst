.. _methods_workflows:

Methods and workflows
=====================

Methods and workflows are the engine of HydroMT. Methods provide the low-level functionality, only accessible through the Python interface, 
to do the required processing of common data types such as grid and vector data. Workflows combine several methods to go from raw input 
data to a model component. Examples of workflows include the delineation of hydrological basins (watersheds), conversion of landuse-landcover to model parameter maps, etc.


.. _workflows:

Workflows
---------

.. toctree::

    ../_examples/delineate_basin.ipynb

.. _gis:

GIS methods
-----------

.. toctree::

    ../_examples/working_with_raster.ipynb


.. _stats:

Statistical methods
-------------------

.. toctree::

    methods_stats.rst