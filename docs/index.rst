====================================================
HydroMT: Build and analyze models like a data-wizard
====================================================

HydroMT is an open-source Python package that aims to facilitate the process of building models 
and analyzing model results based on the state-of-the-art scientific python ecosystem.
The package has been designed with a data-centered modelling process in mind by automating the 
full process to go from raw data to model data.  

Currently, HydroMT has been implemented for several models through a `plugin <plugins>`_ infrastructure. 
Supported models include the distributed rainfall-runoff model wflow, the sediment model wflow_sediment, 
the hydrodynamic flood model SFINCS, the water quality models D-Water Quality and D-Emissions 
and the flood impact model Delft-FIAT. 


.. |BlueEarth| image:: _static/BlueEarth-icon.png

.. toctree::
   :titlesonly:
   :hidden:

   getting_started/intro.rst
   user_guide/intro.rst
   api/api_index.rst
   dev_guide/intro.rst
   plugins.rst