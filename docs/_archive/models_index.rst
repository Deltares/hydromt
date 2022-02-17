.. _ini_options:

Models
======
HydroMT model plugins
---------------------
HydroMT core can easily access and navigate through different model implementations thanks to its plugin architecture.
The hydromt package contains core functionnalities and the command line interface. The models, following HydroMT's 
architecture, are then implemented in separate plugins available in other python packages. 

All the **core functionnalities** are included in the **hydromt package**. This includes:

- Command Line Interface and methods (build, update, clip).
- High end methods for raster and vector processing, GIS, configuration and flow direction functionnalities.
- Data reading, processing and export via the ``DataCatalog``.
- General workflows from input data to model data such as basin_mask or forcing.
- Definition of the hydroMT ``Model API`` class.

Implementation of hydroMT core functionnalities for specific models are realised in separate **plugin packages** (one per model). In each plugin, you can find:

- Definition of a Model class compliant to the ``Model API`` with model specific methods and attributes.
- Model specific workflows to go from input data to model data.
- Specific documentation and examples.

The plugin architecture of HydroMT (including known existing plugins and plugins in development) is then:

.. image:: ../../img/hydromt_plugins.png

Available model plugins
-----------------------

The known existing plugin packages can be found at:

- Delft-FIAT: https://github.com/Deltares/hydromt_fiat
- DELWAQ: https://github.com/Deltares/hydromt_delwaq
- SFINCS: https://github.com/Deltares/hydromt_sfincs
- Wflow: https://github.com/Deltares/hydromt_wflow