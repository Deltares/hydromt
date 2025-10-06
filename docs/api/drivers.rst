.. currentmodule:: hydromt.data_catalog.drivers

.. _drivers:

=======
Drivers
=======

The Hydromt drivers module provides drivers for various datasets and formats.
Each driver implements `read` and optionally `write` methods, along with configuration options.

Driver Base Classes
--------------------

.. autosummary::
   :toctree: ../_generated
   :nosignatures:
   :show-inheritance:

   BaseDriver
   DriverOptions

Raster Data Drivers
--------------------

.. autosummary::
   :toctree: ../_generated
   :nosignatures:
   :show-inheritance:

   RasterDatasetDriver
   RasterDatasetXarrayDriver
   RasterXarrayOptions
   RasterioDriver
   RasterioOptions

Vector & Geospatial Drivers
----------------------------

.. autosummary::
   :toctree: ../_generated
   :nosignatures:
   :show-inheritance:

   GeoDataFrameDriver
   PyogrioDriver
   GeoDataFrameTableDriver
   GeoDataFrameTableOptions

Tabular Data Drivers
---------------------

.. autosummary::
   :toctree: ../_generated
   :nosignatures:
   :show-inheritance:

   DataFrameDriver
   PandasDriver

Geospatial Dataset Drivers
---------------------------

.. autosummary::
   :toctree: ../_generated
   :nosignatures:
   :show-inheritance:

   GeoDatasetDriver
   GeoDatasetOptions
   GeoDatasetXarrayDriver
   GeoDatasetVectorDriver

General Dataset Drivers
------------------------

.. autosummary::
   :toctree: ../_generated
   :nosignatures:
   :show-inheritance:

   DatasetDriver
   DatasetXarrayDriver
   DatasetXarrayOptions

Preprocessing
-------------

.. autosummary::
   :toctree: ../_generated
   :show-inheritance:

   preprocessing.harmonise_dims
   preprocessing.remove_duplicates
   preprocessing.round_latlon
   preprocessing.to_datetimeindex
