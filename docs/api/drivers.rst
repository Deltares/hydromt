.. currentmodule:: hydromt.data_catalog.drivers

.. _driver_api:

=======
Drivers
=======

The Hydromt drivers module provides drivers for various datasets and formats.
Each driver implements `read` and optionally `write` methods, along with configuration options and a file system handler.


FileSystem
--------------

All drivers rely on shared type definitions from :mod:`hydromt.typing`.

.. currentmodule:: hydromt.typing

.. autosummary::
   :toctree: ../_generated
   :nosignatures:

   FSSpecFileSystem

.. currentmodule:: hydromt.data_catalog.drivers


Driver Base Classes
--------------------

.. autosummary::
   :toctree: ../_generated
   :nosignatures:

   BaseDriver
   DriverOptions

Raster Data Drivers
--------------------

.. autosummary::
   :toctree: ../_generated
   :nosignatures:

   RasterDatasetDriver
   RasterDatasetXarrayDriver
   XarrayDriverOptions
   RasterioDriver
   RasterioOptions

Vector & Geospatial Drivers
----------------------------

.. autosummary::
   :toctree: ../_generated
   :nosignatures:

   GeoDataFrameDriver
   PyogrioDriver
   GeoDataFrameTableDriver
   GeoDataFrameTableOptions

Tabular Data Drivers
---------------------

.. autosummary::
   :toctree: ../_generated
   :nosignatures:

   DataFrameDriver
   PandasDriver

Geospatial Dataset Drivers
---------------------------

.. autosummary::
   :toctree: ../_generated
   :nosignatures:

   GeoDatasetDriver
   XarrayDriverOptions
   GeoDatasetXarrayDriver
   GeoDatasetVectorDriver

General Dataset Drivers
------------------------

.. autosummary::
   :toctree: ../_generated
   :nosignatures:

   DatasetDriver
   DatasetXarrayDriver
   XarrayDriverOptions

Preprocessing
-------------

.. autosummary::
   :toctree: ../_generated

   preprocessing.harmonise_dims
   preprocessing.remove_duplicates
   preprocessing.round_latlon
   preprocessing.to_datetimeindex
