.. currentmodule:: hydromt.data_catalog.sources

.. _data_source:

============
Data Sources
============

The Hydromt data sources module provides access to various types of datasets.
Each data source wraps data I/O behavior with standardized interfaces, providing consistent
read and metadata operations across raster, vector, and tabular data.

Base Classes
------------

.. autosummary::
   :toctree: ../_generated
   :nosignatures:

   DataSource

Raster Data Sources
-------------------

.. autosummary::
   :toctree: ../_generated
   :nosignatures:

   RasterDatasetSource

Vector Data Sources
-------------------

.. autosummary::
   :toctree: ../_generated
   :nosignatures:

   GeoDataFrameSource
   GeoDatasetSource

Tabular Data Sources
--------------------

.. autosummary::
   :toctree: ../_generated
   :nosignatures:

   DataFrameSource
