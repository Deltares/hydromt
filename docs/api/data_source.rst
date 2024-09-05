.. currentmodule:: hydromt.data_catalog.sources

.. _data_source:

============
Data sources
============

General
-------

.. autosummary::
   :toctree: ../_generated

   DataSource
   DataSource.summary

RasterDataset
-------------

.. autosummary::
   :toctree: ../_generated

   RasterDatasetSource
   RasterDatasetSource.read_data
   RasterDatasetSource.to_stac_catalog
   RasterDatasetSource.get_bbox
   RasterDatasetSource.get_time_range
   RasterDatasetSource.detect_bbox
   RasterDatasetSource.detect_time_range

GeoDataFrame
------------

.. autosummary::
   :toctree: ../_generated

   GeoDataFrameSource
   GeoDataFrameSource.read_data
   GeoDataFrameSource.to_stac_catalog
   GeoDataFrameSource.get_bbox
   GeoDataFrameSource.detect_bbox

DataFrame
---------

.. autosummary::
   :toctree: ../_generated

   DataFrameSource
   DataFrameSource.read_data
   DataFrameSource.to_stac_catalog

GeoDataset
------------

.. autosummary::
   :toctree: ../_generated

   GeoDatasetSource
   GeoDatasetSource.read_data
   GeoDatasetSource.to_stac_catalog
   GeoDatasetSource.get_bbox
   GeoDatasetSource.detect_bbox
