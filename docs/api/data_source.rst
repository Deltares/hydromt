.. currentmodule:: hydromt.data_catalog.sources

.. _data_source:

============
Data sources
============

.. autosummary::
   :toctree: ../_generated
   :nosignatures:

   DataSource
   RasterDatasetSource
   GeoDataFrameSource
   DataFrameSource
   GeoDatasetSource


General
-------

.. autopydantic_model:: DataSource
   :no-index:
   :show-inheritance:
   :members: summary

RasterDataset
-------------

.. autopydantic_model:: RasterDatasetSource
   :no-index:
   :show-inheritance:
   :members: read_data, to_stac_catalog, get_bbox, get_time_range, detect_bbox, detect_time_range

GeoDataFrame
------------

.. autopydantic_model:: GeoDataFrameSource
   :no-index:
   :show-inheritance:
   :members: read_data, to_stac_catalog, get_bbox, detect_bbox

DataFrame
---------

.. autopydantic_model:: DataFrameSource
   :no-index:
   :show-inheritance:
   :members: read_data, to_stac_catalog

GeoDataset
------------

.. autopydantic_model:: GeoDatasetSource
   :no-index:
   :show-inheritance:
   :members: read_data, to_stac_catalog, get_bbox, detect_bbox
