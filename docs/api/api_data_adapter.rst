.. currentmodule:: hydromt

#############
Data Adapater
#############


Data Catalog
============

General
-------

.. autosummary::
   :toctree: ../_generated

   data_adapter.DataCatalog
   data_adapter.DataCatalog.sources
   data_adapter.DataCatalog.keys
   data_adapter.DataCatalog.to_dict
   data_adapter.DataCatalog.to_dataframe
   data_adapter.DataCatalog.to_yml

Add data sources
----------------

.. autosummary::
   :toctree: ../_generated

   data_adapter.DataCatalog.from_artifacts
   data_adapter.DataCatalog.from_dict
   data_adapter.DataCatalog.from_yml
   data_adapter.DataCatalog.update


Get data
--------

.. autosummary::
   :toctree: ../_generated

   data_adapter.DataCatalog.get_rasterdataset
   data_adapter.DataCatalog.get_geodataset
   data_adapter.DataCatalog.get_geodataframe



RasterDatasetAdapter
====================

.. autosummary::
   :toctree: ../_generated

   data_adapter.RasterDatasetAdapter
   data_adapter.RasterDatasetAdapter.to_dict
   data_adapter.RasterDatasetAdapter.summary
   data_adapter.RasterDatasetAdapter.get_data

GeoDatasetAdapter
==================

.. autosummary::
   :toctree: ../_generated

   data_adapter.GeoDatasetAdapter
   data_adapter.GeoDatasetAdapter.to_dict
   data_adapter.GeoDatasetAdapter.summary
   data_adapter.GeoDatasetAdapter.get_data

GeoDataFrameAdapter
===================

.. autosummary::
   :toctree: ../_generated

   data_adapter.GeoDataFrameAdapter
   data_adapter.GeoDataFrameAdapter.to_dict
   data_adapter.GeoDataFrameAdapter.summary
   data_adapter.GeoDataFrameAdapter.get_data