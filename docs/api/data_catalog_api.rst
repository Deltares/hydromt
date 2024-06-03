.. currentmodule:: hydromt.data_catalog

====
Data
====

.. _api_data_catalog:

Data catalog
============

General
-------

.. autosummary::
   :toctree: ../_generated

   DataCatalog
   DataCatalog.get_source
   DataCatalog.sources
   DataCatalog.predefined_catalogs
   DataCatalog.to_dict
   DataCatalog.to_dataframe
   DataCatalog.to_yml
   DataCatalog.export_data
   DataCatalog.get_source_bbox
   DataCatalog.get_source_time_range

Add data sources
----------------

.. autosummary::
   :toctree: ../_generated

   DataCatalog.add_source
   DataCatalog.update
   DataCatalog.from_predefined_catalogs
   DataCatalog.from_yml
   DataCatalog.from_dict

.. _api_data_catalog_get:

Get data
--------

.. autosummary::
   :toctree: ../_generated

   DataCatalog.get_rasterdataset
   DataCatalog.get_geodataset
   DataCatalog.get_geodataframe
   DataCatalog.get_dataframe
   DataCatalog.get_dataset


Predefined data catalog
=======================

.. autosummary::
   :toctree: ../_generated

   PredefinedCatalog
   PredefinedCatalog.get_catalog_file

   predefined_catalog.create_registry_file


DataSource
==========

General
-------

.. autosummary::
   :toctree: ../_generated

   sources.DataSource
   sources.DataSource.summary

RasterDataset
-------------

.. autosummary::
   :toctree: ../_generated

   sources.RasterDatasetSource
   sources.RasterDatasetSource.read_data
   sources.RasterDatasetSource.to_stac_catalog
   sources.RasterDatasetSource.get_bbox
   sources.RasterDatasetSource.get_time_range
   sources.RasterDatasetSource.detect_bbox
   sources.RasterDatasetSource.detect_time_range

GeoDataFrame
------------

.. autosummary::
   :toctree: ../_generated

   sources.GeoDataFrameSource
   sources.GeoDataFrameSource.read_data
   sources.GeoDataFrameSource.to_stac_catalog
   sources.GeoDataFrameSource.get_bbox
   sources.GeoDataFrameSource.detect_bbox

DataFrame
---------

.. autosummary::
   :toctree: ../_generated

   sources.DataFrameSource
   sources.DataFrameSource.read_data
   sources.DataFrameSource.to_stac_catalog

GeoDataset
------------

.. autosummary::
   :toctree: ../_generated

   sources.GeoDatasetSource
   sources.GeoDatasetSource.read_data
   sources.GeoDatasetSource.to_stac_catalog
   sources.GeoDatasetSource.get_bbox
   sources.GeoDatasetSource.detect_bbox

MetaDataResolver
================

General
-------

.. autosummary::
   :toctree: ../_generated

   uri_resolvers.MetaDataResolver
   uri_resolvers.MetaDataResolver.resolve

ConventionResolver
------------------

.. autosummary::
   :toctree: ../_generated

   uri_resolvers.ConventionResolver
   uri_resolvers.ConventionResolver.resolve

RasterTindexResolver
--------------------
.. autosummary::
   :toctree: ../_generated

   uri_resolvers.RasterTindexResolver
   uri_resolvers.RasterTindexResolver.resolve

Driver
======

General
-------

.. autosummary::
   :toctree: ../_generated

   drivers.base_driver.BaseDriver

RasterDataset
-------------

.. autosummary::
   :toctree: ../_generated

   drivers.raster.raster_dataset_driver.RasterDatasetDriver
   drivers.raster.raster_dataset_driver.RasterDatasetDriver.read
   drivers.raster.raster_dataset_driver.RasterDatasetDriver.read_data
   drivers.raster.raster_dataset_driver.RasterDatasetDriver.write

RasterDatasetXarrayDriver
-------------------------

.. autosummary::
   :toctree: ../_generated

   drivers.raster.raster_xarray_driver.RasterDatasetXarrayDriver
   drivers.raster.raster_xarray_driver.RasterDatasetXarrayDriver.read
   drivers.raster.raster_xarray_driver.RasterDatasetXarrayDriver.read_data
   drivers.raster.raster_xarray_driver.RasterDatasetXarrayDriver.write

RasterioDriver
--------------

.. autosummary::
   :toctree: ../_generated

   drivers.raster.rasterio_driver.RasterioDriver
   drivers.raster.rasterio_driver.RasterioDriver.read
   drivers.raster.rasterio_driver.RasterioDriver.write

GeoDataFrame
------------

.. autosummary::
   :toctree: ../_generated

   drivers.geodataframe.geodataframe_driver.GeoDataFrameDriver
   drivers.geodataframe.geodataframe_driver.GeoDataFrameDriver.read
   drivers.geodataframe.geodataframe_driver.GeoDataFrameDriver.read_data
   drivers.geodataframe.geodataframe_driver.GeoDataFrameDriver.write

PyogrioDriver
-------------

.. autosummary::
   :toctree: ../_generated

   drivers.geodataframe.pyogrio_driver.PyogrioDriver
   drivers.geodataframe.pyogrio_driver.PyogrioDriver.read
   drivers.geodataframe.pyogrio_driver.PyogrioDriver.read_data
   drivers.geodataframe.pyogrio_driver.PyogrioDriver.write

GeoDataFrameTableDriver
-----------------------

.. autosummary::
   :toctree: ../_generated

   drivers.geodataframe.table_driver.GeoDataFrameTableDriver
   drivers.geodataframe.table_driver.GeoDataFrameTableDriver.read
   drivers.geodataframe.table_driver.GeoDataFrameTableDriver.read_data
   drivers.geodataframe.table_driver.GeoDataFrameTableDriver.write

DataFrame
---------

.. autosummary::
   :toctree: ../_generated

   drivers.dataframe.dataframe_driver.DataFrameDriver
   drivers.dataframe.dataframe_driver.DataFrameDriver.read
   drivers.dataframe.dataframe_driver.DataFrameDriver.read_data
   drivers.dataframe.dataframe_driver.DataFrameDriver.write

PandasDriver
------------

.. autosummary::
   :toctree: ../_generated

   drivers.dataframe.pandas_driver.PandasDriver
   drivers.dataframe.pandas_driver.PandasDriver.read_data
   drivers.dataframe.pandas_driver.PandasDriver.write

GeoDataFrame
------------

.. autosummary::
   :toctree: ../_generated

   drivers.geodataset.geodataset_driver.GeoDatasetDriver
   drivers.geodataset.geodataset_driver.GeoDatasetDriver.read
   drivers.geodataset.geodataset_driver.GeoDatasetDriver.read_data
   drivers.geodataset.geodataset_driver.GeoDatasetDriver.write

DataAdapter
===========

General
-------

.. autosummary::
   :toctree: ../_generated

   adapters.DataAdapter

RasterDataset
-------------

.. autosummary::
   :toctree: ../_generated

   adapters.RasterDatasetAdapter
   adapters.RasterDatasetAdapter.transform

GeoDataset
----------

.. autosummary::
   :toctree: ../_generated

   adapters.GeoDatasetAdapter
   adapters.GeoDatasetAdapter.transform

GeoDataFrame
------------

.. autosummary::
   :toctree: ../_generated

   adapters.GeoDataFrameAdapter
   adapters.GeoDataFrameAdapter.transform

DataFrame
---------

.. autosummary::
   :toctree: ../_generated

   adapters.dataframe.DataFrameAdapter
   adapters.dataframe.DataFrameAdapter.transform

Dataset
-------

.. autosummary::
   :toctree: ../_generated

   adapters.DatasetAdapter
