.. currentmodule:: hydromt.data_catalog

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
