.. currentmodule:: hydromt.data_catalog

.. _data_catalog_api:

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
   DataCatalog.to_yml
   DataCatalog.to_stac_catalog
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
   DataCatalog.from_stac_catalog


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
   PredefinedCatalog.versions
   PredefinedCatalog.get_catalog_file

   predefined_catalog.create_registry_file
