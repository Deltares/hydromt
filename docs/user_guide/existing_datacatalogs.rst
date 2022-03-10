
.. _existing_catalog:

Using a pre-defined Data Catalog
================================

The documentation contains a list of (global) datasets which can be used with various HydroMT models and workflows. 
The deltares_data catalog is only available within the Deltares network. However a selection of this data for a the 
Piave basin (north Italy) is available online and will be downloaded to ~/.hydromt_data/ if selected or when no data catalog is provided. 
Local or other datasets can also be included by extending the data catalog with new .yml files. 

Python usage 
^^^^^^^^^^^^

Basic usage to read a dataset in python using the HydroMT data catalog requires two steps:
 - Initialize a DataCatalog with references to user- or pre-defined data catalog files
 - Use one of the get_* methods to access the data.


Example usage to retrieve a raster dataset

.. code-block:: python

    import hydromt
    data_cat = hydromt.DataCatalog(data_libs=r'/path/to/data-catalog.yml')
    ds = data_cat.get_rasterdataset('source_name', bbox=[xmin, ymin, xmax, ymax])  # returns xarray.dataset


Pre-defined Data Catalogs
^^^^^^^^^^^^^^^^^^^^^^^^^

Below are drop down lists with datasets per pre-defined data catalog for use with HydroMT. 
The summary per dataset contains links to the online source and available literature. 

.. include:: ../_generated/deltares_data.rst
