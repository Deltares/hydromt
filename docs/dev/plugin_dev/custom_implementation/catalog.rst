.. _custom_data_catalog:

===========================
Custom Data Catalogs
===========================

HydroMT allows you to define and distribute your own **data catalogs**, providing structured access to datasets that are commonly used by your model or plugin.
These catalogs can reference local files, cloud resources, or APIs, and define how each dataset should be loaded and standardized into HydroMT's data model.

Unlike predefined catalogs, custom catalogs do not need to subclass or register a specific class — they are simply YAML-based configurations listing one or more :class:`DataSource` entries.
Each entry defines how HydroMT can find, read, and interpret a dataset.

Predefined Catalogs
--------------------

Creating your own predefined catalog class is possible if you need more advanced behavior, but this is not required for most use cases.
It can also be useful to bundle custom catalogs within a HydroMT plugin for easy distribution and reuse.
For an example of creating and using a custom predefined data catalog, see the module `predefined_catalogs <https://github.com/Deltares/hydromt/blob/main/hydromt/data_catalog/predefined_catalog.py>`_.

Overview
--------

A HydroMT data catalog is a YAML file that lists datasets by name.
Each dataset entry specifies:

- The data type (e.g., `RasterDataset`, `GeoDataset`, or custom type)
- The driver used to read the data
- The data location (URI)
- Optional metadata, adapters, and configuration parameters

These catalogs can be local files (e.g., `my_catalog.yml`) or hosted remotely for shared use.
They can also be distributed as part of a HydroMT plugin.


DataSource: Simple Example
---------------------------

Here's a minimal example of a custom DataSource that defines a single raster dataset:

.. code-block:: yaml

    # my_catalog.yml
    simple_source:
        data_type: RasterDataset
        uri: topography/my_topo.tif
        driver: rasterio

Usage example:

.. code-block:: python

    from hydromt import DataCatalog

    # Load your catalog
    catalog = DataCatalog("my_catalog.yml")

    # Access the DEM dataset
    dem = catalog.get_rasterdataset("simple_source")
    print(dem)

This example shows how HydroMT reads a single raster file directly using the default raster driver.

DataSource: Complex Example
---------------------------

With HydroMT's flexible catalog system, you can define more complex datasets that include adapters for data transformation, versioning, and cloud storage access.
Many fields are optional and can be customized for each dataset entry. For more details, see the :ref:`DataCatalog API <data_catalog_api>` and :ref:`DataSource API <data_source_api>`.
Here is a more complex catalog defining multiple datasets with adapters and cloud storage:

.. code-block:: yaml

    # my_complex_catalog.yml
    hydro_basin_atlas_level12:
        data_type: GeoDataFrame
        version: 10
        uri: hydrography/hydro_atlas/basin_atlas_v10.gpkg
        driver:
            name: pyogrio
            options:
                layer: BasinATLAS_v10_lev12
        metadata:
            category: hydrography
            notes: renaming and units might require some revision
            paper_doi: 10.1038/s41597-019-0300-6
            paper_ref: Linke et al. (2019)
            url: https://www.hydrosheds.org/hydroatlas
            license: CC BY 4.0
            extent:
                bbox:
                    West: -180.0
                    South: -55.988
                    East: 180.001
                    North: 83.626
                crs: 4326

Usage example:

.. code-block:: python

    from hydromt import DataCatalog

    catalog = DataCatalog("my_complex_catalog.yml")

    # Fetch the dataset and automatically apply transformations
    pr = catalog.get_rasterdataset("hydro_basin_atlas_level12", time_range=("2010-01-01", "2010-12-31"))
    print(pr)

    # The returned dataset is an xarray.Dataset, standardized and ready to use
    print(pr.attrs["units"])  # -> 'mm/day'
