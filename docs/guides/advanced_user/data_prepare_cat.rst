.. _own_catalog:

Preparing a Data Catalog
========================

**Steps in brief:**

1) Have your (local) dataset ready in one of the supported :ref:`raster <raster_formats>`,
   :ref:`vector <vector_formats>` or :ref:`geospatial time-series <geo_formats>`
2) Create your own :ref:`yaml file <data_yaml>` with a reference to your prepared dataset following
   the HydroMT :ref:`data conventions <data_convention>`, see examples below.

A detailed description of the yaml file is given below.
For more information see :py:meth:`~hydromt.data_catalog.DataCatalog.from_yml`
and examples per :ref:`data type <data_types>`

.. _data_yaml:

Data catalog yaml file
----------------------

Each data source, is added to a data catalog yaml file with a user-defined name.

A blue print for a dataset called **my_dataset** is shown below.
The ``path``, ``data_type`` and ``driver`` options are required and the ``meta`` option with the shown keys is highly recommended.
The ``rename``, ``nodata``, ``unit_add`` and ``unit_mult`` options are set per variable (or attribute table column in case of a GeoDataFrame).
``driver_kwargs`` contain any options passed to different drivers.

.. code-block:: yaml

    meta:
      roots:
        - /linux/path/to/data_root/
        - C:\Windows\path\to\data_root
      version: version
      name: data_catalog_name
    my_dataset:
      crs: EPSG/WKT
      data_type: RasterDataset/GeoDataset/GeoDataFrame/DataFrame
      driver: raster/raster_tindex/netcdf/zarr/vector/vector_table/csv/xlsx/xls
      driver_kwargs:
        key: value
      filesystem: local/gcs/s3/http
      meta:
        source_url: zenodo.org/my_dataset
        source_license: CC-BY-3.0
        source_version: vX.X
        paper_ref: Author et al. (year)
        paper_doi: doi
        category: category
      nodata:
        new_variable_name: value
      path: /absolut_path/to/my_dataset.extension OR relative_path/to_my_dataset.extension
      rename:
        old_variable_name: new_variable_name
      unit_add:
        new_variable_name: value
      unit_mult:
        new_variable_name: value
      attrs:
        new_variable_name:
          units: value
          long_name: variable_long_name
      zoom_levels:
        [zoom_level: zoom_resolution]

The yaml file has an *optional* global **meta** data section:

- **roots** (optional): root folders for all the data sources in the yaml file.
  If not provided the folder of where the yaml file is located will be used as root.
  This is used in combination with each data source **path** argument to avoid repetition.
  The roots listed will be checked in the order they are provided. The first one to be found to exist will be used as the actual root.
  This should be used for cross platform and cross machine compatibility only, as can be seen above. Note that in the end
  only one of the roots will be used, so all data should still be located in the same folder tree.
- **version** (recommended): data catalog version; we recommend `calendar versioning <https://calver.org/>`_
- **hydromt_version** (recommended): range of hydromt version that can read this catalog. Format should be acording to `PEP 440 <https://peps.python.org/pep-0440/#version-specifiers>`_.
- **category** (optional): used if all data source in catalog belong to the same category. Usual categories within HydroMT are
  *geography*, *topography*, *hydrography*, *meteo*, *landuse*, *ocean*, *socio-economic*, *observed data*
  but the user is free to define its own categories.

The following are **required data source arguments**:

- **data_type**: type of input data. Either *RasterDataset*, *GeoDataset*, *GeoDataFrame* or *DataFrame*.
- **driver**: data_type specific driver to read a dataset, see overview below.
- **path**: path to the data file.
  Relative paths are combined with the global ``root`` option of the yaml file (if available) or the directory of the yaml file itself.
  To read multiple files in a single dataset (if supported by the driver) a string glob in the form of ``"path/to/my/files/*.nc"`` can be used.
  The filenames can be further specified with ``{variable}``, ``{year}`` and ``{month}`` keys to limit which files are being read
  based on the get_data request in the form of ``"path/to/my/files/{variable}_{year}_{month}.nc"``.
  Note that ``month`` is by default *not* zero-padded (e.g. January 2012 is stored as ``"path/to/my/files/{variable}_2012_1.nc"``).
  Users can optionally add a formatting string to define how the key should be read.
  For example, in a path written as ``"path/to/my/files/{variable}_{year}_{month:02d}.nc"``,
  the month always has two digits and is zero-padded for Jan-Sep (e.g. January 2012 is stored as ``"path/to/my/files/{variable}_2012_01.nc"``).

A full list of **optional data source arguments** is given below

- **driver_kwargs**: pairs of key value arguments to pass to the driver specific open data method
  (eg xr.open_mfdataset for netdcf raster, see the full list below).
  *NOTE*: New with HydroMT v0.7.2 (was called *kwargs* before)
- **filesystem** (optional): specify at what filesystem the data is stored. This is used to select the correct protocol to
  access different filesystems (e.g. local, gcs, s3, http). If not provided the filesystem is inferred from the path.
  See `fsspec <https://filesystem-spec.readthedocs.io/en/latest/index.html>`_ for more available protocols.
- **storage_options** (optional): Additional arguments to pass to the filesystem protocol, these are protocol specific.
  *NOTE*: New in HydroMT v0.8.1
- **version** (recommended): data source version
  *NOTE*: New in HydroMT v0.8.1
- **provider** (recommended): data source provider
  *NOTE*: New in HydroMT v0.8.1
- **meta** (recommended): additional information on the dataset organized in a sub-list.
  Good meta data includes a *source_url*, *source_license*, *source_version*, *paper_ref*, *paper_doi*, *category*, etc. These are added to the data attributes.
  Usual categories within HydroMT are *geography*, *topography*, *hydrography*, *meteo*, *landuse*, *ocean*, *socio-economic*, *observed data*
  but the user is free to define its own categories.
- **nodata** (required if missing in the data): nodata value of the input data. For Raster- and GeoDatasets this is only used if not inferred from the original input data.
  For GeoDataFrame provided nodata values are converted to nan values.
- **rename**: pairs of variable names in the input data (*old_variable_name*) and the corresponding
  :ref:`HydroMT variable naming conventions <data_convention>` and :ref:`recognized dimension names <dimensions>` (*new_variable_name*).
- **unit_add**: add or substract a value to the input data for unit conversion (e.g. -273.15 for conversion of temperature from Kelvin to Celsius).
- **unit_mult**: multiply the input data by a value for unit conversion (e.g. 1000 for conversion from m to mm of precipitation).
- **attrs** (optional): This argument allows for setting attributes like the unit or long name to variables.
  *NOTE*: New in HydroMT v0.7.2
- **placeholder** (optional): this argument can be used to generate multiple sources with a single entry in the data catalog file. If different files follow a logical
  nomenclature, multiple data sources can be defined by iterating through all possible combinations of the placeholders. The placeholder names should be given in the
  source name and the path and its values listed under the placeholder argument.
- **variants** (optional): This argument can be used to generate multiple sources with the same name, but from different providers or versions.
  Any keys here are essentially used to extend/overwrite the base arguments.

The following are **optional data source arguments** for *RasterDataset*, *GeoDataFrame*, and *GeoDataset*:

- **crs** (required if missing in the data): EPSG code or WKT string of the reference coordinate system of the data.
  Only used if not crs can be inferred from the input data.

The following are **optional data source arguments** for *RasterDataset*:

- **zoom_level** (optional): this argument can be used for a *RasterDatasets* that contain multiple zoom levels of different resolution.
  It should contain a list of numeric zoom levels that correspond to the `zoom_level` key in file path, e.g.,  ``"path/to/my/files/{zoom_level}/data.tif"``
  and corresponding resolution, expressed in the unit of the data crs.
  The *crs* argument is therefore required when using zoom_levels to correctly interpret the unit of the resolution.
  The required zoom level can be requested from HydroMT as argument to the `DataCatalog.get_rasterdataset` method,
  see `Reading tiled raster data with different zoom levels <../_examples/working_with_tiled_raster_data.ipynb>`_.

.. note::

  The **alias** argument will be deprecated and should no longer be used, see
  `github issue for more information <https://github.com/Deltares/hydromt/issues/148>`_

.. warning::

  Using cloud data is still experimental and only supported for *DataFrame*, *RasterDataset* and
  *Geodataset* with *zarr*. *RasterDataset* with *raster* driver is also possible
  but in case of multiple files (mosaic) we strongly recommend using a vrt file for speed and computation efficiency.

Data variants
-------------

Data variants are used to define multiple data sources with the same name, but from different providers or versions.
Below, we show an example of a data catalog for a RasterDataset with multiple variants of the same data source (esa_worldcover),
but this works identical for other data types.
Here, the *crs*, *data_type*, *driver* and *filesystem* are common arguments used for all variants.
The variant arguments are used to extend and/or overwrite the common arguments, creating new sources.

.. code-block:: yaml

  esa_worldcover:
    crs: 4326
    data_type: RasterDataset
    driver: raster
    filesystem: local
    variants:
      - provider: local
        version: 2021
        path: landuse/esa_worldcover_2021/esa-worldcover.vrt
      - provider: local
        version: 2020
        path: landuse/esa_worldcover/esa-worldcover.vrt
      - provider: aws
        version: 2020
        path: s3://esa-worldcover/v100/2020/ESA_WorldCover_10m_2020_v100_Map_AWS.vrt
        filesystem: s3


To request a specific variant, the variant arguments can be used as keyword arguments
to the `DataCatalog.get_rasterdataset` method, see code below.
By default the newest version from the last provider is returned when requesting a data
source with specific version or provider.
Requesting a specific version from a HydroMT configuration file is also possible, see :ref:`model_config`.

.. code-block:: python

  from hydromt import DataCatalog
  dc = DataCatalog.from_yml("data_catalog.yml")
  # get the default version. This will return the latest (2020) version from the last provider (aws)
  ds = dc.get_rasterdataset("esa_worldcover")
  # get a 2020 version. This will return the 2020 version from the last provider (aws)
  ds = dc.get_rasterdataset("esa_worldcover", version=2020)
  # get a 2021 version. This will return the 2021 version from the local provider as this verion is not available from aws .
  ds = dc.get_rasterdataset("esa_worldcover", version=2021)
  # get the 2020 version from the local provider
  ds = dc.get_rasterdataset("esa_worldcover", version=2020, provider="local")
