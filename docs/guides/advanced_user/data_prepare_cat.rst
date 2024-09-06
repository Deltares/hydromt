.. _own_catalog:

Preparing a Data Catalog
========================

**Steps in brief:**

1) Have your (local) dataset ready in one of the supported :ref:`raster
   <raster_formats>`, :ref:`vector <vector_formats>` or :ref:`geospatial time-series
   <geo_formats>`
2) Create your own :ref:`yaml file <data_yaml>` with a reference to your prepared
   dataset following the HydroMT :ref:`data conventions <data_convention>`, see examples
   below.

A detailed description of the yaml file is given below. For more information see
:py:meth:`~hydromt.data_catalog.DataCatalog.from_yml` and examples per :ref:`data type
<data_types>`

.. _data_yaml:

Data catalog yaml file
----------------------

Each data source, is added to a data catalog yaml file with a user-defined name.

A blue print for a dataset called **my_dataset** is shown below. The ``uri``,
``data_type`` and ``driver`` options are required and the ``metadata`` option with the
shown keys is highly recommended. The ``rename``, ``nodata``, ``unit_add`` and
``unit_mult`` options are set per variable (or attribute table column in case of a
GeoDataFrame).

.. literalinclude:: ../../assets/example_catalog.yml
  :language: yaml

.. testsetup:: *

  from hydromt import DataCatalog

.. testcode:: read_catalog
  :hide:

  catalog = DataCatalog(fallback_lib=None)  # do not read default catalog
  catalog.from_yml("docs/assets/example_catalog.yml")

The yaml file has an *optional* global **metadata** data section:

- **roots** (optional): root folders for all the data sources in the yaml file. If not
  provided the folder of where the yaml file is located will be used as root. This is
  used in combination with each data source **uri** argument to avoid repetition. The
  roots listed will be checked in the order they are provided. The first one to be found
  to exist will be used as the actual root. This should be used for cross platform and
  cross machine compatibility only, as can be seen above. Note that in the end only one
  of the roots will be used, so all data should still be located in the same folder
  tree.
- **version** (recommended): data catalog version
- **hydromt_version** (recommended): range of hydromt version that can read this
  catalog. Format should be acording to `PEP 440
  <https://peps.python.org/pep-0440/#version-specifiers>`_.
- **category** (optional): used if all data source in catalog belong to the same
  category. Usual categories within HydroMT are *geography*, *topography*,
  *hydrography*, *meteo*, *landuse*, *ocean*, *socio-economic*, *observed data* but the
  user is free to define its own categories.

The following are **required data source arguments**:

- **data_type**: type of input data. Either *RasterDataset*, *GeoDataset*, *Dataset*
  *GeoDataFrame* or *DataFrame*.
- **driver**: data_type specific :Class:`Driver` to read a dataset. If the default
  settings of a driver are sufficient, then a string with the name of the driver is
  enough. Otherwise, a dictionary with the driver class properties can be used. Refer to
  the :Class:`Driver` documentation to see which options are available.
- **uri**: URI pointing to where the data can be queried. Relative paths are combined
  with the global ``root`` option of the yaml file (if available) or the directory of
  the yaml file itself. To read multiple files in a single dataset (if supported by the
  driver) a string glob in the form of ``"path/to/my/files/*.nc"`` can be used. The
  filenames can be further specified with ``{variable}``, ``{year}`` and ``{month}``
  keys to limit which files are being read based on the get_data request in the form of
  ``"path/to/my/files/{variable}_{year}_{month}.nc"``. Note that ``month`` is by default
  *not* zero-padded (e.g. January 2012 is stored as
  ``"path/to/my/files/{variable}_2012_1.nc"``). Users can optionally add a formatting
  string to define how the key should be read. For example, in a path written as
  ``"path/to/my/files/{variable}_{year}_{month:02d}.nc"``, the month always has two
  digits and is zero-padded for Jan-Sep (e.g. January 2012 is stored as
  ``"path/to/my/files/{variable}_2012_01.nc"``).

A full list of **optional data source arguments** is given below

- **version** (recommended): data source version
- **provider** (recommended): data source provider
- **metadata** (recommended): additional information on the dataset. In
  :Class:`SourceMetaData` there are many different metadata options available. Some
  metadata properties, like the `crs`, `nodata` or `temporal_extent` and
  `spatial_extent` can help HydroMT more efficiently read the data. Good meta data
  includes a *source_url*, *source_license*, *source_version*, *paper_ref*, *paper_doi*,
  *category*, etc. These are added to the data attributes. Usual categories within
  HydroMT are *geography*, *topography*, *hydrography*, *meteo*, *landuse*, *ocean*,
  *socio-economic*, *observed data* but the user is free to define its own categories.
- **data_adapter**: the data adapter harmonizes the data so that within HydroMT, there
  are strong conventions on for example variable naming, :ref:`HydroMT variable naming
  conventions <data_convention>` and variable names. :ref:`recognized dimension names
  <dimensions>`. There are multiple different parameters available for each
  :Class:`DataAdapter`.
- **placeholder** (optional): this argument can be used to generate multiple sources
  with a single entry in the data catalog file. If different files follow a logical
  nomenclature, multiple data sources can be defined by iterating through all possible
  combinations of the placeholders. The placeholder names should be given in the source
  name and the path and its values listed under the placeholder argument.
- **variants** (optional): This argument can be used to generate multiple sources with
  the same name, but from different providers or versions. Any keys here are essentially
  used to extend/overwrite the base arguments.

Data variants
-------------

Data variants are used to define multiple data sources with the same name, but from
different providers or versions. Below, we show an example of a data catalog for a
RasterDataset with multiple variants of the same data source (esa_worldcover), but this
works identical for other data types. Here, the *metadata*, *data_type*, *driver* and
are common arguments used for all variants. The variant arguments are used
to extend and/or overwrite the common arguments, creating new sources.

.. code-block:: yaml

  esa_worldcover:
    metadata:
      crs: 4326
    data_type: RasterDataset
    driver:
      name: raster
      filesystem: local
    variants:
      - provider: local
        version: 2021
        uri: landuse/esa_worldcover_2021/esa-worldcover.vrt
      - provider: local
        version: 2020
        uri: landuse/esa_worldcover/esa-worldcover.vrt
      - provider: aws
        version: 2020
        uri: s3://esa-worldcover/v100/2020/ESA_WorldCover_10m_2020_v100_Map_AWS.vrt
        driver:
          name: raster
          filesystem: s3

To request a specific variant, the variant arguments can be used as keyword arguments to
the `DataCatalog.get_rasterdataset` method, see code below. By default the newest
version from the last provider is returned when requesting a data source with specific
version or provider. Requesting a specific version from a HydroMT configuration file is
also possible, see :ref:`model_config`.


.. code-block:: python

  from hydromt import DataCatalog
  dc = DataCatalog().from_yml("data_catalog.yml")
  # get the default version. This will return the latest (2020) version from the last
  # provider (aws)
  ds = dc.get_rasterdataset("esa_worldcover")
  # get a 2020 version. This will return the 2020 version from the last provider (aws)
  ds = dc.get_rasterdataset("esa_worldcover", version=2020)
  # get a 2021 version. This will return the 2021 version from the local provider as
  # this verion is not available from aws .
  ds = dc.get_rasterdataset("esa_worldcover", version=2021)
  # get the 2020 version from the local provider
  ds = dc.get_rasterdataset("esa_worldcover", version=2020, provider="local")
