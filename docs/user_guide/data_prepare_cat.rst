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
``kwargs`` contain any options passed to different drivers.

.. code-block:: yaml

    meta:
      root: /path/to/data_root/
      version: version
    my_dataset:
      crs: EPSG/WKT
      data_type: RasterDataset/GeoDataset/GeoDataFrame
      driver: raster/raster_tindex/netcdf/zarr/vector/vector_table
      kwargs:
        key: value
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
      placeholders: 
        [placeholder_key: [placeholder_values]]
      rename:
        old_variable_name: new_variable_name   
      unit_add:
        new_variable_name: value
      unit_mult:
        new_variable_name: value

The yaml file has an *optional* global **meta** data section:

- **root** (optional): root folder for all the data sources in the yaml file. 
  If not provided the folder of where the yaml file is located will be used as root.
  This is used in combination with each data source **path** argument to avoid repetition.
- **version** (recommended): data catalog version; we recommend `calendar versioning <https://calver.org/>`
- **category** (optional): used if all data source in catalog belong to the same category. Usual categories within HydroMT are 
  *geography*, *topography*, *hydrography*, *meteo*, *landuse*, *ocean*, *socio-economic*, *observed data* 
  but the user is free to define its own categories.

The following are **required data source arguments**: 

- **data_type**: type of input data. Either *RasterDataset*, *GeoDataset* or *GeoDataFrame*.
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

- **crs** (required if missing in the data): EPSG code or WKT string of the reference coordinate system of the data. 
- **kwargs**: pairs of key value arguments to pass to the driver specific open data method (eg xr.open_mfdataset for netdcf raster, see the full list below).
  Only used if not crs can be inferred from the input data.
- **meta** (recommended): additional information on the dataset organized in a sub-list. 
  Good meta data includes a *source_url*, *source_license*, *source_version*, *paper_ref*, *paper_doi*, *category*, etc. These are added to the data attributes.
  Usual categories within HydroMT are *geography*, *topography*, *hydrography*, *meteo*, *landuse*, *ocean*, *socio-economic*, *observed data* 
  but the user is free to define its own categories. 
- **nodata** (required if missing in the data): nodata value of the input data. For Raster- and GeoDatasets this is only used if not inferred from the original input data. 
  For GeoDataFrame provided nodata values are converted to nan values.
- **placeholder**: this argument can be used to generate multiple sources with a single entry in the data catalog file. If different files follow a logical
  nomenclature, multiple data sources can be defined by iterating through all possible combinations of the placeholders. The placeholder names should be given in the 
  source name and the path and its values listed under the placeholder argument.
- **rename**: pairs of variable names in the input data (*old_variable_name*) and the corresponding 
  :ref:`HydroMT variable naming conventions <data_convention>` and :ref:`recognized dimension names <dimensions>` (*new_variable_name*). 
- **units** (optional and for *RasterDataset* only). specify the units of the input data: supported are [m3], [m], [mm], and [m3/s].
  This option is used *only* for the forcing of the Delwaq models in order to do specific unit conversions that cannot be handled from simple 
  addition or multiplication (e.g. conversion from mm water equivalent to m3/s of water which requires a multiplication by each grid cell area and not a fixed number).
- **unit_add**: add or substract a value to the input data for unit conversion (e.g. -273.15 for conversion of temperature from Kelvin to Celsius). 
- **unit_mult**: multiply the input data by a value for unit conversion (e.g. 1000 for conversion from m to mm of precipitation).

.. note::

  The **alias** argument will be deprecated and should no longer be used, see `github issue for more information <https://github.com/Deltares/hydromt/issues/148>`_