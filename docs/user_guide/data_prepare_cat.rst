.. _own_catalog:

Preparing a Data Catalog 
========================

**Steps in brief:**

1) Have your (local) dataset ready in one of the supported :ref:`raster <raster_formats>`, 
   :ref:`vector <vector_formats>` or :ref:`geospatial time-series <geo_formats>`
2) Create your own :ref:`yaml file <data_yaml>` with a reference to your prepared dataset following 
   the HydroMT :ref:`data conventions <data_convention>`, see examples below.

A detailed description of the yaml file is given below.
For more information see :py:meth:`~hydromt.data_adapter.DataCatalog.from_yml`
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

    my_dataset:
      path: /path/to/my_dataset.extension
      data_type: RasterDataset/GeoDataset/GeoDataFrame
      driver: raster/raster_tindex/netcdf/zarr/vector/vector_table
      crs: EPSG/WKT
      kwargs:
        key: value
      rename:
        old_variable_name: new_variable_name   
      nodata:
        new_variable_name: value
      unit_add:
        new_variable_name: value
      unit_mult:
        new_variable_name: value
      meta:
        source_url: zenodo.org/my_dataset
        source_license: CC-BY-3.0
        source_version: vX.X
        paper_ref: Author et al. (year)
        paper_doi: doi
        category: category


A full list of **data source options** is given below

- **path** (required): path to the data file. 
  Relative paths are combined with the global ``root`` option of the yaml file (if available) or the directory of the yaml file itself. 
  To read multiple files in a single dataset (if supported by the driver) a string glob in the form of ``"path/to/my/files/*.nc"`` can be used.
  The filenames can be further specified with ``{variable}``, ``{year}`` and ``{month}`` keys to limit which files are being read 
  based on the get_data request in the form of ``"path/to/my/files/{variable}_{year}_{month}.nc"``. Users can additionally add a formatting string to mention how 
  the key should be read. For example, a path written as ``"path/to/my/files/{variable}_{year:04d}_{month:02d}.nc"`` will be read as ``{year}`` having four digits and 
  ``{month}`` two digits (e.g. January 2012 is stored as ``"path/to/my/files/{variable}_2012_01.nc"``).
- **data_type** (required): type of input data. Either *RasterDataset*, *GeoDataset* or *GeoDataFrame*.
- **crs** (required if missing in the data): EPSG code or WKT string of the reference coordinate system of the data. 
- **driver** (required): data_type specific driver to read a dataset, see overview below.
- **kwargs** (optional): pairs of key value arguments to pass to the driver specific open data method (eg xr.open_mfdataset for netdcf raster, see the full list below).
  Only used if not crs can be inferred from the input data.
- **rename** (optional): pairs of variable names in the input data (*old_variable_name*) and the corresponding 
  :ref:`HydroMT variable naming conventions <data_convention>` and :ref:`recognized dimension names <dimensions>` (*new_variable_name*). 
- **nodata** (optional): nodata value of the input data. For Raster- and GeoDatasets this is only used if not inferred from the original input data. 
  For GeoDataFrame provided nodata values are converted to nan values.
- **unit_add** (optional): add or substract a value to the input data for unit conversion (e.g. -273.15 for conversion of temperature from Kelvin to Celsius). 
- **unit_mult** (optional): multiply the input data by a value for unit conversion (e.g. 1000 for conversion from m to mm of precipitation).
- **units** (optional and for *RasterDataset* only). specify the units of the input data: supported are [m3], [m], [mm], and [m3/s].
  This option is used *only* for the forcing of the Delwaq models in order to do specific unit conversions that cannot be handled from simple 
  addition or multiplication (e.g. conversion from mm water equivalent to m3/s of water which requires a multiplication by each grid cell area and not a fixed number).
- **meta** (optional): additional information on the dataset organized in a sub-list. 
  Good meta data includes a *source_url*, *source_license*, *source_version*, *paper_ref*, *paper_doi*, *category*, etc. These are added to the data attributes.
  Usual categories within HydroMT are *geography*, *topography*, *hydrography*, *meteo*, *landuse*, *ocean*, *socio-economic*, *observed data* 
  but the user is free to define its own categories. 

Apart from the data entries, the yaml file also has **global options**:

- **root** (optional): root folder for all the data sources in the yaml file. 
  If not provided the folder of where the yaml fil is located will be used as root.
  This is used in combination with each data source **path** argument to avoid repetition.


Placeholder and alias
---------------------
There are two convenience options to limit repetition between data sources in data catalog files:

- The ``placeholder`` argument can be used to generate multiple sources with a single entry in the data catalog file. If different files follow a logical
  nomenclature, multiple data sources can be defined by iterating through all possible combinations of the placeholders. The placeholder names should be given in the 
  source name and the path and its values listed under the placeholder argument, see example below with an *epoch* and *epsg* placeholders.
- The ``alias`` argument can be used to define a data source under a second short name, or to avoid repeating large sections with the same meta-data.
  If an alias is provided all information from the alias source is used to read the data except for the info that is overwritten by the current data source. 
  The alias source should also be provided in the same file. Note that this only works at the first level of arguments, if e.g. the rename option is used in 
  the current data source it overwrites all rename entries of the alias data source. In the example below *ghs_pop* is short for a specific version (epoch=2015; epsg=54009)
  of that dataset. 

.. code-block:: yaml

  ghs_pop:
    alias: ghs_pop_2015_54009
  ghs_pop_{epoch}_{epsg}:
    data_type: RasterDataset
    driver: raster
    kwargs:
      chunks: {x: 3600, y: 3600}
    placeholder:
      epoch: [2015, 2020]
      epsg: [54009, 4326]
    meta:
      category: socio-economic
      paper_doi: 10.2905/0C6B9751-A71F-4062-830B-43C9F432370F
      paper_ref: Schiavina et al (2019)
      source_author: JRC-ISPRA EC
      source_license: https://data.jrc.ec.europa.eu/licence/com_reuse
      source_url: https://data.jrc.ec.europa.eu/dataset/0c6b9751-a71f-4062-830b-43c9f432370f
      source_version: R2019A_v1.0
    path: socio_economic/ghs/GHS_POP_E{epoch}_GLOBE_R2019A_{epsg}.tif
