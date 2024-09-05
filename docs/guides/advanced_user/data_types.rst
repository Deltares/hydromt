.. _data_types:

.. currentmodule:: hydromt.data_catalog.drivers

Supported data types
====================

HydroMT currently supports the following data types:

- :ref:`RasterDataset <RasterDataset>`: static and dynamic raster (or gridded) data
- :ref:`GeoDataFrame <GeoDataFrame>`: static vector data
- :ref:`GeoDataset <GeoDataset>`: dynamic point location data
- :ref:`Dataset <Dataset>`:  non-spatial n-dimensional data
- :ref:`DataFrame <DataFrame>`: 2D tabular data

Internally the RasterDataset, GeoDataset, and Dataset are represented by
:py:class:`xarray.Dataset` objects, the GeoDataFrame by
:py:class:`geopandas.GeoDataFrame`, and the DataFrame by :py:class:`pandas.DataFrame`.
We use drivers, typically from third-party packages and sometimes wrapped in HydroMT
functions, to parse many different file formats to this standardized internal data
representation.

.. note::

  It is also possible to create your own driver. See at :ref:`Custom Driver<custom_driver>`

.. _dimensions:

Recognized dimension names
--------------------------

- **time**: time or date stamp ["time"].
- **x**: x coordinate ["x", "longitude", "lon", "long"].
- **y**: y-coordinate ["y", "latitude", "lat"].

.. _RasterDataset:

Raster data (RasterDataset)
---------------------------

- :ref:`Single variable GeoTiff raster <GeoTiff>`
- :ref:`Multi variable Virtual Raster Tileset (VRT) <VRT>`
- :ref:`Tiled raster dataset <Tile>`
- :ref:`Netcdf raster dataset <NC_raster>`


.. _raster_formats:

.. list-table::
   :widths: 17, 25, 30
   :header-rows: 1

   * - Driver
     - File formats
     - Comments
   * - :py:class:`raster <raster.rasterio_driver.RasterioDriver>`
     - GeoTIFF, ArcASCII, VRT, etc. (see `GDAL formats <http://www.gdal.org/formats_list.html>`_)
     - Based on :py:func:`xarray.open_rasterio`
       and :py:func:`rasterio.open`
   * - :py:class:`raster <raster.rasterio_driver.RasterioDriver>` with the
       :py:class:`raster_tindex <hydromt.data_catalog.uri_resolvers.raster_tindex_resolver.RasterTindexResolver>` resolver
     - raster tile index file (see `gdaltindex <https://gdal.org/programs/gdaltindex.html>`_)
     - Options to merge tiles via `options -> mosaic_kwargs`.
   * - :py:class:`raster_xarray <raster.raster_xarray_driver.RasterDatasetXarrayDriver>`
     - NetCDF and Zarr
     - required y and x dimensions


.. _GeoTiff:

**Single variable GeoTiff raster**

Single raster files are parsed to a **RasterDataset** based on the **raster** driver.
This driver supports 2D raster for which the dimensions are names "x" and "y".
A potential third dimension is called "dim0".
The variable name is based on the filename, in this case `"GLOBCOVER_200901_200912_300x300m"`.
The `chunks` key-word argument is passed to :py:meth:`~hydromt.io.open_mfraster`
and allows lazy reading of the data.

.. literalinclude:: ../../assets/data_types/single_variable_geotiff_raster.yml
   :language: yaml

.. testsetup:: *

  from hydromt import DataCatalog

.. testcode:: geotiff
  :hide:

  catalog_path = "docs/assets/data_types/single_variable_geotiff_raster.yml"

  catalog = DataCatalog(fallback_lib=None)  # do not read default catalog
  catalog.from_yml(catalog_path)

.. _VRT:

Multi-variable Virtual Raster Tileset (VRT)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Multiple raster layers from different files are parsed using the **raster** driver.
Each raster becomes a variable in the resulting RasterDataset based on its filename.
The path to multiple files can be set using a sting glob or several keys,
see description of the `uri` argument in the :ref:`yaml file description <data_yaml>`.
Note that the rasters should have identical grids.

Here multiple .vrt files (dir.vrt, bas.vrt, etc.) are combined based on their variable
name into a single dataset with variables flwdir, basins, etc. Other multiple file
raster datasets (e.g. GeoTIFF files) can be read in the same way. VRT files are useful
for large raster datasets which are often tiled and can be combined using
`gdalbuildvrt. <https://gdal.org/programs/gdalbuildvrt.html>`_


.. literalinclude:: ../../assets/data_types/vrt_raster_dataset.yml
   :language: yaml

.. testcode:: geotiff
  :hide:

  catalog_path = "docs/assets/data_types/vrt_raster_dataset.yml"

  catalog = DataCatalog(fallback_lib=None)  # do not read default catalog
  catalog.from_yml(catalog_path)

.. _Tile:

Tiled raster dataset
^^^^^^^^^^^^^^^^^^^^

Tiled index datasets are parsed using the
:py:Class:`raster_tindex <hydromt.data_catalog.uri_resolvers.raster_tindex_resolver.RasterTindexResolver>`
:py:class:`~hydromt.data_catalog.uri_resolvers.uri_resolver.URIResolver`.
This data format is used to combine raster tiles with different CRS projections.
A polygon vector file (e.g. GeoPackage) is used to make a tile index with the spatial
footprints of each tile. When reading a spatial slice of this data the files with
intersecting footprints will be merged together in the CRS of the most central tile.
Use `gdaltindex <https://gdal.org/programs/gdaltindex.html>`_ to build an excepted tile index file.

Here a GeoPackage with the tile index referring to individual GeoTiff raster tiles is used.
The `mosaic_kwargs` are passed to :py:meth:`~hydromt._io._open_raster_from_tindex` to
set the resampling `method`. The name of the column in the tile index attribute table
`tileindex` which contains the raster tile file names is set in the `driver.options``
(to be directly passed as an argument to
:py:meth:`~hydromt._io._open_raster_from_tindex`).

.. literalinclude:: ../../assets/data_types/tiled_raster_dataset.yml
   :language: yaml

.. testcode:: geotiff
  :hide:

  catalog_path = "docs/assets/data_types/tiled_raster_dataset.yml"

  catalog = DataCatalog(fallback_lib=None)  # do not read default catalog
  catalog.from_yml(catalog_path)

.. NOTE::

  Tiled raster datasets are not read lazily as different tiles have to be merged
  together based on their values. For fast access to large raster datasets, other
  formats might be more suitable.

.. _NC_raster:

Netcdf raster dataset
^^^^^^^^^^^^^^^^^^^^^

Netcdf and Zarr raster data are typically used for dynamic raster data and
parsed using the **netcdf** and **zarr** drivers.
A typical raster netcdf or zarr raster dataset has the following structure with
two ("y" and "x") or three ("time", "y" and "x") dimensions.
See list of recognized dimensions_ names.

.. code-block:: console

    Dimensions:      (latitude: NY, longitude: NX, time: NT)
    Coordinates:
      * longitude    (longitude)
      * latitude     (latitude)
      * time         (time)
    Data variables:
        temp         (time, latitude, longitude)
        precip       (time, latitude, longitude)


To read a raster dataset from a multiple file netcdf archive the following data entry
is used, where the `options` are passed to :py:func:`xarray.open_mfdataset`
(or :py:func:`xarray.open_zarr` for zarr data).
In case the CRS cannot be inferred from the netcdf metadata it should be defined with
the `crs` `metadata`` here.
The path to multiple files can be set using a sting glob or several keys,
see description of the `uri` argument in the :ref:`yaml file description <data_yaml>`.
In this example additional renaming and unit conversion preprocessing steps are added to
unify the data to match the HydroMT naming and unit :ref:`terminology <terminology>`.

.. literalinclude:: ../../assets/data_types/netcdf_raster_dataset.yml
   :language: yaml

.. testcode:: geotiff
  :hide:

  catalog_path = "docs/assets/data_types/netcdf_raster_dataset.yml"

  catalog = DataCatalog(fallback_lib=None)  # do not read default catalog
  catalog.from_yml(catalog_path)

Preprocess functions when combining multiple files
""""""""""""""""""""""""""""""""""""""""""""""""""

In :py:func:`xarray.open_mfdataset`, xarray allows for a **preprocess** function to be
run before merging several netcdf files together. In hydroMT, some preprocess functions
are available and can be passed through the options in the same way as any
xr.open_mfdataset options. These preprocess functions are found at
:py:obj:`hydromt.data_catalog.preprocessing.py`

.. _GeoDataFrame:

Vector data (GeoDataFrame)
--------------------------

- :ref:`GeoPackage spatial vector data <GPKG_vector>`
- :ref:`Point vector from text delimited data <textdelimited_vector>`

.. _vector_formats:

.. list-table::
   :widths: 17, 25, 30
   :header-rows: 1

   * - Driver
     - File formats
     - Comments
   * - :py:class:`pyogrio <geodataframe.pyogrio_driver.PyogrioDriver>`
     - ESRI Shapefile, GeoPackage, GeoJSON, etc.
     - Point, Line and Polygon geometries. Uses :py:func:`pyogrio.read_dataframe`
   * - :py:class:`geodataframe_table <geodataframe.table_driver.GeoDataFrameTableDriver>`
     - CSV, XY, PARQUET and EXCEL.
     - Point geometries only. Uses :py:meth:`~hydromt._io._open_vector_from_table`

.. _GPKG_vector:

GeoPackage spatial vector data
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Spatial vector data is parsed to a **GeoDataFrame** using the **vector** driver.
For large spatial vector datasets we recommend the GeoPackage format as it includes a
spatial index for fast filtering of the data based on spatial location. An example is
shown below. Note that the rename, ``unit_mult``, ``unit_add`` and ``nodata`` options refer to
columns of the attribute table in case of a GeoDataFrame.

.. literalinclude:: ../../assets/data_types/gpkg_geodataframe.yml
   :language: yaml

.. testsetup:: *

  from hydromt import DataCatalog

.. testcode:: geotiff
  :hide:

  catalog_path = "docs/assets/data_types/gpkg_geodataframe.yml"

  catalog = DataCatalog(fallback_lib=None)  # do not read default catalog
  catalog.from_yml(catalog_path)

.. _textdelimited_vector:

Point vector from text delimited data
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Tabulated point vector data files can be parsed to a **GeoDataFrame** with the **vector_table**
driver. This driver reads CSV (or similar delimited text files), EXCEL and XY
(white-space delimited text file without headers) files. See this list of :ref:`dimension names <dimensions>`
for recognized x and y column names.

A typical CSV point vector file is given below. A similar setup with headers
can be used to read other text delimited files or excel files.

.. code-block:: console

    index, x, y, col1, col2
    <ID1>, <X1>, <Y1>, <>, <>
    <ID2>, <X2>, <Y2>, <>, <>
    ...

A XY files looks like the example below. As it does not contain headers or an index, the first column
is assumed to contain the x-coordinates, the second column the y-coordinates and the
index is a simple enumeration starting at 1. Any additional column is saved as column
of the GeoDataFrame attribute table.

.. code-block:: console

    <X1>, <Y1>, <>, <>
    <X2>, <Y2>, <>, <>
    ...

As the CRS of the coordinates cannot be inferred from the data it must be set in the
data entry in the yaml file as shown in the example below.

.. literalinclude:: ../../assets/data_types/csv_geodataframe.yml
   :language: yaml

.. testsetup:: *

  from hydromt import DataCatalog

.. testcode:: geotiff
  :hide:

  catalog_path = "docs/assets/data_types/csv_geodataframe.yml"

  catalog = DataCatalog(fallback_lib=None)  # do not read default catalog
  catalog.from_yml(catalog_path)

.. _binary_vector:

HydroMT also supports reading and writing vector data in binary format. Currently only
parquet is supported, but others could be added if desired. The structure of the files
should be the same as the text format files described above but writing according to the
parquet file spec. Since this is a binary format, not examples are provided, but for
example pandas can write the same data structure to parquet as it can csv.


.. _GeoDataset:

Geospatial point time-series (GeoDataset)
-----------------------------------------

- :ref:`Netcdf point time-series dataset <NC_point>`
- :ref:`CSV point time-series data <CSV_point>`

.. _geo_formats:

.. list-table::
   :widths: 17, 25, 30
   :header-rows: 1

   * - Driver
     - File formats
     - Comments
   * - :py:class:`geodataset_vector <geodataset.vector_driver.GeoDatasetVectorDriver>`
     - Combined point location (e.g. CSV or GeoJSON) and text delimited time-series (e.g. CSV) data.
     - Uses :py:meth:`~hydromt._io._open_vector`, :py:meth:`~hydromt._io._open_timeseries_from_table`
   * - :py:class:`geodataset_xarray <geodataset.xarray_driver.GeoDatasetXarrayDriver>`
     - NetCDF and Zarr
     - required time and index dimensions_ and x- and y coordinates.


.. _NC_point:

Netcdf point time-series dataset
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Netcdf and Zarr point time-series data are parsed to **GeoDataset** using the **netcdf** and **zarr** drivers.
A typical netcdf or zarr point time-series dataset has the following structure with
two ("time" and "index") dimensions, where the index dimension has x and y coordinates.
The time dimension and spatial coordinates are inferred from the data based
on a list of recognized dimensions_ names.

.. code-block:: console

    Dimensions:      (stations: N, time: NT)
    Coordinates:
      * time         (time)
      * stations     (stations)
        lon          (stations)
        lat          (stations)
    Data variables:
        waterlevel   (time, stations)

To read a point time-series dataset from a multiple file netcdf archive the following data entry
is used, where the options are passed to :py:func:`xarray.open_mfdataset`
(or :py:func:`xarray.open_zarr` for zarr data).
In case the CRS cannot be inferred from the netcdf data it is defined here.
The path to multiple files can be set using a sting glob or several keys,
see description of the `uri` argument in the :ref:`yaml file description <data_yaml>`.
In this example additional renaming and unit conversion preprocessing steps are added to
unify the data to match the HydroMT naming and unit :ref:`terminology <terminology>`.

.. literalinclude:: ../../assets/data_types/netcdf_geodataset.yml
   :language: yaml

.. testsetup:: *

  from hydromt import DataCatalog

.. testcode:: geotiff
  :hide:

  catalog_path = "docs/assets/data_types/netcdf_geodataset.yml"

  catalog = DataCatalog(fallback_lib=None)  # do not read default catalog
  catalog.from_yml(catalog_path)

.. _CSV_point:

CSV point time-series data
^^^^^^^^^^^^^^^^^^^^^^^^^^

Point time-series data where the geospatial point geometries and time-series are saved
in separate (text) files are parsed to **GeoDataset** using the **vector** driver. The
GeoDataset must at least contain a location index with point geometries which is
referred to by the `uri` argument The path may refer to both GIS vector data such as
GeoJSON with only Point geometries or tabulated point vector data such as csv files, see
earlier examples for GeoDataFrame datasets. Finally, certain binary formats such as
parquet are also supported. In addition a tabulated time-series text file can be passed
to be used as a variable of the GeoDataset. This data is added by a second file which is
referred to using the `data_path` option. The index of the time-series (in the columns
header) and point locations must match. For more options see the
:py:meth:`~hydromt._io._open_geodataset` method.

.. literalinclude:: ../../assets/data_types/csv_geodataset.yml
   :language: yaml

.. testsetup:: *

  from hydromt import DataCatalog

.. testcode:: geotiff
  :hide:

  catalog_path = "docs/assets/data_types/csv_geodataset.yml"

  catalog = DataCatalog(fallback_lib=None)  # do not read default catalog
  catalog.from_yml(catalog_path)

*Tabulated time series text file*

This data is read using the :py:meth:`~hydromt._io._open_timeseries_from_table` method.
To read the time stamps the :py:func:`pandas.to_datetime` method is used.

.. code-block:: console

    time, <ID1>, <ID2>
    <time1>, <value>, <value>
    <time2>, <value>, <value>
    ...


.. _Dataset:

NetCDF time-series dataset (Dataset)
------------------------------------

.. _dataset_formats:

.. list-table::
   :widths: 17, 25, 30
   :header-rows: 1

   * - Driver
     - File formats
     - Comments
   * - :py:Class:`dataset_xarray <dataset.xarray_driver.DatasetXarrayDriver>`
     - NetCDF and Zarr
     - required time and index dimensions_.

.. _NC_timeseries:

Netcdf time-series dataset
^^^^^^^^^^^^^^^^^^^^^^^^^^

NetCDF and zarr timeseries data are parsed to **Dataset** with the
:py:class:`~dataset.xarray_driver.DatasetXarrayDriver`.
The resulting dataset is similar to the **GeoDataset** except that it lacks a spatial
dimension.

.. literalinclude:: ../../assets/data_types/netcdf_dataset.yml
  :language: yaml

.. testsetup:: *

  from hydromt import DataCatalog

.. testcode:: geotiff
  :hide:

  catalog_path = "docs/assets/data_types/netcdf_dataset.yml"

  catalog = DataCatalog(fallback_lib=None)  # do not read default catalog
  catalog.from_yml(catalog_path)

.. _DataFrame:

2D tabular data (DataFrame)
---------------------------

.. _dataframe_formats:

.. list-table::
   :widths: 17, 25, 30
   :header-rows: 1

   * - Driver
     - File formats
     - Comments
   * - :py:class:`csv <dataframe.pandas_driver.PandasDriver>`
     - any file readable by pandas
     - Provide a sheet name or formatting through options

.. note::

    Only 2-dimensional data tables are supported, please contact us through the issue list if you would like to have support for n-dimensional tables.


Supported files
^^^^^^^^^^^^^^^

The DataFrameAdapter is quite flexible in supporting different types of tabular data
formats. The driver allows for flexible reading of files: for example both mapping
tables and time series data are supported. Please note that for timeseries, the
`options` need to be used to set the correct column for indexing, and formatting and
parsing of datetime-strings. See the relevant pandas function for which arguments can be
used. Also note that the driver is not restricted to comma-separated files, as
the delimiter can be given to the reader through the `options`.

.. literalinclude:: ../../assets/data_types/csv_dataframe.yml
  :language: yaml

.. testsetup:: *

  from hydromt import DataCatalog

.. testcode:: geotiff
  :hide:

  catalog_path = "docs/assets/data_types/csv_dataframe.yml"

  catalog = DataCatalog(fallback_lib=None)  # do not read default catalog
  catalog.from_yml(catalog_path)

.. note::
    The yml-parser does not correctly parses `None` arguments. When this is required, the `null` argument should be used instead.
    This is parsed to the Python code as a `None`.
