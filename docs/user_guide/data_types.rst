.. _data_types: 

Supported data types
--------------------

HydroMT currently supports the following data types:

- :ref:`RasterDataset <RasterDataset>`: static and dynamic raster (or gridded) data 
- :ref:`GeoDataFrame <GeoDataFrame>`: static vector data 
- :ref:`GeoDataset <GeoDataset>`: dynamic point location data
- :ref:`DataFrame <DataFrame>`: 2D tabular data

Internally the RasterDataset and GeoDataset are represented by :py:class:`xarray.Dataset` objects,
the GeoDataFrame by :py:class:`geopandas.GeoDataFrame`, and the DataFrame by 
:py:class:`pandas.DataFrame`. We use drivers, typically from third-party packages and sometimes 
wrapped in HydroMT functions, to parse many different file formats to this standardized internal 
data representation. 

.. note::

    Please contact us through the issue list if you would like to add other drivers.

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
   :widths: 17, 25, 28, 30
   :header-rows: 1

   * - Driver
     - File formats
     - Method
     - Comments
   * - ``raster`` 
     - GeoTIFF, ArcASCII, VRT, etc. (see `GDAL formats <http://www.gdal.org/formats_list.html>`_)
     - :py:meth:`~hydromt.io.open_mfraster`
     - Based on :py:func:`xarray.open_rasterio` 
       and :py:func:`rasterio.open`
   * - ``raster_tindex`` 
     - raster tile index file (see `gdaltindex <https://gdal.org/programs/gdaltindex.html>`_)
     - :py:meth:`~hydromt.io.open_raster_from_tindex`
     - Options to merge tiles via ``mosaic_kwargs``.
   * - ``netcdf`` or ``zarr``
     - NetCDF and Zarr
     - :py:func:`xarray.open_mfdataset`, :py:func:`xarray.open_zarr`
     - required y and x dimensions_


.. _GeoTiff: 

**Single variable GeoTiff raster**

Single raster files are parsed to a **RasterDataset** based on the **raster** driver.
This driver supports 2D raster for which the dimensions are names "x" and "y". 
A potential third dimension is called "dim0". 
The variable name is based on the filename, in this case "GLOBCOVER_200901_200912_300x300m". 
The ``chunks`` key-word argument is passed to :py:meth:`~hydromt.io.open_mfraster` 
and allows lazy reading of the data. 

.. code-block:: yaml

    globcover:
      path: base/landcover/globcover/GLOBCOVER_200901_200912_300x300m.tif
      data_type: RasterDataset
      driver: raster
      kwargs:
        chunks: {x: 3600, y: 3600}
      meta:
        category: landuse
        source_url: http://due.esrin.esa.int/page_globcover.php
        source_license: CC-BY-3.0
        paper_ref: Arino et al (2012)
        paper_doi: 10.1594/PANGAEA.787668

.. _VRT: 

Multi-variable Virtual Raster Tileset (VRT)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Multiple raster layers from different files are parsed using the **raster** driver.
Each raster becomes a variable in the resulting RasterDataset based on its filename.
The path to multiple files can be set using a sting glob or several keys, 
see description of the ``path`` argument in the :ref:`yaml file description <data_yaml>`.
Note that the rasters should have identical grids. 

Here multiple .vrt files (dir.vrt, bas.vrt, etc.) are combined based on their variable name 
into a single dataset with variables flwdir, basins, etc.
Other multiple file raster datasets (e.g. GeoTIFF files) can be read in the same way.
VRT files are useful for large raster datasets which are often tiled and can be combined using
`gdalbuildvrt. <https://gdal.org/programs/gdalbuildvrt.html>`_


.. code-block:: yaml

    merit_hydro:
      path: base/merit_hydro/{variable}.vrt
      data_type: RasterDataset
      driver: raster
      crs: 4326
      kwargs:
        chunks: {x: 6000, y: 6000}
      rename:
        dir: flwdir
        bas: basins
        upa: uparea
        elv: elevtn
        sto: strord
      meta:
        category: topography
        source_version: 1.0
        paper_doi: 10.1029/2019WR024873
        paper_ref: Dai Yamazaki
        source_url: http://hydro.iis.u-tokyo.ac.jp/~yamadai/MERIT_Hydro
        source_license: CC-BY-NC 4.0 or ODbL 1.0

.. _Tile:

Tiled raster dataset
^^^^^^^^^^^^^^^^^^^^

Tiled index datasets are parsed using the **raster_tindex** driver.
This data format is used to combine raster tiles with different CRS projections. 
A polygon vector file (e.g. GeoPackage) is used to make a tile index with the spatial 
footprints of each tile. When reading a spatial slice of this data the files with 
intersecting footprints will be merged together in the CRS of the most central tile. 
Use `gdaltindex <https://gdal.org/programs/gdaltindex.html>`_ to build an excepted tile index file.

Here a GeoPackage with the tile index referring to individual GeoTiff raster tiles is used. 
The ``mosaic_kwargs`` are passed to :py:meth:`~hydromt.io.open_raster_from_tindex` to 
set the resampling ``method``. The name of the column in the tile index attribute table ``tileindex``
which contains the raster tile file names is set in the ``kwargs`` (to be directly passed as an argument to 
:py:meth:`~hydromt.io.open_raster_from_tindex`).

.. code-block:: yaml

    grwl_mask:
      path: static_data/base/grwl/tindex.gpkg
      data_type: RasterDataset
      driver: raster_tindex
      nodata: 0
      kwargs:
        chunks: {x: 3000, y: 3000}
        mosaic_kwargs: {method: nearest}
        tileindex: location
      meta:
        category: hydrography
        paper_doi: 10.1126/science.aat0636
        paper_ref: Allen and Pavelsky (2018)
        source_license: CC BY 4.0
        source_url: https://doi.org/10.5281/zenodo.1297434
        source_version: 1.01

.. NOTE::

  Tiled raster datasets are not read lazily as different tiles have to be merged together based on 
  their values. For fast access to large raster datasets, other formats might be more suitable.

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
is used, where the ``kwargs`` are passed to :py:func:`xarray.open_mfdataset` 
(or :py:func:`xarray.open_zarr` for zarr data). 
In case the CRS cannot be inferred from the netcdf data it should be defined with the ``crs`` option here. 
The path to multiple files can be set using a sting glob or several keys, 
see description of the ``path`` argument in the :ref:`yaml file description <data_yaml>`.
In this example additional renaming and unit conversion preprocessing steps are added to 
unify the data to match the HydroMT naming and unit :ref:`terminology <terminology>`. 

.. code-block:: yaml

    era5_hourly:
      path: forcing/ERA5/org/era5_{variable}_{year}_hourly.nc
      data_type: RasterDataset
      driver: netcdf
      crs: 4326
      kwargs:
        chunks: {latitude: 125, longitude: 120, time: 50}
        combine: by_coords
        decode_times: true
        parallel: true
      meta:
        category: meteo
        paper_doi: 10.1002/qj.3803
        paper_ref: Hersbach et al. (2019)
        source_license: https://cds.climate.copernicus.eu/cdsapp/#!/terms/licence-to-use-copernicus-products
        source_url: https://doi.org/10.24381/cds.bd0915c6
      rename:
        t2m: temp
        tp: precip
      unit_add:
        temp: -273.15
      unit_mult:
        precip: 1000


Preprocess functions when combining multiple files
""""""""""""""""""""""""""""""""""""""""""""""""""

In :py:func:`xarray.open_mfdataset`, xarray allows for a *preprocess* function to be run before merging several 
netcdf files together. In hydroMT, some preprocess functions are available and can be passed through the ``kwargs`` 
options in the same way as any xr.open_mfdataset options. These preprocess functions are:

- **round_latlon**: round x and y dimensions to 5 decimals to avoid merging problems in xarray due to small differences
  in x, y values in the different netcdf files of the same data source.
- **to_datetimeindex**: force parsing the time dimension to a datetime index.
- **remove_duplicates**: remove time duplicates



.. _GeoDataFrame: 

Vector data (GeoDataFrame)
--------------------------

- :ref:`GeoPackage spatial vector data <GPKG_vector>`
- :ref:`Point vector from text delimited data <textdelimited_vector>`

.. _vector_formats:

.. list-table::
   :widths: 17, 25, 28, 30
   :header-rows: 1

   * - Driver
     - File formats
     - Method
     - Comments
   * - ``vector`` 
     - ESRI Shapefile, GeoPackage, GeoJSON, etc.
     - :py:meth:`~hydromt.io.open_vector` 
     - Point, Line and Polygon geometries. Uses :py:func:`geopandas.read_file`
   * - ``vector_table``
     - CSV, XY, and EXCEL. 
     - :py:meth:`~hydromt.io.open_vector`
     - Point geometries only. Uses :py:meth:`~hydromt.io.open_vector_from_table`



.. _GPKG_vector:

GeoPackage spatial vector data
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Spatial vector data is parsed to a **GeoDataFrame** using the **vector** driver.
For large spatial vector datasets we recommend the GeoPackage format as it includes a 
spatial index for fast filtering of the data based on spatial location. An example is 
shown below. Note that the rename, ``unit_mult``, ``unit_add`` and ``nodata`` options refer to
columns of the attribute table in case of a GeoDataFrame.

.. code-block:: yaml

      GDP_world:
        path: base/emissions/GDP-countries/World_countries_GDPpcPPP.gpkg
        data_type: GeoDataFrame
        driver: vector
        kwargs:
          layer: GDP
        rename:
          GDP: gdp
        unit_mult:
          gdp: 0.001
        meta:
          category: socio-economic
          source_version: 1.0

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
data entry in the yaml file as shown in the example below. The internal data format 
is based on the file extension unless the ``kwargs`` ``driver`` option is set.
See :py:meth:`~hydromt.io.open_vector` and :py:func:`~hydromt.io.open_vector_from_table` for more
options.

.. code-block:: yaml

    stations:
      path: /path/to/stations.csv
      data_type: GeoDataFrame
      driver: vector_table
      crs: 4326
      kwargs:
        driver: csv

.. _GeoDataset: 

Geospatial point time-series (GeoDataset)
-----------------------------------------

- :ref:`Netcdf point time-series dataset <NC_point>`
- :ref:`CSV point time-series data <CSV_point>`

.. _geo_formats:

.. list-table::
   :widths: 17, 25, 28, 30
   :header-rows: 1

   * - Driver
     - File formats
     - Method
     - Comments
   * - ``vector`` 
     - Combined point location (e.g. CSV or GeoJSON) and text delimited time-series (e.g. CSV) data.
     - :py:meth:`~hydromt.io.open_geodataset`
     - Uses :py:meth:`~hydromt.io.open_vector`, :py:meth:`~hydromt.io.open_timeseries_from_table`
   * - ``netcdf`` or ``zarr``
     - NetCDF and Zarr
     - :py:func:`xarray.open_mfdataset`, :py:func:`xarray.open_zarr`
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
is used, where the ``kwargs`` are passed to :py:func:`xarray.open_mfdataset` 
(or :py:func:`xarray.open_zarr` for zarr data). 
In case the CRS cannot be inferred from the netcdf data it is defined here. 
The path to multiple files can be set using a sting glob or several keys, 
see description of the ``path`` argument in the :ref:`yaml file description <data_yaml>`.
In this example additional renaming and unit conversion preprocessing steps are added to 
unify the data to match the HydroMT naming and unit :ref:`terminology <terminology>`. 

.. code-block:: yaml

    gtsmv3_eu_era5:
      path: reanalysis-waterlevel-{year}-m{month:02d}.nc
      data_type: GeoDataset
      driver: netcdf
      crs: 4326
      kwargs:
        chunks: {stations: 100, time: 1500}
        combine: by_coords
        decode_times: true
        parallel: true
      rename:
        station_x_coordinate: lon
        station_y_coordinate: lat
        stations: index
      meta:
        category: ocean
        paper_doi: 10.24381/cds.8c59054f
        paper_ref: Copernicus Climate Change Service 2019
        source_license: https://cds.climate.copernicus.eu/cdsapp/#!/terms/licence-to-use-copernicus-products
        source_url: https://cds.climate.copernicus.eu/cdsapp#!/dataset/10.24381/cds.8c59054f?tab=overview

.. _CSV_point: 

CSV point time-series data
^^^^^^^^^^^^^^^^^^^^^^^^^^

Point time-series data where the geospatial point geometries and time-series are saved in
separate (text) files are parsed to **GeoDataset** using the **vector** driver. 
The GeoDataset must at least contain a location index with point geometries which is referred to by the ``path`` argument
The path may refer to both GIS vector data such as GeoJSON with only Point geometries 
or tabulated point vector data such as csv files, see earlier examples for GeoDataFrame datasets. 
In addition a tabulated time-series text file can be passed to be used as a variable of the GeoDataset. 
This data is added by a second file which is referred to using the ``fn_data`` key-word argument. 
The index of the time-series (in the columns header) and point locations must match. 
For more options see the :py:meth:`~hydromt.io.open_geodataset` method.

.. code-block:: yaml

    waterlevels_txt:
      path: /path/to/stations.csv
      data_type: GeoDataset
      driver: vector
      crs: 4326
      kwargs:
        fn_data: /path/to/stations_data.csv

*Tabulated time series text file*

This data is read using the :py:meth:`~hydromt.io.open_timeseries_from_table` method. To 
read the time stamps the :py:func:`pandas.to_datetime` method is used.

.. code-block:: console

    time, <ID1>, <ID2> 
    <time1>, <value>, <value>
    <time2>, <value>, <value>
    ...

.. _DataFrame: 

2D tabular data (DataFrame)
---------------------------

.. _dataframe_formats:

.. list-table::
   :widths: 17, 25, 28, 30
   :header-rows: 1

   * - Driver
     - File formats
     - Method
     - Comments
   * - ``csv`` 
     - Comma-separated files (or using another delimiter)
     - :py:func:`pandas.read_csv`
     - See :py:func:`pandas.read_csv` for all 
   * - ``excel`` 
     - Excel files
     - :py:func:`pandas.read_excel`
     - If required, provide a sheet name through kwargs
   * - ``fwf`` 
     - Fixed width delimited text files
     - :py:func:`pandas.read_fwf`
     - The formatting of these files can either be inferred or defined by the user, both through the kwargs.


.. note::

    Only 2-dimensional data tables are supported, please contact us through the issue list if you would like to have support for n-dimensional tables.


Supported files
^^^^^^^^^^^^^^^

The DataFrameAdapter is quite flexible in supporting different types of tabular data formats. All drivers allow for flexible reading of 
files: for example both mapping tables and time series data are supported. Please note that for timeseries, the kwargs need to be used to 
set the correct column for indexing, and formatting and parsing of datetime-strings. See the relevant pandas function for which arguments
can be used. Also note that the **csv** driver is not restricted to comma-separated files, as the delimiter can be given to the reader 
throught the kwargs.

.. code-block:: yaml

    observations:
      path: data/lulc/globcover_mapping.csv
      data_type: DataFrame
      driver: csv
      meta:
        category: parameter_mapping
      kwargs:
        header: null
        index_col: 0
        parse_dates: false

.. note::
    The yml-parser does not correctly parses `None` arguments. When this is required, the `null` argment should be used instead.
    This is parsed to the Python code as a `None`.