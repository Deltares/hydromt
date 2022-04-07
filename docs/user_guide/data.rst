.. currentmodule:: hydromt

.. _data:

Data input
==========

HydroMT makes use of various types of data sources such as vector data, raster (timeseries) data, 
point location timeseries and tabulated data. All but the tabulate data can be accessed
through the so-called data-catalog which is build from **.yml** files. The yml file
contains the path, reading and pre-processing arguments as well as meta data for each
dataset. The goal of this data catalog is to provide simple and standardized access to (slices of) 
many datasets parsed in convienent Python data objects. Pre-processing steps to unify 
the datasets include renaming of variables, unit conversion and setting/adding nodata 
values. 

The documentation contains a list of (global) datasets_  which can be downloaded to be 
used with various hydroMT models and workflows. The full datasets are available within 
the Deltares network and a slice of these datasets will be downloaded for demonstration 
purposes if no other yml file is provided. Local or other datasets can also be included 
by extending the data catalog with new .yml files. 

Providing a data catalog for the CLI ``hydromt build`` and ``hydromt update`` methods is done with 
``-d /path/to/data_catalog.yml``. Entries from the data_catalog can then be used 
in the *options.ini*. Multiple yml files can be added by reusing the ``-d`` option.
To read data from the deltares network use the ``--dd`` flag (no path required).

.. code-block:: console

    hydromt build MODEL REGION -i options.ini -d /path/to/data_catalog.yml

Basic usage to read a raster dataset

.. code-block:: python

    import hydromt
    data_cat = hydromt.DataCatalog(data_libs=r'/path/to/data-catalog.yml')
    ds = data_cat.get_rasterdataset('merit_hydro', bbox=[xmin, ymin, xmax, ymax])  # returns xarray.dataset

.. _data_yaml:

Data catalog yaml file
----------------------

Each dataset, is added in the yaml file with a user-defined name. This name is used in 
the ini file (CLI) or :py:class:`~hydromt.data_adapter.DataCatalog` *get_data*  methods (Python), see basic usage above. 
A full dataset entry for a dataset called **my_dataset** is given in the example below. 
The ``path``, ``data_type`` and ``driver`` options are required and the ``meta`` option 
with the shown keys is highly recommended. The ``rename``, ``nodata``, ``unit_add`` and 
``unit_mult`` options are set per variable (or attribute table column in case of a GeoDataFrame).
``kwargs`` is an option to pass additional options to each open data method of the different drivers.
For more information see :py:meth:`~hydromt.data_adapter.DataCatalog.from_yml`

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
        paper_ref: Author et al. (2020)
        paper_doi: doi
        source_license: CC-BY-3.0
        category: <category>


A full list of **data entry options** is given below

- **path** (required): path to the data file. 
  Relative paths are combined with the global ``root`` option of the yaml file (if available) or the directory of the yaml file itself. 
  To read multiple files in a single dataset (if supported by the driver) a string glob in the form of ``"path/to/my/files/*.nc"`` can be used.
  The filenames can be futher specified with ``{variable}``, ``{year}`` and ``{month}`` keys to limit which files are being read based on the get_data request in the form of ``"path/to/my/files/{variable}_{year}_{month:02d}.nc"``
- **data_type** (required): type of input data. Either *RasterDataset*, *GeoDataset* or *GeoDataFrame*.
- **driver** (required): data_type specific driver to read a dataset, see overview below.
- **crs** (required if missing in the data): EPSG code or WKT string of the reference coordinate system of the data. Only used if not crs can be infered from the input data.
- **kwargs** (optional): pairs of key value arguments to pass to the driver specific open data method (eg xr.open_mfdataset for netdcf raster, see the full list below).
- **rename** (optional): pairs of variable names in the input data (*old_variable_name*) and the corresponding generic HydroMT name for renaming (*new_variable_name*). 
- **nodata** (optional): nodata value of the input data. For Raster- and GeoDatasets this is only used if not inferred from the original input data, For GeoDataFrame provided nodata values are converted to nan values.
- **unit_add** (optional): add or substract a value to the input data for unit conversion (e.g. -273.15 for conversion of temperature from Kelvin to Celsius). 
- **unit_mult** (optional): multiply the input data by a value for unit conversion (e.g. 1000 for conversion from m to mm of precipitation).
- **meta** (optional): additional information on the dataset organised in a sub-list, for example version or data source url etc. These are added to the data attributes.
- **units** (optional and for *RasterDataset* only). specify the units of the input data: supported are [m3], [m], [mm], and [m3/s].
  This option is used *only* for the forcing of the Delwaq models in order to do specific unit conversions that cannot be handled from simple 
  addition or multiplication (e.g. conversion from mm water equivalent to m3/s of water which requires a multiplication by each grid cell area and not a fixed number).
  
Apart from the data entries, the yaml file also has two **global options**:

- **root** (optional): root folder for all the data sources in the yaml file. 
  If not  provide the folder of where the yaml fil is located will be used as root.
  This is used in combination with each data source **path** argument to avoid repetition.
- **category** (optional): type of datasets listed in the yaml file. Will be added to each meta attributes of the data sources listed. Usual categories 
  within HydroMT are *topography*, *meteo*, *soil*, *landuse & landcover*, *surface water*, *ocean*, *socio economic*, *observed data* 
  but the user is free to define its own categories. The category attribute can also be added to each source meta attributes.

Supported data types and associated drivers
-------------------------------------------

HydroMT currently supports the following data types:

- **RasterDataset**: static and dynamic raster data 
- **GeoDataFrame**: static vector data 
- **GeoDataset** dynamic point location data

Internally the RasterDataset and GeoDataset are represented by :py:class:`xarray.Dataset` objects 
and GeoDataFrame by :py:class:`geopandas.GeoDataFrame`. We use externaly 
availabe data readers, often wrapped in hydroMT functions, to parse many different file
formats to this standardized internal data representation. An overview of the supported 
data formats and associated drivers and python methods are shown below followed by 
some examples.

Tabulated data without a spatial component such as mapping tables are planned to be added. 
Please contact us through the issue list if you would like to add other drivers.

.. list-table:: RasterDataset
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

.. list-table:: GeoDataFrame
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

.. list-table:: GeoDataset
   :widths: 17, 25, 28, 30
   :header-rows: 1

   * - Driver
     - File formats
     - Method
     - Comments
   * - ``vector`` 
     - Combined point location (e.g. CSV or GeoJSON) and text delimited timeseries (e.g. CSV) data.
     - :py:meth:`~hydromt.io.open_geodataset`
     - Uses :py:meth:`~hydromt.io.open_vector`, :py:meth:`~hydromt.io.open_timeseries_from_table`
   * - ``netcdf`` or ``zarr``
     - NetCDF and Zarr
     - :py:func:`xarray.open_mfdataset`, :py:func:`xarray.open_zarr`
     - required time and index dimensions_ and x- and y coordinates.


.. _dimensions: 

recognized dimension and coordinate names:

- time: time or date stamp ["time"].
- x: x coordinate ["x", "longitude", "lon", "long"]. 
- y: y-coordinate ["y", "latitude", "lat"].


Single variable GeoTiff raster
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

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
        source_url: http://due.esrin.esa.int/page_globcover.php
        paper_ref: Arino et al (2012)
        paper_doi: 10.1594/PANGAEA.787668
        source_license: CC-BY-3.0



Multi variable Virtual Raster Tileset (VRT)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Multiple raster layers from different files are parsed to a **RasterDataset** using the **raster** driver.
Each raster becomes a variable in the resulting RasterDataset based on its filename.
The path to multiple files can be set using a sting glob or several keys, 
see description of the ``path`` argument in the :ref:`yaml file description <data_yaml>`.
Note that the rasters should have identical grids. 

Here multiple .vrt files (dir.vrt, bas.vrt, etc.) are combined based on their variable name 
into a single dataset with variables flwdir, basins, etc.
Other multiple file raster datasets (e.g. GeoTIFF files) can be read in the same way.
VRT files are usefull for large raster datasets which are often tiled and can be combined using
gdalbuildvrt (see https://gdal.org/programs/gdalbuildvrt.html).


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
        upg: upgrid
        elv: elevtn
        sto: strord
        slp: lndslp
        wth: rivwth
      meta:
        category: topography
        source_version: 1.0
        paper_doi: 10.1029/2019WR024873
        paper_ref: Dai Yamazaki
        source_url: http://hydro.iis.u-tokyo.ac.jp/~yamadai/MERIT_Hydro
        source_license: CC-BY-NC 4.0 or ODbL 1.0


Tiled raster dataset
^^^^^^^^^^^^^^^^^^^^

Tiled index datasets are parsed to a **RasterDataset** using the **raster_tindex** driver.
This data format is used to combine raster tiles with different CRS projections. 
A polygon vector file (e.g. GeoPackage) is used to make a tile index with the spatial 
footprints of each tile. When reading a spatial slice of this data the files with 
intersecting footprints will be mosaiced together in the CRS of the most central tile. 
Use gdaltindex to build an excepted tile index file (see https://gdal.org/programs/gdaltindex.html)

Here a GeoPackage with the tile index refering to individual GeoTiff raster tiles is used. 
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
        category: surface water
        paper_doi: 10.1126/science.aat0636
        paper_ref: Allen and Pavelsky (2018)
        source_license: CC BY 4.0
        source_url: https://doi.org/10.5281/zenodo.1297434
        source_version: 1.01


Netcdf raster dataset
^^^^^^^^^^^^^^^^^^^^^

Netcdf and Zarr raster data are parsed to **RasterDataset** using the **netcdf** and **zarr** drivers.
A typical raster netcdf or zarr raster dataset has the following structure with 
two ("y" and "x") or three ("dim0", "y" and "x") dimensions. 
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
In case the CRS cannot be infered from the netcdf data it is defined here. 
The path to multiple files can be set using a sting glob or several keys, 
see description of the ``path`` argument in the :ref:`yaml file description <data_yaml>`.
In this example additional renaming and unit conversion preprocessing steps are added to 
unify the data to match the hydroMT naming and unit :ref:`data convention <data_convention>`. 

.. code-block:: yaml

    era5_hourly:
      path: forcing/ERA5/org/era5_{variable}_{year}_hourly.nc
      data_type: RasterDataset
      driver: netcdf
      crs: 4326
      kwargs:
        chunks: {latitude: 125, longitude: 120, time: 50}
        combine: by_coords
        concat_dim: time
        decode_times: true
        parallel: true
      meta:
        category: meteo
        history: Extracted from Copernicus Climate Data Store
        paper_doi: 10.1002/qj.3803
        paper_ref: Hersbach et al. (2019)
        source_license: https://cds.climate.copernicus.eu/cdsapp/#!/terms/licence-to-use-copernicus-products
        source_url: https://doi.org/10.24381/cds.bd0915c6
        source_version: ERA5 hourly data on pressure levels
      rename:
        t2m: temp
        tp: precip
      unit_add:
        temp: -273.15
      unit_mult:
        precip: 1000

In :py:func:`xarray.open_mfdataset`, xarray allows for a *preprocess* function to be run before merging several 
netcdf files together. In hydroMT, some preprocess functions are availabel and can be passed through the ``kwargs`` 
options in the same way as any xr.open_mfdataset options. These preprocess functions are:

- **round_latlon**: round x and y dimensions to 5 decimals to avoid merging problems in xarray due to small differences
  in x, y values in the different netcdf files of the same data source.
- **to_datetimeindex**: transpose time dimension of gridded timeseries to datetime index.
- **remove_duplicates**: remove time duplicates when opening xarray mfdataset.

GeoPackage spatial vector data
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Sptial vector data is parsed to a **GeoDataFrame** using the **vector** driver.
For large spatial vector datasets we recommend the GeoPackage format as it includes a 
spatial index for fast filtering of the data based on spatial location. An example is 
shown below. Not that the rename, unit_mult, unit_add and nodata options refer to
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
          source_version: 1.0
          source_author: Wilfred Altena
          source_info: data combined from World Bank and CIA World Factbook


Point vector from text delimited data
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Tabulated point vector data files can be parsed to a **GeoDataFrame** with the **vector_table** 
driver. This driver reads CSV (or similar delimited text files), EXCEL and XY 
(white-space delimited text file without headers) files. See this list of dimensions_ 
name for recognized x and y column names.  
  
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

As the CRS of the coordinates cannot be infered from the data it must be set in the 
data entry in the yaml file as shown in the example below. The internal data format 
is based on the file extension unless the ``kwargs`` ``driver`` option is set.
See py:meth:`~hydromt.io.open_vector` and py:meth:`~hydromt.io.open_vector_from_table` for more
options.

.. code-block:: yaml

    stations:
      path: /path/to/stations.csv
      data_type: GeoDataFrame
      driver: vector_table
      crs: 4326
      kwargs:
        driver: csv



Netcdf point timeseries dataset
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Netcdf and Zarr point timeseries data are parsed to **GeoDataset** using the **netcdf** and **zarr** drivers.
A typical netcdf or zarr point timeseries dataset has the following structure with 
two ("time" and "index") dimensions, where the index dimension has x and y coordinates. 
The time dimension and spatial coordinates are infered from the data based 
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

To read a point timeseries dataset from a multiple file netcdf archive the following data entry
is used, where the ``kwargs`` are passed to :py:func:`xarray.open_mfdataset` 
(or :py:func:`xarray.open_zarr` for zarr data). 
In case the CRS cannot be infered from the netcdf data it is defined here. 
The path to multiple files can be set using a sting glob or several keys, 
see description of the ``path`` argument in the :ref:`yaml file description <data_yaml>`.
In this example additional renaming and unit conversion preprocessing steps are added to 
unify the data to match the hydroMT naming and unit :ref:`data convention <data_convention>`. 

.. code-block:: yaml

    gtsmv3_eu_era5:
      path: reanalysis-waterlevel-{year}-m{month:02d}.nc
      data_type: GeoDataset
      driver: netcdf
      crs: 4326
      kwargs:
        chunks: {stations: 100, time: 1500}
        combine: by_coords
        concat_dim: time
        decode_times: true
        parallel: true
      meta:
        paper_doi: 10.24381/cds.8c59054f
        paper_ref: Copernicus Climate Change Service 2019
        source_license: https://cds.climate.copernicus.eu/cdsapp/#!/terms/licence-to-use-copernicus-products
        source_url: https://cds.climate.copernicus.eu/cdsapp#!/dataset/10.24381/cds.8c59054f?tab=overview


CSV point timeseries data
^^^^^^^^^^^^^^^^^^^^^^^^^

Point timeseries data where the geospatial point geometries and timeseries are saved in
seperate (text) files are parsed to **GeoDataset** using the **vector** driver. 
The GeoDataset must at least contain a location index with point geometries which is refered to by the ``path`` argument
The path may refer to both GIS vector data such as GeoJSON with only Point geometries 
or tabulated point vector data such as csv files, see earlier examples for GeoDataFrame datasets. 
In addition a tabulated timeseries text file can be passed to be used as a variable of the GeoDataset. 
This data is added by a second file which is refered to using the ``fn_data`` key-word argument. 
The index of the timeseries (in the columns header) and point locations must match. 
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


.. _data_convention:

Conventions on variable names and units
---------------------------------------

This section lists the different variable naming and unit conventions of HydroMT by types. This list is still in development. 
Names and units mentioned here are mandatory in order for the input data to be processed correctly and produced the right derived data. 
It is also possible to use the rename option so that variables and model data produced by HydroMT have more explicit names.
A list of recognized dimensions_ is found here.

Topography
^^^^^^^^^^
   
- elevtn: altitude [m].
- mdt: mean dynamic topography [m].
- flwdir: flow direction. Format supported are ArcGIS D8, LDD, NEXTXY. The format is infered from the data.
- uparea: upstream area [km2].
- lndslp: slope [m/m].
- strord: Stralher streamorder [-].
- basins: basins ID mapping [-].


Surface waters
^^^^^^^^^^^^^^
Rivers:

- rivlen: river length [m].
- rivslp: river slope [m/m].
- rivwth: river width [m].
- rivmsk: mask of river cells (for raster models) [bool].

Reservoirs / Lakes:

- waterbody_id: reservoir/lake ID [-].
- Hylak_id: ID from the HydroLAKES database (to connect to the hydroengine library) [-].
- Area_avg: average waterbody area [m2].
- Vol_avg: average waterbody volume [m3].
- Depth_avg: average waterbody depth [m].
- Dis_avg: average waterbody discharge [m3/s].
- xout: longitude of the waterbody outlet [-].
- yout: latitude of the waterbody outlet [-].
- Capacity_max: maximum reservoir capacity volume [m3].
- Capacity_norm: normal/average reservoir capacity volume [m3].
- Capacity_min: minimum reservoir capcity volume [m3].
- Dam_height: height of the dam [m].

Glaciers:

- simple_id: glacier ID in the current database [-].

Landuse and landcover
^^^^^^^^^^^^^^^^^^^^^

- landuse: landuse classification [-].
- LAI: Leaf Area Index [-].

Soil
^^^^

- bd_sl*: bulk density of the different soil layers (1 to 7 in soilgridsv2017) [g cm-3].
- clyppt_sl*: clay content of the different soil layers (1 to 7 in soilgridsv2017) [%].
- oc_sl*: organic carbon contant of the different soil layers (1 to 7 in soilgridsv2017) [%].
- ph_sl*: pH of the different soil layers (1 to 7 in soilgridsv2017) [-].
- sltppt_sl*: silt content of the different soil layers (1 to 7 in soilgridsv2017) [%].
- sndppt_sl*: sand content of the different soil layers (1 to 7 in soilgridsv2017) [%].
- soilthickness: soil thickness [cm].
- tax_usda: USDA soil classification [-].


Meteorology
^^^^^^^^^^^

- precip: precipitation (rainfall+snowfall) [mm].
- temp: average temperature [oC].
- temp_min: minimum temperature [oC].
- temp_max: maximum temperature [oC].
- press_msl: atmospheric pressure [hPa].
- kin: shortwave incoming radiation [W m-2].
- kout: TOA incident solar radiation [W m-2].

Hydrology
^^^^^^^^^

- run: surface water runoff (overland flow + river discharge) [m3/s].
- vol: water volumes [m3].
- infilt: water infiltration in the soil [m3/s].
- runPav: excess infiltration runoff on paved areas [m3/s].
- runUnp: excess infiltration runoff on unpaved areas [m3/s].
- inwater: sum of all fluxes entering/leaving the surface waters (precipitation, evaporation, infiltration...) [m3/s].
- inwaterInternal: sum of all fluxes between the land and river surface waters (part of inwater) [m3/s].

.. _datasets:

Suggested global datasets
-------------------------
Below is the list of suggested data sources for use with HydroMT. The overview contains
links to the source of and available literature behind each dataset. The complete 
datasets are available within the Deltares network and a slice of data is available 
for demonstration purposes.  

.. csv-table:: Data Catalog
   :file: ../_generated/data_sources.csv
   :header-rows: 1
   :widths: auto
   :width: 50
