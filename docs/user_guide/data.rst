.. _data:

Data input
==========

HydroMT can make use of various types of data sources such as vector data, GDAL rasters or NetCDF files. 
The path and attributes of each of these dataset are listed in different *.yml* library files. HydroMT already 
contains a list of default global datasets that can be used as is within the Deltares network (displayed below). 
Local or other datasets can also be included by extending or using another local .yml file. We will see what are the 
steps and conventions to add data to the yaml libaries of HydroMT.

Adding data sources in a HydroMT yaml libary
--------------------------------------------
Organisation of the yaml file
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
The great strength of HydroMT is that it can easily read and use very different types of input data for model preparation, 
such as :

- static and dynamic raster data in any format supported by GDAL (eg geotiff, ArcASCII, NetCDF...)
- static vector data in any format supported by GDAL (eg shapefile, geopackage, geojson, csv..)
- point timeseries (eg NetCDF)

In order to use different input data within HydroMT, they need to be added to a yaml library. The HydroMT yaml libraries 
are organised as a list of datasets. Input data is divided into three types:

- **RasterDataset** for static or dynamic rasters
- **GeoDataFrame** for vectors
- **GeoDataset** for point timeseries

Each new data, is added in a yml library list with an internal name for HydroMT. This name is not 
used directly inside the tool, but is used in the ini file (CLI options) where the user specifies which dataset is used to build or 
update a specific model component (usually the 'source_name' option under the different [sections] of the ini file).


Apart from the data sources list, the yaml library also has two **optional global properties**:

- **root**: path to a folder containing all the data sources in the yaml file. Is used in combination with each data source **path** 
  argument to avoid repetition.
- **category**: type of datasets listed in the yaml file. Will be added to each meta attributes of the data sources listed. Usual categories 
  within HydroMT are *topography*, *meteo*, *soil*, *landuse & landcover*, *surface water*, *ocean*, *socio economic*, *observed data* 
  but the user is free to define its own categories. The category attribute can also be added to each source meta attributes.


In order to be automatically read and processes by HydroMT, the yaml library also contains a set of properties of the dataset. 
The properties can differ depending on the data type. Here is the list of properties that are expected in HydroMT:

- **path** (required): path to the data file (eg D:/Dataset/my_raster.tif). Can be combined with the **root** option of the yaml file.
- **data_type** (required): tytpe of input data. Either *RasterDataset*, *GeoDataFrame* or *GeoDataset*.
- **driver** (required if different than default): type of driver to read the dataset. Either *raster* (default), *netcdf*, *zarr* or *vector*.
- **kwargs** (optional): optional sub-list of arguments that can be passed to the driver when reading data. These are driver specific.
- **crs** (required if different than default): reference coordinate system of the data (by default 4326)
- **nodata** (optional): sets or updates the nodata value of the input data. By default the nodata value in inferred from the original input data 
  or set to zero if not available.
- **rename** (optional): sub-list of varibales names in the input data and the corresponding generic HydroMT name for renaming. Written as 
  'name_in_input_data': 'HydroMT_name'.
- **unit_add** (optional): sub-list allowing the user to add or substract a certain number to a variable in the input data for unit conversion 
  (e.g. -273.15 for conversion of temperature from Kelvin to Celsius). Written as 'HydroMT_name': 'Float_number_to_add'.
- **unit_mult** (optional): sub-list allowing the user to multiply a variable in the input data by a certain number for unit conversion 
  (e.g. 1000 for conversion from m to mm of precipitation). Written as 'HydroMT_name': 'Float_number_to_multiply'.
- **meta** (optional): additional information on the dataset organised in a sub-list, for example version or data source etc.

Here is an example of how the global GlobCover land use classification raster is included in the HydroMT yaml library with the **globcover** 
internal name to be used from the ini file or CLI options:

.. code-block:: console

    root: p:/wflow_global/static_data
    category: landuse & landcover
    globcover:
      path: base/landcover/globcover/GLOBCOVER_200901_200912_300x300m.tif
      data_type: RasterDataset
      driver: raster
      crs: 4326
      meta:
        source_url: http://due.esrin.esa.int/page_globcover.php
        paper_ref: Arino et al (2012)
        paper_doi: 10.1594/PANGAEA.787668
        source_license: CC-BY-3.0

An additional option, available for **RasterDataset** only, is to specify the units 
of the varibales inside of the input data. This option is now used **only for the forcing of the Delwaq models** in order 
to do specific unit conversions that cannot be handled from simple addition or multiplication (eg conversion from mm water equivalent 
to m3/s of water which requires a multiplication by each grid cell area and not a fixed number).

- **units**: sub-list allowing the user to specify the unit of a variable in the forcing input data of Delwaq for conversion of water fluxes from 
  mm to m3/s where necessary. Written as 'HydroMT_name': 'unit' (e.g. precip: mm).

Data names and units
^^^^^^^^^^^^^^^^^^^^
The example above works when the input data is very simple, for example a single raster containing a classification or 
a variable already at the right unit. In order to handle many different data sources for the same purpose, HydroMT requires 
that the input data follows certain naming conventions and units. If the input data is not directly containing the right variables names 
and units, the yaml library can be extended to contain the necessary information to rename the variables and convert ther units. 
Certain data providers sometimes add a scale factor to store more efficiently their data (as int8 instead of float32, used for example with 
MODIS LAI). This scale factor can also be taken into account when reading the data as a unit conversion by HydroMT. 
Required conventions for HydroMT names and units are detailed in the :ref:`data conventions <data_convention>` section.

The options of the yaml libraries handling renaming and unit conversions are: **rename**, **unit_add** and **unit_mult**.

Below are two examples where these options are used for a raster and a vector file.

.. code-block:: console

      hydro_merit:
        path: base/hydro_merit/*.vrt
        data_type: RasterDataset
        chunks: {x: 6000, y: 6000}
        crs: 4326
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
      hydro_reservoirs:
        path: base/waterbodies/reservoir-db.gpkg
        data_type: GeoDataFrame
        crs: 4326
        nodata: [-99]
        rename:
          Grand_id: waterbody_id
          Hylak_id: Hylak_id
          Lake_area: Area_avg
          G_CAP_MAX: Capacity_max
          G_CAP_REP: Capacity_norm
          G_CAP_MIN: Capacity_min
          G_DAM_HGT_: Dam_height
          Vol_total: Vol_avg
          Depth_avg: Depth_avg
          Dis_avg: Dis_avg
          Pour_long: xout
          Pour_lat: yout
        unit_mult:
          Area_avg: 1000000.
          Vol_avg: 1000000.
          Capacity_max: 1000000.
          Capacity_norm: 1000000.
          Capacity_min: 1000000.
        meta:
          category: surface water
          source_version: 1.0
          paper_ref: Alessia Matano
          source_info: GRanD.v1.1_HydroLAKES.v10_JRC.2016

Additional reading properties with the driver
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
In order to read specific datasets, HydroMT uses several drivers depending on the type of datasets. These options can be added in the yaml library 
by specifying the **driver** to use to read the data and then providing a list of related arguments in the **kwargs** sub-list.


In order to read *RasterDataset* and *GeoDataset*, HydroMT uses functions from the `xarray library <http://xarray.pydata.org/en/stable/index.html>`_. 
Thus any available option in xarray to open raster data can be initialised in the HydroMT yaml file in the **kwargs** sub-list. 
Depending on the type of the raster data, several drivers connected to different xarray functions are used:

- *raster*: for GDAL rasters. Uses `open_rasterio <http://xarray.pydata.org/en/stable/generated/xarray.open_rasterio.html>`_ function of xarray.
- *netcdf*: for NetCDF rasters. Uses `open_mfdataset <http://xarray.pydata.org/en/stable/generated/xarray.open_mfdataset.html>`_ function of xarray.
- *zarr*: for zarr rasters. Uses `open_zarr <http://xarray.pydata.org/en/stable/generated/xarray.open_zarr.html>`_ function of xarray.

|xarrayIcon|

An example for a dynamic raster dataset (read using the open_mfdataset from xarray) is 
shown below:

.. code-block:: console

    root: p:/wflow_global/forcing
    category: meteo
    chirps:
      path: CHIRPS/CHIRPS_rainfall_{year}.nc
      data_type: RasterDataset
      driver: netcdf
      kwargs:
        chunks: {time: 100, lat: 100, lon: 100}
        concat_dim: time
        decode_times: True
        combine: by_coords
        parallel: True
      crs: 4326
      rename:
        precipitation: precip
      unit_add:
        time: 86400 # [sec] 1D shift to set 'right' labels
      meta:
        source_version: v2.0
        source_url: https://www.chc.ucsb.edu/data/chirps
        paper_ref: Funk et al (2015)
        paper_doi: 10.1038/sdata.2015.66
        source_license: CC


In order to read *GeoDataFrame* and *GeoDataset*, HydroMT uses functions from the `GeoPandas library <https://geopandas.org/index.html>`_. 
Thus any available option in geopandas to open vector data can be initialised in the HydroMT yaml file in the **kwargs** sub-list. 
For vector data, there is only one driver defined:

- *vector*: for GDAL vectors. Uses `read_file <https://geopandas.org/docs/reference/api/geopandas.read_file.html#geopandas.read_file>`_ function of GeoPandas.

|geopandasIcon|

One example of vector data is shown below.

.. code-block:: console

      GDP_world:
        path: base/emissions/GDP-countries/World_countries_GDPpcPPP.gpkg
        data_type: GeoDataFrame
        crs: 4326
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


.. _data_convention:

Conventions on variable names and units
---------------------------------------
This section lists the different variable naming and unit conventions of HydroMT by types. This list is still in development. 
Names and units mentioned here are mandatory in order for the input data to be processed correctly and produced the right derived data. 
It is also possible to use the rename option so that variables and model data produced by HydroMT have more explicit names.

Coordinates
^^^^^^^^^^^
- time: time or date stamp [datetime].
- x: longitude. Several names are supported in HydroMT ["x", "longitude", "lon", "long"]. If the name is different, please rename using the yaml.
- y: longitude. Several names are supported in HydroMT ["y", "latitude", "lat"]. If the name is different, please rename using the yaml.


Topography
^^^^^^^^^^

- elevtn: altitude [m].
- mdt: mean dynamic topography [m].
- flwdir: flow direction. Format supported are ArcGIS D8, LDD, NEXTXY. The format is inferred from the data.
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


Available global datasets
-------------------------
Below is the list of data sources directly available in HydroMT (within the Deltares network).

.. csv-table:: Data Catalog
   :file: ../_generated/data_sources.csv
   :header-rows: 1
   :widths: auto
   :width: 50%

.. |xarrayIcon| image:: ../img/xarray-icon.png
.. |geopandasIcon| image:: ../img/geopandas-icon.png