.. _data_convention:

Data conventions
================

In order for HydroMT to process data from one dataset or another, we use a set of
conventions for the data naming and units. This allows HydroMT to process the input data
correctly and produce the right derived data.

If your input data has different names and units, it is possible to use the ``rename``
option in the :ref:`data catalog yaml file <data_yaml>` so that data variables have
hydroMT-compatible names and the ``unit_mult`` and ``unit_add`` options for the units.

This section lists the different variable naming and unit conventions of HydroMT by
types. A list of recognized :ref:`dimension names <dimensions>` is found here.

.. NOTE::

    Variables names in **bold** are mandatory if you wish to use functionalities
    from HydroMT core or one of the pre-defined catalogs. Other variables are advices
    for example for use with plugins (e.g. hydromt_wflow). Always check the plugins
    requirements when using them and creating your own data catalogs.

General rules:

- Variable names should be lowercase and use underscores as word separators.
- Variable names should be self-explanatory.
- *index* is used to specify index or identifier variables (for example, station, ID,
  basin ID, country ID etc.)
- To specify a variable with statistics, use the variable name and statistic name
  separated by underscores (e.g. ``precipitation_avg``, ``precipitation_min`` etc.).
  Recognized statistics are: ``avg``, ``min``, ``max``, ``std``, ``sum``, ``count``.
- To specify additional types for a variable, use the variable name and type name
  separated by underscores (e.g. ``precipitation_rate``, ``precipitation_wet``,
  ``gdp_net``, ``gdp_gross``, ``roughness_manning``, ``id_lake`` etc.).
- Units should follow the base SI units as much as possible (e.g. ``m`` for meter, ``s``
  for second, ``kg`` for kilogram, ``m3`` for cubic meter, ``m3 s-1`` for cubic meter
  per second, ``m3 s-1 m-2`` for cubic meter per second per square meter etc.).
  Base SI units: ``m`` for meter, ``s`` for second, ``kg`` for kilogram, ``m3`` for
  cubic meter, ``oC`` for degree Celsius, ``W`` for Watt, ``J`` for Joule, ``Pa`` for
  Pascal, ``N`` for Newton.



Dimensions
^^^^^^^^^^
These conventions apply to the dimensions names of the data variable(s).

======================================== ================================ ======================
Name                                     Explanation                      Unit
======================================== ================================ ======================
**time**                                 time or date stamps              [datetime]
month                                    month of the year                [month]
year                                     year                             [year]
**x**, **longitude**, **lon**, **long**  longitude                        [degree_east] or [m]
**y**, **latitude**, **lat**             latitude                         [degree_north] or [m]
**z**                                    altitude                         [m+ref] or [m]
**return_period**                        return period of the variable    [year]
**index**                                index (e.g. station index or ID) [-]
layer                                    (soil) layer                     [-]
======================================== ================================ ======================

General characteristics
^^^^^^^^^^^^^^^^^^^^^^^
This category describes general variables that are not specific to a particular
type of data. They are used for example to describe the id or dimension (area, length)
of an object (e.g. basin, river, reservoir, glacier, landuse, soil etc.).

============================  =======================================================================  ================
Name                          Explanation                                                              Unit
============================  =======================================================================  ================
**id**                        unique identifier of the object                                          [-]
**mask**                      mask of the object (for gridded/mesh data)                               [bool]
**area**                      area of the object                                                       [m2]
**length**                    length of the object                                                     [m]
**width**                     width of the object                                                      [m]
**height**                    height of the object                                                     [m]
**depth**                     depth of the object                                                      [m]
diameter                      diameter of the object                                                   [m]
**volume**                    volume of the object                                                     [m3]
============================  =======================================================================  ================

Geography
^^^^^^^^^
Geographic variables.

============================  =======================================================================  ================
Name                          Explanation                                                              Unit
============================  =======================================================================  ================
country                       country name                                                             [-]
**country_iso**               country ID / ISO3 code                                                   [-]
**coast**                     coast line                                                               [-]
**land**                      land boundaries                                                          [-]
============================  =======================================================================  ================

Topography
^^^^^^^^^^
Topography and elevation related variables.

============================  =======================================================================  ================
Name                          Explanation                                                              Unit
============================  =======================================================================  ================
**elevation**                 altitude                                                                 [m+ref]
**flow_direction**            flow direction. Format supported are ArcGIS D8, LDD, NEXTXY.
                              The format is inferred from the data.
**upstream_area**             upstream area                                                            [m2]
**slope**                     (land) slope                                                             [m/m]
**streamorder**               Stralher streamorder                                                     [-]
**basin**                     basin ID mapping                                                         [-]
============================  =======================================================================  ================

Surface water
^^^^^^^^^^^^^
Surface water related variables.

River
"""""
============================  =======================================================================  ================
Name                          Explanation                                                              Unit
============================  =======================================================================  ================
**length**                    river length                                                             [m]
**slope**                     river slope                                                              [m/m]
**width**                     (average) river width                                                    [m]
depth                         river depth                                                              [m]
bed_level                     river bed level                                                          [m]
bed_width                     river bed width                                                          [m]
bankfull_level                river bankfull level                                                     [m]
bankfull_width                river bankfull width                                                     [m]
bankfull_discharge            river bankfull discharge                                                 [m3/s]
roughness                     roughness of the river bed                                               [s/m1/3]
**mask**                      mask of river cells (for raster models)                                  [bool]
============================  =======================================================================  ================

Reservoir / lake
""""""""""""""""
============================  =======================================================================  ================
Name                          Explanation                                                              Unit
============================  =======================================================================  ================
**id**                        reservoir/lake ID                                                        [-]
**id_hydrolakes**             ID from the HydroLAKES database (to connect to the hydroengine library)  [-]
area                          reservoir/lake surface area (dynamic)                                    [m2]
**area_avg**                  average waterbody surface area                                           [m2]
volume                        reservoir/lake volume (dynamic)                                          [m3]
**volume_avg**                average waterbody volume                                                 [m3]
**volume_max**                maximum waterbody volume                                                 [m3]
depth                         reservoir/lake depth (dynamic)                                           [m]
**depth_avg**                 average waterbody depth                                                  [m]
waterlevel                    reservoir/lake water level (dynamic)                                     [m+ref]
discharge                     reservoir/lake discharge (dynamic)                                       [m3/s]
**discharge_avg**             average waterbody discharge                                              [m3/s]
xout                          longitude of the waterbody outlet                                        [-]
yout                          latitude of the waterbody outlet                                         [-]
**capacity_max**              maximum reservoir capacity volume                                        [m3]
**capacity_norm**             normal/average reservoir capacity volume                                 [m3]
**capacity_min**              minimum reservoir capacity volume                                        [m3]
**dam_height**                height of the dam                                                        [m]
============================  =======================================================================  ================

Glacier
"""""""
============================  =======================================================================  ================
Name                          Explanation                                                              Unit
============================  =======================================================================  ================
**id**                        glacier ID in the current database                                       [-]
============================  =======================================================================  ================

Landuse and landcover
^^^^^^^^^^^^^^^^^^^^^
Landuse and landcover related variables.

============================  =======================================================================  ================
Name                          Explanation                                                              Unit
============================  =======================================================================  ================
**landuse**                   landuse classification                                                   [-]
crop                          crop type                                                                [-]
roughness                     surface roughness of the land                                            [s/m1/3]
vegetation_height             vegetation height (canopy)                                               [m]
root_depth                    depth of the vegetation roots                                            [m]
**lai**                       leaf area index                                                          [-]
============================  =======================================================================  ================

Soil
^^^^
Soil related variables and properties.

============================  =======================================================================  ================
Name                          Explanation                                                              Unit
============================  =======================================================================  ================
**bulk_density_sl\***          bulk density of the different soil layers (1 to 7 in soilgridsv2017)     [kg m-3]
**clay_sl\***                  clay content of the different soil layers (1 to 7 in soilgridsv2017)     [kg/kg]
**organic_carbon_sl\***        organic carbon content of the different soil layers
                              (1 to 7 in soilgridsv2017)                                               [kg/kg]
**ph_sl\***                   pH of the different soil layers (1 to 7 in soilgridsv2017)               [-]
**silt_sl\***                 silt content of the different soil layers (1 to 7 in soilgridsv2017)     [kg/kg]
**sand_sl\***                 sand content of the different soil layers (1 to 7 in soilgridsv2017)     [kg/kg]
**thickness**                 soil thickness                                                           [m]
porosity                      soil porosity / saturated soil water content                             [m3/m3]
hydraulic_conductivity        (saturated) hydraulic conductivity of the soil                           [m s-1]
texture                       USDA soil texture classification                                         [-]
**taxonomy**                  USDA soil classification                                                 [-]
============================  =======================================================================  ================

Meteorology
^^^^^^^^^^^
Meteorological variables.

================================  =======================================================================  ================
Name                              Explanation                                                              Unit
================================  =======================================================================  ================
**precipitation**                 precipitation (rainfall+snowfall)                                        [mm]
precipitation_rate                precipitation rate                                                       [mm hr-1]
**potential_evapotranspiration**  potential evapotranspiration                                             [mm]
evapotranspiration                actual evapotranspiration                                                [mm]
**temperature**                   (average) temperature                                                    [oC]
**temperature_min**               minimum temperature                                                      [oC]
**temperature_max**               maximum temperature                                                      [oC]
**temperature_dew**               dewpoint temperature                                                     [oC]
**pressure_msl**                  atmospheric pressure at mean sea level                                   [Pa]
**pressure**                      atmospheric pressure at 2m elevation                                     [Pa]
**humidity**                      relative humidity                                                        [%]
**radiation**                     shortwave incoming radiation                                             [J m-2]
**radiation_incident**            TOA incident solar radiation                                             [J m-2]
**radiation_net**                 surface net solar radiation                                              [J m-2]
**cloud_cover**                   total fraction of cloud cover                                            [0-1]
**wind**                          2m wind speed                                                            [m s-1]
**wind_u**                        2m wind U-component                                                      [m s-1]
**wind_v**                        2m wind V-component                                                      [m s-1]
================================  =======================================================================  ================

Hydrology
^^^^^^^^^
Hydrological variables.

============================  =======================================================================  ================
Name                          Explanation                                                              Unit
============================  =======================================================================  ================
discharge                     (river) discharge                                                        [m3/s]
overland_flow                 overland flow                                                            [m3/s]
subsurface_flow               subsurface flow                                                          [m3/s]
volume                        water volume                                                             [m3]
waterlevel                    water level (above reference)                                            [m+ref]
waterdepth                    water depth                                                              [m]
infiltration                  water infiltration in the soil                                           [m3/s]
infiltration_capacity         soil infiltration capacity                                               [m3/s]
curve_number                  runoff curve number                                                      [-]
**demand_domestic_gross**     gross domestic water demand                                              [m3/s]
**demand_domestic_net**       net domestic water demand                                                [m3/s]
**demand_industry_gross**     gross industrial water demand                                            [m3/s]
**demand_industry_net**       net industrial water demand                                              [m3/s]
demand_agriculture_gross      gross agricultural water demand                                          [m3/s]
demand_agriculture_net        net agricultural water demand                                            [m3/s]
**demand_livestock_gross**    gross livestock water demand                                             [m3/s]
**demand_livestock_net**      net livestock water demand                                               [m3/s]
demand_energy_gross           gross energy water demand                                                [m3/s]
demand_energy_net             net energy water demand                                                  [m3/s]
demand_environmental_gross    gross environmental water demand                                         [m3/s]
demand_environmental_net      net environmental water demand                                           [m3/s]
============================  =======================================================================  ================

Oceanograhy
^^^^^^^^^^^
Oceanographic variables.

============================  =======================================================================  ================
Name                          Explanation                                                              Unit
============================  =======================================================================  ================
wave_level                    wave level or height                                                     [m]
wave_period                   wave period                                                              [s]
wave_direction                mean wave direction                                                      [degree]
============================  =======================================================================  ================

Socio-economic
^^^^^^^^^^^^^^
Socio-economic variables.

============================  =======================================================================  ================
Name                          Explanation                                                              Unit
============================  =======================================================================  ================
**population**                population (capita)                                                      [cap]
population_density            population density (capita per m2)                                       [cap/m2]
**gdp**                       gross domestic product (per capita)                                      [USD/cap]
============================  =======================================================================  ================
