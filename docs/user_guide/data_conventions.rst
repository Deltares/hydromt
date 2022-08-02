.. _data_convention:

Data conventions
================

Names and units mentioned here are mandatory in order for the input data to be processed correctly and produced the right derived data.
It is possible to use the ``rename`` option in the :ref:`data catalog yaml file <data_yaml>` so that data variables have more correct names.
This section lists the different variable naming and unit conventions of HydroMT by types.  
A list of recognized :ref:`dimension names <dimensions>` is found here.

.. NOTE::

    This list is still in development.


Topography
^^^^^^^^^^

============================  =======================================================================  ================
Name                          Explanation                                                              Unit
============================  =======================================================================  ================
elevtn                        altitude                                                                 [m]
mdt                           mean dynamic topography                                                  [m]
flwdir                        flow direction. Format supported are ArcGIS D8, LDD, NEXTXY.
                              The format is inferred from the data.
uparea                        upstream area                                                            [km2]
lndslp                        slope                                                                    [m/m]
strord                        Stralher streamorder                                                     [-]
basins                        basins ID mapping                                                        [-]
============================  =======================================================================  ================

Surface waters
^^^^^^^^^^^^^^
Rivers
""""""
============================  =======================================================================  ================
Name                          Explanation                                                              Unit
============================  =======================================================================  ================
rivlen                        river length                                                             [m]
rivslp                        river slope                                                              [m/m]
rivwth                        river width                                                              [m]
rivmsk                        mask of river cells (for raster models)                                  [bool]
============================  =======================================================================  ================

Reservoirs / lakes
""""""""""""""""""
============================  =======================================================================  ================
Name                          Explanation                                                              Unit
============================  =======================================================================  ================
waterbody_id                  reservoir/lake ID                                                        [-]
Hylak_id                      ID from the HydroLAKES database (to connect to the hydroengine library)  [-]
Area_avg                      average waterbody area                                                   [m2]
Vol_avg                       average waterbody volume                                                 [m3]
Depth_avg                     average waterbody depth                                                  [m]
Dis_avg                       average waterbody discharge                                              [m3/s]
xout                          longitude of the waterbody outlet                                        [-]
yout                          latitude of the waterbody outlet                                         [-]
Capacity_max                  maximum reservoir capacity volume                                        [m3]
Capacity_norm                 normal/average reservoir capacity volume                                 [m3]
Capacity_min                  minimum reservoir capcity volume                                         [m3]
Dam_height                    height of the dam                                                        [m]
============================  =======================================================================  ================

Glaciers
""""""""
============================  =======================================================================  ================
Name                          Explanation                                                              Unit
============================  =======================================================================  ================
simple_id                     glacier ID in the current database                                       [-]

============================  =======================================================================  ================

Landuse and landcover
^^^^^^^^^^^^^^^^^^^^^

============================  =======================================================================  ================
Name                          Explanation                                                              Unit
============================  =======================================================================  ================
landuse                       landuse classification                                                   [-]
LAI                           Leaf Area Index                                                          [-]
============================  =======================================================================  ================

Soil
^^^^

============================  =======================================================================  ================
Name                          Explanation                                                              Unit
============================  =======================================================================  ================
bd_sl*                        bulk density of the different soil layers (1 to 7 in soilgridsv2017)     [g cm-3]
clyppt_sl*                    clay content of the different soil layers (1 to 7 in soilgridsv2017)     [%]
oc_sl*                        organic carbon contant of the different soil layers
                              (1 to 7 in soilgridsv2017)                                               [%]
ph_sl*                        pH of the different soil layers (1 to 7 in soilgridsv2017)               [-]
sltppt_sl*                    silt content of the different soil layers (1 to 7 in soilgridsv2017)     [%]
sndppt_sl*                    sand content of the different soil layers (1 to 7 in soilgridsv2017)     [%]
soilthickness                 soil thickness                                                           [cm]
tax_usda                      USDA soil classification                                                 [-]
============================  =======================================================================  ================

Meteorology
^^^^^^^^^^^

============================  =======================================================================  ================
Name                          Explanation                                                              Unit
============================  =======================================================================  ================
precip                        precipitation (rainfall+snowfall)                                        [mm]
temp                          average temperature                                                      [oC]
temp_min                      minimum temperature                                                      [oC]
temp_max                      maximum temperature                                                      [oC]
press_msl                     atmospheric pressure                                                     [hPa]
kin                           shortwave incoming radiation                                             [W m-2]
kout                          TOA incident solar radiation                                             [W m-2]
============================  =======================================================================  ================

Hydrology
^^^^^^^^^

============================  =======================================================================  ================
Name                          Explanation                                                              Unit
============================  =======================================================================  ================
run                           surface water runoff (overland flow + river discharge)                   [m3/s]
vol                           water volumes                                                            [m3]
infilt                        water infiltration in the soil                                           [m3/s]
runPav                        excess infiltration runoff on paved areas                                [m3/s]
runUnp                        excess infiltration runoff on unpaved areas                              [m3/s]
inwater                       sum of all fluxes entering/leaving the surface waters (precipitation,
                              evaporation, infiltration...)                                            [m3/s]
inwaterInternal               sum of all fluxes between the land and river surface waters
                              (part of inwater)                                                        [m3/s]
============================  =======================================================================  ================