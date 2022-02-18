

Terminology and conventions
============================

.. _terminology:

Terminology
-----------

HydroMT and this documentation use a specific terminology to describe specific objects or processes.

==============================  ======================================================================================
Term                            Explanation
==============================  ======================================================================================
Attributes                      direct properties of a model, such as root or crs. They can be called when using hydroMT from python.
Basemaps                        basic maps representing the model schematization/grid (usually DEM at the model resolution). This is the first thing HydroMT
                                prepares when building a model from a region argument.
Command Line Interface (CLI)    high-level interface to HydroMT methods.
Components                      parts of a model linked to a specific HydroMT function. For example, basemaps, rivers, soil, forcing etc. They are specific
                                to each model.
Configuration (general)         (.ini) file setting the different model components and options to be processed by HydroMT methods.
Configuration (models)          for the model, this is one or several files used to set-up and run the model. They can be updated using the setup_config
                                component. In HydroMT, the config object is a nested dictionary.
Data catalog                    complete list of data sources available for HydroMT. This object is internal to HydroMT and can be viewed in a csv file
                                after running HydroMT methods.
Data library                    (.yml) files containing one or several data sources to be used by HydroMT and their properties.
Data source                     input data. To be processed by HydroMT, data sources are listed in data libraries.
Forcing                         model (dynamic) forcing data (meteo or hydrological for example). In HydroMT, this is a dictionary of xarray DataArray that is updated
                                each time a component of the forcing type is run (eg setup_precip_forcing for wflow).
Method                          HydroMT high level functions available from the CLI to interact with models. These are *build*, *update* and *cli*.
Model                           models that are integrated into the HydroMT framework and with which the user can interact. For example *wflow*, *sfincs* etc.
Region                          argument of the *build* method that specifies the region of interest where the model should be prepared.
Staticgeoms                     model (static) vector data or information. In HydroMT, this is a dictionary of GeoPandas GeoDataFrame that is updated
                                when certain components are run (eg setup_basemaps).
Staticmaps                      model (static) gridded data such as land properties and model parameters. In HydroMT, this is a xarray DataSet that is updated
                                when most of the components are run (eg setup_basemaps).
==============================  ======================================================================================

.. _data_convention:

Conventions on variable names and units
---------------------------------------

This section lists the different variable naming and unit conventions of HydroMT by types. This list is still in development.
Names and units mentioned here are mandatory in order for the input data to be processed correctly and produced the right derived data.
It is also possible to use the rename option so that variables and model data produced by HydroMT have more explicit names.
A list of recognized :ref:`dimensions <dimensions>` is found here.

Topography
^^^^^^^^^^

============================  =======================================================================  ================
Name                          Explanation                                                              Unit
============================  =======================================================================  ================
elevtn                        altitude                                                                 [m]
mdt                           mean dynamic topography                                                  [m]
flwdir                        flow direction. Format supported are ArcGIS D8, LDD, NEXTXY.
                              The format is infered from the data.
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