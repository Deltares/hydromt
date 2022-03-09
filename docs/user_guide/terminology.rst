
.. _terminology:

Terminology
===========

HydroMT and this documentation use a specific terminology to describe specific objects or processes.

==============================  ======================================================================================
Term                            Explanation
==============================  ======================================================================================
Attributes                      direct properties of a model, such as root or crs. They can be called when using hydroMT from python.
Basemaps                        basic maps representing the model schematization/grid (usually DEM at the model resolution). This is the first thing HydroMT
                                prepares when building a model from a region argument.
Command Line Interface (CLI)    high-level interface to HydroMT *build*, *update* and *clip* methods.
Components                      parts of a model linked to a specific HydroMT function. For example, basemaps, rivers, soil, forcing etc. They are specific
                                to each model.
Configuration (HydroMT)         (.ini) file describing the complete pipeline with all components and its arguments to *build* or *update* a model.
Configuration (models)          for the model, this is one or several files used to configure a model simulation. They can be updated using the setup_config
                                component. In HydroMT, it is exposed in the model config attribute as a nested dictionary.
Data catalog                    complete list of data sources available for HydroMT. This object is internal to HydroMT and can be viewed as a table (DataFrame)
                                It is usually build up from *yml* files containing one or several data sources to be used by HydroMT and their properties.
Data source                     input data. To be processed by HydroMT, data sources are listed in yml files.
Forcing                         A model attribute with (dynamic) forcing data (meteo or hydrological for example). In HydroMT, this is a dictionary of xarray DataArray that is updated
                                each time a component of the forcing type is run (eg setup_precip_forcing for wflow).
Model                           models that are integrated into the HydroMT framework and with which the user can interact. For example *wflow*, *sfincs* etc.
Region                          argument of the *build* method that specifies the region of interest where the model should be prepared.
Staticgeoms                     A model attribute with static vector data. In HydroMT, this is a dictionary of GeoPandas GeoDataFrame that is updated
                                when certain components are run (eg setup_basemaps).
Staticmaps                      A model attribute with (static) gridded data such as land properties and model parameters. In HydroMT, this is a xarray DataSet that is updated
                                when most of the components are run (eg setup_basemaps).
==============================  ======================================================================================