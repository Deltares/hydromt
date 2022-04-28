
.. _terminology:

Terminology
===========

HydroMT and this documentation use a specific terminology to describe specific objects or processes.

==============================  ======================================================================================
Term                            Explanation
==============================  ======================================================================================
Basemaps                        basic maps representing the model schematization/grid (usually DEM at the model resolution). This is the first thing HydroMT
                                prepares when building a model from a region argument.
Command Line Interface (CLI)    high-level interface to HydroMT *build*, *update* and *clip* methods.
Configuration (HydroMT)         (.ini) file describing the complete pipeline with all methods and their arguments to *build* or *update* a model.
Configuration (models)          For the model, this is one or several files used to configure a model simulation. They can be updated using the setup_config
                                method. In HydroMT, it is exposed in the model config attribute as a nested dictionary.
Data catalog                    A set of data sources available for HydroMT. It is build up from *yml* files containing one or more data sources with 
                                information about how to read and optionally preprocess the data and meta-data about the data source.
Data source                     Input data. To be processed by HydroMT, data sources are listed in yml files.
Forcing                         A model attribute with (dynamic) forcing data (meteo or hydrological for example). In HydroMT, this is a dictionary of xarray DataArray that is updated
                                each time a model *forcing* method is run (eg setup_precip_forcing for wflow).
Model                           A set of files describing the schematization, forcing, states, simulation configuration and results for any supported model kernel.
Model attributes                Direct properties of a model, such as the model root. They can be called when using HydroMT from python.
Model data component            The model data of all model plugins is described by HydroMT with the following components: staticmaps, staticgeoms, forcing, results, states, config
Model plugin                    Model software for HydroMT can build and update models and analyze its simulation results. For example *wflow*, *sfincs* etc.
Model kernel                    The model software to execute a model simulation. This is *not* part of any HydroMT plugin.
Region                          argument of the *build* method that specifies the region of interest where the model should be prepared.
Staticgeoms                     A model data component with static vector data. In HydroMT, this is a dictionary of GeoPandas GeoDataFrame that is updated
                                when certain model methods are run.
Staticmaps                      A model attribute with (static) gridded data such as land properties and model parameters. In HydroMT, this is a xarray DataSet that is updated
                                when certain model methods are run.
==============================  ======================================================================================