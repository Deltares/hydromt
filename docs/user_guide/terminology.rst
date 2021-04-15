.. _terminology:

Terminology
===========

HydroMT and this documentation use a specific terminology to describe specific objects or processes.


For **HydroMT** in general, these are:

- Command Line Interface (CLI): high-level interface to HydroMT methods.
- Configuration: (*.ini) file setting the different model components and options to be processed by HydroMT methods.
- Method: HydroMT high level functions available from the CLI to interact with models. These are *build*, *update* and *cli*.
- Model: models that are integrated into the HydroMT framework and with which the user can interact. For example *wflow*, *sfincs* etc.
- Region: argument of the *build* method that specifies the region of interest where the model should be prepared.

For **Data** in HydroMT, these are:

- Data catalog: complete list of data sources available for HydroMT. This object is internal to HydroMT and can be viewed in a csv file 
  after running HydroMT methods.
- Data library: (*.yml) files containing one or several data sources to be used by HydroMT and their properties.
- Data source: input data. To be processed by HydroMT, data sources are listed in data libraries.

For **Models** in HydroMT, these are:

- Attributes: direct properties of a model, such as root or crs. They can be called when using hydroMT from python.
- Basemaps: basic maps representing the model schematization/grid (usually DEM at the model resolution). This is the first thing HydroMT 
  prepares when building a model from a region argument.
- Components: parts of a model linked to a specific HydroMT function. For example, basemaps, rivers, soil, forcing etc. They are specific 
  to each model.
- Configuration: for the model, this is one or several files used to set-up and run the model. They can be updated using the setup_config 
  component. In HydroMT, the config object is a nested dictionnary.
- Forcing: model (dynamic) forcing data (meteo or hydrological for example). In HydroMT, this is a dictionnary of xarray DataArray that is updated 
  each time a component of the forcing type is run (eg setup_precip_forcing for wflow).
- Staticgeoms: model (static) vector data or information. In HydroMT, this is a dictionnary of GeoPandas GeoDataFrame that is updated 
  when certain components are run (eg setup_basemaps).
- Staticmaps: model (static) gridded data such as land properties and model parameters. In HydroMT, this is a xarray DataSet that is updated 
  when most of the components are run (eg setup_basemaps).

