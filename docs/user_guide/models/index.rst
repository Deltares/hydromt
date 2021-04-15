.. _ini_options:

Models
======
General options and components
------------------------------

Model building with HydroMT is a very flexible process where the user is fully in control of which 
model components he wishes to prepare, using a specific dataset and process options. The choices of the 
different components to build/update and their options are defined in a configuration or ini file. The ini file is divided 
into different sections between brackets []. Each section corresponds to a specific model component in HydroMT,
for examples to setup rivers, reservoirs, landuse or soil data and parameters etc.

The list of available components is different for each model but here are a few generalities:

- [global]: contains options that are valid for every other section/model components as well. It is the first section of the ini file.
- [setup_config]: contains (run) options to change in the model configuration file.
- [setup_basemaps]: section that sets the base maps that define the model region (usually DEM and water direction).

The other components and model options are model specific.

.. note::

   For each model, examples of a default set-up ini file are available in the HydroMT package in the hydromt/examples folder.

Available model plugins
-----------------------

TO BE ADDED link to the model plugin documentation