Why HydroMT
============

Scope and aspirations
---------------------

Setting up spatially-distributed geoscientific models typically requires many (manual) steps 
to process input data and might therefore be time consuming and hard to reproduce. Furthermore, 
it can be hard to improve models based on new or updated (large) datasets, such as (global) 
digital elevation models and land use maps, potentially slowing down the uptake of such datasets 
for geoscientific modelling.

HydroMT is an open-source Python package that aims to facilitate the process of building models 
and analyzing model results based on the state-of-the-art scientific python ecosystem, including 
xarray, geopandas, rioxarray, pyflwdir, numpy, scipy and dask. The package provides a common interface 
to data and models as well as workflows to transform data to models and analyze model results based on 
(hydrological) GIS and statistical methods. The common data interface is implemented through a data 
catalog, which is setup with a simple text yaml file, and supports many different (GIS) data formats 
and some simple pre-processing steps such as unit conversion. The common model interface is implemented 
per model software package and provides a standardized representation of the model configuration, maps, 
geometries, forcing, states and results. The user can describe a full model setup including its forcing 
in a single ini text file based on a sequence of workflows, making the process reproducible, fast and 
modular. Besides the Python interface, HydroMT has a command line interface (CLI) to build, update or 
analyze models. 

The package has been designed with an iterative, data-centered modelling process in mind. First-order 
models can be setup for any location in the world by leveraging open global datasets. These models can 
later be improved by updating the input datasets with detailed local datasets. This iterative process 
enables the user to quickly get an initial model and analyze its result to then make informed decisions 
about the most relevant model improvements and/or required data collection and to kick-start discussions 
with stakeholders. Furthermore, model parameter maps or forcing data can easily be modified for model 
sensitivity analysis or model calibration to support robust modelling practices. 

Usage
-----
# TODO data / models / cli

Model plugins
-------------
While the HydroMT core package lays out the common data an model interface and has many useful workflows and 
methods, model specific implementations are supported through plugins. This makes it easy to implement HydroMT
for a specific model, while using the core hydroMT functionality and sharing the sharing the same way of working. 


