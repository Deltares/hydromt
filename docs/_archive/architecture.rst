Architecture
============
General Architecture
--------------------
The HydroMT package contains GIS, hydrological, statistical and plotting methods to enable
the use of global and local datasets in environmental models and analyze these data as well as 
model results. It leverage the rich scientific python eco-system by building on many 
packages including xarray, geopandas, rasterio, scipy, pyflwdir. 

The architecture of the package contains several submodules (shown in the next figure):

- **Methods**: core of the package containing a range of methods for 
  raster (raster) and geometry (vector) based GIS, hydro- and topograpahy data (flw), plotting (plots) 
  and statistics (stats).
- **Data adapter**: parser from input data to unified internal raster, point or vector representation.
- **Worflows**: methods are combined in workflows for commenly used transformations from input 
  to model data.
- **Models**: contains a basic general model API class which can easiliy be 
  implemented and, if required extended, for individual models.
- **Interface**: the package is accessible both via a python interface and a higher level command line interface 
  for setting up models.

.. image:: img/hydromt_architecture_full.png

The package so far is optimized for models which require regular gridded data, vector data 
as static input maps. Data analysis is focussed on timeseries at point locations or 
regular gridded data. Methods to support models which require input data or preduce output 
in other types of grid definitions are not (yet) supported.

Methods
-------
At it's core the package exists of a range of methods for raster (raster) and geometry (vector) 
based GIS, hydro- and topograpahy data (flw), plotting (plots) and statistics (stats).
The package adopts the xarray data structure for any type of regular gridded data, while
geometry data is handled by the geopandas package. We combined methods from rasterio and xarray into a powerfull 
**raster submodule** for all raster GIS methods. For hydro- and topographical methods we make 
use of the pyflwdir package of which some function are wrapped to work directly on 
xarray object in the **flw submodule**. We anticipate to extent the library with a 
**stats submudule** for statistical functions for xarray objects which are particular
to the hydroshpere and/or not available in xarray. The **plots submudule** will contain
convienence methods for commenly used plots of input maps as well as output maps and 
timeseries in a standardized layout. The total suite of these methods, conveniently 
wrapped for xarray and (geo)pandas datatypes, creates a rich environment for 
analysis and manipulation of geospatial data.

Data
----
HydroMT uses many (global) datasets to build models. These datasets are available
within the Deltares network and linked to hydroMT through a library yml file. 
This file is parsed by hydroMT after which the the dataset can be accessed by just
its name through the :meth:`~hydromt.data_adapter.RasterDataAdapter` and 
:meth:`~hydromt.data_adapter.VectorDataAdapter` classes. 

Workflows
---------
Methods are combined in workflows for commenly used transformations from input 
to model data. Workflows typically receive xarray and or geopandas data and have key-word
arguments to define the standerdized variable names of required variables. An example
workflow is to define a list of input hydrography parameters (e.g.: flow direction, 
river length, river slope, upstream area, etc.) at model resolution based on input 
elevation data from (global) datasets.

Models
------
The models submodule contains a basic general model API class which can easiliy be 
implemented and, if required extended, for individual models. The general model API 
contains methods to read, write and build models and ensures a common interface to 
different models.

Interface
---------
The python interface is more flexible and allows to build models from 
a short script of just a few lines and is the only interface for analysis of model 
results. Besides the python interface we developed a higher level command line interface 
for setting up models. Highly variable arguments such as model in/output path and model 
region are set from the command line wheras other required argumens which very less between 
model instances can be passed on through a hydromt configuration file. 