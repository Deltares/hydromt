Model API
=========
The models submodule contains a basic general model API class which can easiliy be 
implemented and, if required extended, for individual models. The general model API 
contains methods to read, write and build models and ensures a common interface to 
different models. Workflows can be combined and wrapped into methods such as the 
build method of the model class.

HydroMT has a general model class which has identical methods for each model.
The most important methods are discussed here and model specific methods in each of 
subsections. A full list of all methods can be found in the reference API.

.. image:: ../img/hydromt_architecture_models.png

Model API:

* Properties:

    * :attr:`~hydromt.models.model_api.Model.root`: path to model root.

    * :attr:`~hydromt.models.model_api.Model.config`: :meth:`dict` interface to the model configuration.

    * :attr:`~hydromt.models.model_api.Model.staticmaps`: an :meth:`xarray.Dataset` interface to all model staticmaps. 

    * :attr:`~hydromt.models.model_api.Model.staticgeoms`: a :meth:`dict` of :meth:`geopandas.GeoDataFrame` interface to all model geometry files.

    * :attr:`~hydromt.models.model_api.Model.crs`: the model coordinate reference system.

    * :attr:`~hydromt.models.model_api.Model.bounds`: the model bounding box.

    * :attr:`~hydromt.models.model_api.Model.region`: the geomtery of model bounding box as :meth:`geopandas.GeoDataFrame`.


* Methods:

    * :meth:`~hydromt.models.model_api.Model.build`: method to build a complete model from scratch.

    * :meth:`~hydromt.models.model_api.Model.setup_component`: method to update or setup a component once the basemaps have been setup.

    * :meth:`~hydromt.models.model_api.Model.read_config` / :meth:`~hydromt.models.model_api.Model.write_config`: parse model config to and from file.

    * :meth:`~hydromt.models.model_api.Model.get_config`: get a config option

    * :meth:`~hydromt.models.model_api.Model.set_config`: set a value to a config option

    * :meth:`~hydromt.models.model_api.Model.read_staticmaps` / :meth:`~hydromt.models.model_api.Model.write_staticmaps`: parse model maps to and from disk (often several files in model specific format).

    * :meth:`~hydromt.models.model_api.Model.set_staticmaps`: add a map to the model staticmaps.

    * :meth:`~hydromt.models.model_api.Model.read_staticgeoms` / :meth:`~hydromt.models.model_api.Model.write_staticgeoms`: parse model geometry data to and from file.

    * :meth:`~hydromt.models.model_api.Model.set_staticgeoms`: Add a geometry to the model staticgeoms.

