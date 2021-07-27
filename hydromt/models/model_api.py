# -*- coding: utf-8 -*-
"""General and basic API for models in HydroMT"""

from abc import ABCMeta, abstractmethod
import os
from os.path import join, isdir, isfile, abspath
import xarray as xr
import numpy as np
import geopandas as gpd
from shapely.geometry import box
import logging
from pathlib import Path
import inspect

from ..data_adapter import DataCatalog
from .. import config

__all__ = ["Model"]

logger = logging.getLogger(__name__)


class Model(object, metaclass=ABCMeta):
    """General and basic API for models in HydroMT"""

    # FIXME
    _DATADIR = ""  # path to the model data folder
    _NAME = "modelname"
    _CONF = "model.ini"
    _CF = dict()  # configreader kwargs
    _GEOMS = {"<general_hydromt_name>": "<model_name>"}
    _MAPS = {"<general_hydromt_name>": "<model_name>"}
    _FOLDERS = [""]

    def __init__(
        self,
        root=None,
        mode="w",
        config_fn=None,
        data_libs=None,
        deltares_data=None,
        artifact_data=None,
        logger=logger,
    ):
        from . import ENTRYPOINTS  # load within method to avoid circular imports

        self.logger = logger
        ep = ENTRYPOINTS.get(self._NAME, None)
        version = ep.distro.version if ep is not None else ""
        dist = ep.distro.name if ep is not None else "unknown"
        self.logger.info(f"Initializing {self._NAME} model from {dist} (v{version}).")

        # link to data
        self.data_catalog = DataCatalog(
            data_libs=data_libs,
            deltares_data=deltares_data,
            artifact_data=artifact_data,
            logger=self.logger,
        )

        # placeholders
        self._staticmaps = xr.Dataset()
        self._staticgeoms = dict()  # dictionnary of gdp.GeoDataFrame
        self._forcing = dict()  # dictionnary of xr.DataArray
        self._config = dict()  # nested dictionary
        self._states = dict()  # dictionnary of xr.DataArray
        self._results = dict()  # dictionnary of xr.DataArray and/or xr.Dataset

        # model paths
        self._config_fn = self._CONF if config_fn is None else config_fn
        self.set_root(root, mode)

    def _check_get_opt(self, opt):
        """Check all opt keys and raise sensible error messages if unknonwn."""
        for method in opt.keys():
            m = method.strip("0123456789")
            if not callable(getattr(self, m, None)):
                if not hasattr(self, m) and hasattr(self, f"setup_{m}"):
                    raise DeprecationWarning(
                        f'Use full name "setup_{method}" instead of "{method}"'
                    )
                else:
                    raise ValueError(f'Model {self._NAME} has no method "{method}"')
        return opt

    def _run_log_method(self, method, *args, **kwargs):
        """Log method paramters before running a method"""
        method = method.strip("0123456789")
        func = getattr(self, method)
        signature = inspect.signature(func)
        for i, (k, v) in enumerate(signature.parameters.items()):
            v = kwargs.get(k, v.default)
            if v is inspect.Parameter.empty:
                if len(args) >= i + 1:
                    v = args[i]
                else:
                    continue
            self.logger.info(f"{method}.{k}: {v}")
        return func(*args, **kwargs)

    def build(
        self, region: dict, res: float = None, write: bool = True, opt: dict = None
    ):
        """Single method to setup and write a full model schematization and
        configuration from scratch

        Parameters
        ----------
        region: dict
            Description of model region. See :py:meth:`~hydromt.workflows.parse_region`
            for all options.
        res: float, optional
            Model restolution. Use only if applicable to your model. By default None.
        write: bool, optional
            Write the complete model schematization after setting up all model components.
            By default True.
        opt: dict, optional
            Model setup configuration. This is a nested dictionary where the first-level
            keys are the names of model spedific setup methods and the second-level
            keys the arguments of the method:

            ```{
                <name of method1>: {
                    <argument1>: <value1>, <argument2>: <value2>
                    }
                <name of method2>: {
                    ...
                    }
                }
            }```
        """
        opt = self._check_get_opt(opt)

        # run setup_config and setup_basemaps first!
        self._run_log_method("setup_config", **opt.pop("setup_config", {}))
        kwargs = opt.pop("setup_basemaps", {})
        kwargs.update(region=region)

        if res is not None:  # res is optional
            kwargs.update(res=res)
        self._run_log_method("setup_basemaps", **kwargs)

        # then loop over other methods
        for method in opt:
            self._run_log_method(method, **opt[method])

        # write
        if write:
            self.write()

    def update(self, model_out=None, write=True, opt=None):
        """Single method to setup and write a full model schematization and
        configuration from scratch


        Parameters
        ----------
        model_out: str, path, optional
            Desitation folder to write the model schematization after updating
            the model. If None the updated model components are overwritten in the
            current model schematization if these exist. By defualt None.
        write: bool, optional
            Write the updated model schematization to disk. By default True.
        opt: dict, optional
            Model update configuration. This is a nested dictionary where the first-level
            keys are the names of model spedific setup methods and the second-level
            keys the arguments of the method:

            ```{
                <name of method1>: {
                    <argument1>: <value1>, <argument2>: <value2>
                    }
                <name of method2>: {
                    ...
                    }
            }```
        """
        opt = self._check_get_opt(opt)

        # read current model
        if not self._write:
            if model_out is None:
                raise ValueError(
                    '"model_out" directory required when updating in "read-only" mode'
                )
            self.read()
            self.set_root(model_out, mode="w")

        # check if model has a region
        if self.region is None:
            raise ValueError(
                'Model region not found, setup basemaps using "build" method first.'
            )

        # remove setup_basemaps from options and throw warning
        if "setup_basemaps" in opt:
            opt.pop("setup_basemaps")  # remove from opt
            self.logger.warning(
                '"setup_basemaps" can only be called when building a model.'
            )

        # loop over other methods from ini file
        self._run_log_method("setup_config", **opt.pop("setup_config", {}))
        for method in opt:
            self._run_log_method(method, **opt[method])

        # write
        if write:
            self.write()

    ## file system

    @property
    def root(self):
        """Path to model folder."""
        if self._root is None:
            raise ValueError("Root unknown, use set_root method")
        return self._root

    def set_root(self, root, mode="w"):
        """Initialized the model root.
        In read mode it checks if the root exists.
        In write mode in creates the required model folder structure

        Parameters
        ----------
        root : str, optional
            path to model root
        mode : {"r", "r+", "w"}, optional
            read/write-only mode for model files
        """
        if mode not in ["r", "r+", "w"]:
            raise ValueError(f'mode "{mode}" unknown, select from "r", "r+" or "w"')
        old_root = getattr(self, "_root", None)
        self._root = root if root is None else abspath(root)
        self._read = mode.startswith("r")
        self._write = mode != "r"
        if root is not None:
            if self._write:
                for name in self._FOLDERS:
                    path = join(self._root, name)
                    if not isdir(path):
                        os.makedirs(path)
                    elif not self._read:
                        self.logger.warning(
                            "Model dir already exists and "
                            f"files might be overwritten: {path}."
                        )
            # check directory
            elif not isdir(self._root):
                raise IOError(f'model root not found at "{self._root}"')

    ## I/O

    @abstractmethod
    def read(self):
        """Method to read the complete model schematization and configuration from file."""
        self.read_config()
        self.read_staticmaps()

    @abstractmethod
    def write(self):
        """Method to write the complete model schematization and configuration to file."""
        self.write_config()
        self.write_staticmaps()

    def _configread(self, fn):
        return config.configread(fn, abs_path=False)

    def _configwrite(self, fn):
        return config.configwrite(fn, self.config)

    def read_config(self, config_fn=None):
        """Parse config from file. If no config file found a default config file is
        read in writing mode."""
        prefix = "User defined"
        if config_fn is None:  # prioritize user defined config path (new v0.4.1)
            if not self._read:  # write-only mode > read default config
                config_fn = join(self._DATADIR, self._NAME, self._CONF)
                prefix = "Default"
            elif self.root is not None:  # append or write mode > read model config
                config_fn = join(self.root, self._config_fn)
                prefix = "Model"
        cfdict = dict()
        if config_fn is not None:
            if isfile(config_fn):
                cfdict = self._configread(config_fn)
                self.logger.debug(f"{prefix} config read from {config_fn}")
            else:
                self.logger.error(f"{prefix} config file not found at {config_fn}")
        self._config = cfdict

    def write_config(self, config_name=None, config_root=None):
        """Write config to <root/config_fn>"""
        if not self._write:
            raise IOError("Model opened in read-only mode")
        if config_name is not None:
            self._config_fn = config_name
        elif self._config_fn is None:
            self._config_fn = self._CONF
        if config_root is None:
            config_root = self.root
        fn = join(config_root, self._config_fn)
        self.logger.info(f"Writing model config to {fn}")
        self._configwrite(fn)

    @abstractmethod
    def read_staticmaps(self):
        """Read staticmaps at <root/?/> and parse to xarray Dataset"""
        # to read gdal raster files use: hydromt.open_mfraster()
        # to read netcdf use: xarray.open_dataset()
        if not self._write:
            # start fresh in read-only mode
            self._staticmaps = xr.Dataset()
        raise NotImplementedError()

    @abstractmethod
    def write_staticmaps(self):
        """Write staticmaps at <root/?/> in model ready format"""
        # to write to gdal raster files use: self.staticmaps.raster.to_mapstack()
        # to write to netcdf use: self.staticmaps.to_netcdf()
        if not self._write:
            raise IOError("Model opened in read-only mode")
        raise NotImplementedError()

    @abstractmethod
    def read_staticgeoms(self):
        """Read staticgeoms at <root/?/> and parse to dict of geopandas"""
        if not self._write:
            # start fresh in read-only mode
            self._staticgeoms = dict()
        raise NotImplementedError()

    @abstractmethod
    def write_staticgeoms(self):
        """Write staticmaps at <root/?/> in model ready format"""
        # to write use self.staticgeoms[var].to_file()
        if not self._write:
            raise IOError("Model opened in read-only mode")
        raise NotImplementedError()

    @abstractmethod
    def read_forcing(self):
        """Read forcing at <root/?/> and parse to dict of xr.DataArray"""
        if not self._write:
            # start fresh in read-only mode
            self._forcing = dict()
        raise NotImplementedError()

    @abstractmethod
    def write_forcing(self):
        """write forcing at <root/?/> in model ready format"""
        if not self._write:
            raise IOError("Model opened in read-only mode")
        raise NotImplementedError()

    @abstractmethod
    def read_states(self):
        """Read states at <root/?/> and parse to dict of xr.DataArray"""
        if not self._write:
            # start fresh in read-only mode
            self._states = dict()
        raise NotImplementedError()

    @abstractmethod
    def write_states(self):
        """write states at <root/?/> in model ready format"""
        if not self._write:
            raise IOError("Model opened in read-only mode")
        raise NotImplementedError()

    @abstractmethod
    def read_results(self):
        """Read results at <root/?/> and parse to dict of xr.DataArray"""
        if not self._write:
            # start fresh in read-only mode
            self._results = dict()
        raise NotImplementedError()

    ## model configuration

    @property
    def config(self):
        """Returns parsed model configuration."""
        if not self._config:
            self.read_config()  # initialize default config
        return self._config

    def set_config(self, *args):
        """Update the config dictionary at key(s) with values.

        Parameters
        ----------
        args : key(s), value tuple, with minimal length of two
            keys can given by multiple args: ('key1', 'key2', 'value')
            or a string with '.' indicating a new level: ('key1.key2', 'value')

        Examples
        --------
        >> # self.config = {'a': 1, 'b': {'c': {'d': 2}}}

        >> set_config('a', 99)
        >> {'a': 99, 'b': {'c': {'d': 2}}}

        >> set_config('b', 'c', 'd', 99) # identical to set_config('b.d.e', 99)
        >> {'a': 1, 'b': {'c': {'d': 99}}}
        """
        if len(args) < 2:
            raise TypeError("set_config() requires a least one key and one value.")
        args = list(args)
        value = args.pop(-1)
        if len(args) == 1 and "." in args[0]:
            args = args[0].split(".") + args[1:]
        branch = self.config  # reads config at first call
        for key in args[:-1]:
            if not key in branch or not isinstance(branch[key], dict):
                branch[key] = {}
            branch = branch[key]
        branch[args[-1]] = value

    def setup_config(self, **cfdict):
        """Update config with a dictionary"""
        # TODO rename to update_config
        if len(cfdict) > 0:
            self.logger.debug(f"Setting model config options.")
        for key, value in cfdict.items():
            self.set_config(key, value)

    def get_config(self, *args, fallback=None, abs_path=False):
        """Get a config value at key(s).

        Parameters
        ----------
        args : tuple or string
            keys can given by multiple args: ('key1', 'key2')
            or a string with '.' indicating a new level: ('key1.key2')
        fallback: any, optional
            fallback value if key(s) not found in config, by default None.
        abs_path: bool, optional
            If True return the absolute path relative to the model root, by deafult False.
            NOTE: this assumes the config is located in model root!

        Returns
        value : any type
            dictionary value

        Examples
        --------
        >> # self.config = {'a': 1, 'b': {'c': {'d': 2}}}

        >> get_config('a')
        >> 1

        >> get_config('b', 'c', 'd') # identical to get_config('b.c.d')
        >> 2

        >> get_config('b.c') # # identical to get_config('b','c')
        >> {'d': 2}
        """
        args = list(args)
        if len(args) == 1 and "." in args[0]:
            args = args[0].split(".") + args[1:]
        branch = self.config  # reads config at first call
        for key in args[:-1]:
            branch = branch.get(key, {})
            if not isinstance(branch, dict):
                branch = dict()
                break
        value = branch.get(args[-1], fallback)
        if abs_path and isinstance(value, str):
            value = Path(value)
            if not value.is_absolute():
                value = Path(abspath(join(self.root, value)))
        return value

    ## model parameter maps, geometries and spatial properties

    @property
    def staticmaps(self):
        """xarray.Dataset representation of all static parameter maps"""
        if len(self._staticmaps) == 0:
            if self._read:
                self.read_staticmaps()
        return self._staticmaps

    def set_staticmaps(self, data, name=None):
        """Add data to staticmaps.

        All layers of staticmaps must have identical spatial coordinates.

        Parameters
        ----------
        data: xarray.DataArray or xarray.Dataset
            new map layer to add to staticmaps
        name: str, optional
            Name of new map layer, this is used to overwrite the name of a DataArray
            or to select a variable from a Dataset.
        """
        if name is None:
            if isinstance(data, xr.DataArray) and data.name is not None:
                name = data.name
            elif not isinstance(data, xr.Dataset):
                raise ValueError("Setting a map requires a name")
        elif name is not None and isinstance(data, xr.Dataset):
            data_vars = list(data.data_vars)
            if len(data_vars) == 1 and name not in data_vars:
                data = data.rename_vars({data_vars[0]: name})
            elif name not in data_vars:
                raise ValueError("Name not found in DataSet")
            else:
                data = data[[name]]
        if isinstance(data, xr.DataArray):
            data.name = name
            data = data.to_dataset()
        if len(self._staticmaps) == 0:  # new data
            self._staticmaps = data
        else:
            if isinstance(data, np.ndarray):
                if data.shape != self.shape:
                    raise ValueError("Shape of data and staticmaps do not match")
                data = xr.DataArray(dims=self.dims, data=data, name=name).to_dataset()
            for dvar in data.data_vars.keys():
                if dvar in self._staticmaps:
                    if self._read:
                        self.logger.warning(f"Replacing staticmap: {dvar}")
                self._staticmaps[dvar] = data[dvar]

    @property
    def staticgeoms(self):
        """geopandas.GeoDataFrame representation of all model geometries"""
        if not self._staticgeoms:
            if self._read:
                self.read_staticgeoms()
        return self._staticgeoms

    def set_staticgeoms(self, geom, name):
        """Add geom to staticmaps"""
        gtypes = [gpd.GeoDataFrame, gpd.GeoSeries]
        if not np.any([isinstance(geom, t) for t in gtypes]):
            raise ValueError(
                "First parameter map(s) should be geopandas.GeoDataFrame or geopandas.GeoSeries"
            )
        if name in self._staticgeoms:
            if self._read:
                self.logger.warning(f"Replacing staticgeom: {name}")
        self._staticgeoms[name] = geom

    @property
    def forcing(self):
        """dict of xarray.dataarray representation of all forcing"""
        if not self._forcing:
            if self._read:
                self.read_forcing()
        return self._forcing

    def set_forcing(self, data, name=None):
        """Add data to forcing attribute which is a dictionary of xarray.DataArray.
        The dictionary key is taken from the variable name. In case of a DataArray
        without name, the name can be passed using the optional name argument.

        Arguments
        ---------
        data: xarray.Dataset or xarray.DataArray
            New forcing data to add
        name: str, optional
            Variable name, only in case data is of type DataArray
        """
        # check dataset dtype
        dtypes = [xr.DataArray, xr.Dataset]
        if not np.any([isinstance(data, t) for t in dtypes]):
            raise ValueError("Data type not recognized")
        if isinstance(data, xr.DataArray):
            # NOTE name can be different from data.name !
            if data.name is None and name is not None:
                data.name = name
            elif name is None and data.name is not None:
                name = data.name
            elif data.name is None and name is None:
                raise ValueError("Name required for forcing DataArray.")
            data = {name: data}
        for name in data:
            if name in self._forcing:
                self.logger.warning(f"Replacing forcing: {name}")
            self._forcing[name] = data[name]

    @property
    def states(self):
        """dict xarray.dataarray representation of all states"""
        if not self._states:
            if self._read:
                self.read_states()
        return self._states

    def set_states(self, data, name=None):
        """Add data to states attribute which is a dictionary of xarray.DataArray.
        The dictionary key is taken from the variable name. In case of a DataArray
        without name, the name can be passed using the optional name argument.

        Arguments
        ---------
        data: xarray.Dataset or xarray.DataArray
            New forcing data to add
        name: str, optional
            Variable name, only in case data is of type DataArray
        """
        # check dataset dtype
        dtypes = [xr.DataArray, xr.Dataset]
        if not np.any([isinstance(data, t) for t in dtypes]):
            raise ValueError("Data type not recognized")
        if isinstance(data, xr.DataArray):
            # NOTE name can be different from data.name !
            if data.name is None and name is not None:
                data.name = name
            elif name is None and data.name is not None:
                name = data.name
            elif data.name is None and name is None:
                raise ValueError("Name required for forcing DataArray.")
            data = {name: data}
        for name in data:
            if name in self._states:
                self.logger.warning(f"Replacing state: {name}")
            self._states[name] = data[name]

    @property
    def results(self):
        """dict xarray.dataarray representation of model results"""
        if not self._results:
            if self._read:
                self.read_results()
        return self._results

    def set_results(self, data, name=None, split_dataset=False):
        """Add data to results attribute which is a dictionary of xarray.DataArray and/or xarray.Dataset.

        The dictionary key is taken from the variable name. In case of a DataArray
        without name, the name can be passed using the optional name argument. In case of
        a Dataset, the dictionnary key is passed using the name argument.

        Dataset can either be added as is to the dictionnary (default) or split into several
        DataArrays using the split_dataset argument.

        Arguments
        ---------
        data: xarray.Dataset or xarray.DataArray
            New forcing data to add
        name: str, optional
            Variable name, only in case data is of type DataArray or if a Dataset is added as is (split_dataset=False).
        split_dataset: bool, optional
            If data is a xarray.Dataset, either add it as is to results or split it into several xarray.DataArrays.
        """
        # check data dtype
        dtypes = [xr.DataArray, xr.Dataset]
        if not np.any([isinstance(data, t) for t in dtypes]):
            raise ValueError("Data type not recognized")
        if isinstance(data, xr.DataArray):
            # NOTE name can be different from data.name !
            if data.name is None and name is not None:
                data.name = name
            elif name is None and data.name is not None:
                name = data.name
            elif data.name is None and name is None:
                raise ValueError("Name required for result DataArray.")
            data = {name: data}
        # Add to results
        if isinstance(data, xr.Dataset) and not split_dataset:
            if name is not None:
                if name in self._results:
                    self.logger.warning(f"Replacing result: {name}")
                self._results[name] = data
            else:
                raise ValueError("Name required to add DataSet directly to results")
        else:
            for name in data:
                if name in self._results:
                    self.logger.warning(f"Replacing result: {name}")
                self._results[name] = data[name]

    ## properties / methods below can be used directly in actual class

    @property
    def crs(self):
        """Returns coordinate reference system embedded in staticmaps."""
        return self.staticmaps.raster.crs

    def set_crs(self, crs):
        """Embed coordinate reference system staticmaps metadata."""
        return self.staticmaps.raster.set_crs(crs)

    @property
    def dims(self):
        """Returns spatial dimension names of staticmaps."""
        return self.staticmaps.raster.dims

    @property
    def coords(self):
        """Returns coordinates of staticmaps."""
        return self.staticmaps.raster.coords

    @property
    def res(self):
        """Returns coordinates of staticmaps."""
        return self.staticmaps.raster.res

    @property
    def transform(self):
        """Returns spatial transform staticmaps."""
        return self.staticmaps.raster.transform

    @property
    def width(self):
        """Returns width of staticmaps."""
        return self.staticmaps.raster.width

    @property
    def height(self):
        """Returns height of staticmaps."""
        return self.staticmaps.raster.height

    @property
    def shape(self):
        """Returns shape of staticmaps."""
        return self.staticmaps.raster.shape

    @property
    def bounds(self):
        """Returns shape of staticmaps."""
        return self.staticmaps.raster.bounds

    @property
    def region(self):
        """Returns geometry of region of the model area of interest."""
        region = None
        if "region" in self.staticgeoms:
            region = self.staticgeoms["region"]
        elif len(self.staticmaps) > 0:
            crs = self.crs
            if crs is None and crs.to_epsg() is not None:
                crs = crs.to_epsg()  # not all CRS have an EPSG code
            region = gpd.GeoDataFrame(geometry=[box(*self.bounds)], crs=crs)
        return region

    def test_model_api(self):
        """Test compliance to model API instances.

        Returns
        -------
        non_compliant: list
            List of objects that are non-compliant with the model API structure.
        """
        non_compliant = []
        # Staticmaps
        if not isinstance(self.staticmaps, xr.Dataset):
            non_compliant.append("staticmaps")
        # Staticgeoms
        if not isinstance(self.staticgeoms, dict):
            non_compliant.append("staticgeoms")
        elif self.staticgeoms:  # non-empty dict
            for name, geom in self.staticgeoms.items():
                if not isinstance(geom, gpd.GeoDataFrame):
                    non_compliant.append(f"staticgeoms.{name}")
        # Forcing
        if not isinstance(self.forcing, dict):
            non_compliant.append("forcing")
        elif self.forcing:  # non-empty dict
            for name, data in self.forcing.items():
                if not isinstance(data, xr.DataArray):
                    non_compliant.append(f"forcing.{name}")
        # Config
        if not isinstance(self.config, dict):
            non_compliant.append("config")
        # States
        if not isinstance(self.states, dict):
            non_compliant.append("states")
        elif self.states:  # non-empty dict
            for name, data in self.states.items():
                if not isinstance(data, xr.DataArray):
                    non_compliant.append(f"states.{name}")
        # Results
        if not isinstance(self.results, dict):
            non_compliant.append("results")
        elif self.results:  # non-empty dict
            dtypes = [xr.DataArray, xr.Dataset]
            for name, data in self.results.items():
                if not np.any([isinstance(data, t) for t in dtypes]):
                    non_compliant.append(f"results.{name}")

        return non_compliant
