# -*- coding: utf-8 -*-
"""HydroMT LumpedModel class definition"""

import  os
from os.path import join, isfile, isdir, dirname
import xarray as xr
import numpy as np
import geopandas as gpd
from typing import Union, Optional, List, Dict
from shapely.geometry import box

import logging
from .. import workflows, flw
from .model_api import Model

__all__ = ["LumpedModel"]
logger = logging.getLogger(__name__)
    
class LumpedMixin:
    _API = {
        "response_units": xr.Dataset,
    }

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._response_units = xr.Dataset()

    @property
    def response_units(self) -> xr.Dataset:
        """Model response unit (lumped) data. Returns xr.Dataset geometry coordinate."""
        if not self._response_units:
            if self._read:
                self.read_response_units()
        return self._response_units

    def set_response_units(
        self,
        data: Union[xr.DataArray, xr.Dataset, np.ndarray],
        name: Optional[str] = None,
    ) -> None:
        """Add data to response_units.

        All layers of response_units must have identical spatial index.

        Parameters
        ----------
        data: xarray.DataArray or xarray.Dataset
            new data to add to response_units
        name: str, optional
            Name of new data, this is used to overwrite the name of a DataArray
            or to select a variable from a Dataset.
        """
        name_required = isinstance(data, np.ndarray) or (
            isinstance(data, xr.DataArray) and data.name is None
        )
        if name is None and name_required:
            raise ValueError(f"Unable to set {type(data).__name__} data without a name")
        if isinstance(data, np.ndarray) and "geometry" in self._response_units:
            # TODO: index name is hard coded. Using GeoDataset.index property once ready
            index = self._response_units["index"]
            if data.size != index.size and data.ndim == 1:
                raise ValueError(
                    "Size of data and number of response_units do not match"
                )
            data = xr.DataArray(dims=["index"], data=data)
        if isinstance(data, xr.DataArray):
            if name is not None:  # rename
                data.name = name
            data = data.to_dataset()
        elif not isinstance(data, xr.Dataset):
            raise ValueError(f"cannot set data of type {type(data).__name__}")
        
        if np.all(
            len(self._response_units) == 0 and "geometry" in data.coords
        ):  # new data with a geometry
            self._response_units = data
        else:
            for dvar in data.data_vars:
                if dvar in self._response_units:
                    self.logger.warning(f"Replacing response_units variable: {dvar}")
                # TODO: check on index coordinate before merging
                self._response_units[dvar] = data[dvar]

    def read_response_units(
        self,
        fn: str = "response_units/response_units.nc",
        fn_geom: str = "response_units/response_units.geojson",
        **kwargs,
    ) -> None:
        """Read model response units from combined netcdf file at <root>/<fn> and geojson file at <root>/<fn_geom>.
        The netcdf file contains the attribute data and the geojson file the geometry vector data.

        key-word arguments are passed to :py:func:`xarray.open_dataset`

        Parameters
        ----------
        fn : str, optional
            netcdf filename relative to model root, by default 'response_units/response_units.nc'
        fn_geom : str, optional
            geojson filename relative to model root, by default 'response_units/response_units.geojson'
        """
        ds = xr.merge(self._read_nc(fn, **kwargs).values())
        if isfile(join(self.root, fn_geom)):
            gdf = gpd.read_file(join(self.root, fn_geom))
            # TODO: index name is hard coded. Using GeoDataset.index property once ready
            ds = ds.assign_coords(geometry=(["index"], gdf["geometry"]))
            if gdf.crs is not None:  # parse crs
                ds = ds.rio.write_crs(gdf.crs)
        self.set_response_units(ds)

    def write_response_units(
        self,
        fn: str = "response_units/response_units.nc",
        fn_geom: str = "response_units/response_units_{}.geojson",
        **kwargs,
    ):
        """Write model response units to combined netcdf file at <root>/<fn> and geojson file at <root>/<fn_geom>.
        The netcdf file contains the attribute data and the geojson file the geometry vector data.

        key-word arguments are passed to :py:meth:`xarray.Dataset.to_netcdf`

        Parameters
        ----------
        fn : str, optional
            netcdf filename relative to model root, by default 'response_units/response_units.nc'
        fn_geom : str, optional
            geojson filename relative to model root, by default 'response_units/response_units.geojson'
        """
        nc_dict = dict()
        if len(self._response_units) > 0:
            # write geometries
            ds = self._response_units
            # get list of coordinates that hold geometries
            corl = list(ds.coords)
            geom_coords = [v for v in corl if 'geometry' in v]
            
            for g in geom_coords:
                gdf = gpd.GeoDataFrame(geometry=ds[g].values, crs=ds.rio.crs)
                fn_geom_g = fn_geom.format(g)
                if not isdir(dirname(join(self.root, fn_geom_g))):
                    os.makedirs(dirname(join(self.root, fn_geom_g)))
                gdf.to_file(join(self.root, fn_geom_g), driver="GeoJSON")
                
            # _write_nc requires dict - use dummy key
            nc_dict.update({"response_units": ds.drop_vars(geom_coords)})
        self._write_nc(nc_dict, fn, **kwargs)


class LumpedModel(LumpedMixin, Model):

    _CLI_ARGS = {"region": "setup_region"}

    def __init__(
        self,
        root: str = None,
        mode: str = "w",
        config_fn: str = None,
        data_libs: List[str] = None,
        logger=logger,
    ):
        # Initialize with the Model class
        super().__init__(
            root=root,
            mode=mode,
            config_fn=config_fn,
            data_libs=data_libs,
            logger=logger,
        )
    
    def read(
        self,
        components: List = [
            "config",
            "response_units",
            "geoms",
            "forcing",
            "states",
            "results",
        ],
    ) -> None:
        """Read the complete model schematization and configuration from model files.
        Parameters
        ----------
        components : List, optional
            List of model components to read, each should have an associated read_<component> method.
            By default ['config', 'maps', 'response_units', 'geoms', 'forcing', 'states', 'results']
        """
        super().read(components=components)

    def write(
        self,
        components: List = [
            "config",
            "response_units",
            "geoms",
            "forcing",
            "states",
        ],
    ) -> None:
        """Write the complete model schematization and configuration to model files.
        Parameters
        ----------
        components : List, optional
            List of model components to write, each should have an associated write_<component> method.
            By default ['config', 'maps', 'response_units', 'geoms', 'forcing', 'states']
        """
        super().write(components=components)

    @property
    def region(self) -> gpd.GeoDataFrame:
        """Returns the geometry of the model area of interest."""
        region = gpd.GeoDataFrame()
        if "region" in self.geoms:
            region = self.geoms["region"]
        elif len(self.response_units) > 0:
            ds = self.response_units
            gdf = gpd.GeoDataFrame(geometry=ds["geometry"].values, crs=ds.rio.crs)
            region = gpd.GeoDataFrame(geometry=[box(*gdf.total_bounds)], crs=gdf.crs)
        return region
    
    def setup_response_unit(
        self,
        split_regions = False,
        index_col = "index",
        split_method = "us_area",
        hydrography_fn = "merit_hydro",
        **kwargs
        ):
        """
        This component sets up the region and response units.
        
        """

        if len(self.region) == 0:
            raise ValueError("No region defined. Define region first with setup_region.")
        
        if not split_regions:
            ds_response_units = workflows.gpd_to_response_unit(self.region,index_col=index_col)
            self.logger.info(f"setup_response_unit.split_regions set to False. response_units set up from region geometries")
        
        if split_regions:
            self.logger.info(f"setup_response_unit.split_regions set to True. Deriving response_units based on hydrography")
            
            ds = self.data_catalog.get_rasterdataset(hydrography_fn, geom = self.region)
            ds_response_units = workflows.hydrography_to_basins(
                ds,
                self.region,
                split_method,
                **kwargs
            )
        self.set_response_units(ds_response_units)
    
    def setup_downstream_links(
        self,
        hydrography_fn="merit_hydro",
        ):
        """Link basins from upstream to downstream based on flow direction map

        Args:
            outflow_gpd (gpd.GeoDataFrame): point geometries of the subbasin outlets
            hydrography_fn (str, optional): Hydrography dataset, must include flwdir variable.
            Defaults to "merit_hydro".

        Returns:
            np.array: array of downstream link ids
        """
        outflow_gpd = workflows.ru_geometry_to_gpd(self.response_units,geometry_name='outlet_geometry')
        basins_gpd =  workflows.ru_geometry_to_gpd(self.response_units)       
        ds = self.data_catalog.get_rasterdataset(hydrography_fn, geom=self.region)
        flwdir = flw.flwdir_from_da(ds["flwdir"], ftype="d8")#, mask= workflows.make_ds_mask(ds, basins_gpd, col_name='value'))
        rasterized_map = ds.raster.rasterize(basins_gpd, col_name='value')

        # TODO: make one function in flw.py
        dwn_basin = flwdir.downstream(rasterized_map.values.flatten()).reshape(ds.raster.shape)
        da_out = xr.DataArray(
                data=dwn_basin,
                coords=ds.raster.coords,
                dims=ds.raster.dims,
            )
        da_out.raster.set_nodata(0)
        da_out.raster.set_crs(ds.raster.crs)
        basins_downstream = da_out.raster.sample(outflow_gpd)
        basins_downstream = basins_downstream.drop(['x','y','spatial_ref'])
        self.set_response_units(basins_downstream, name='down_id')