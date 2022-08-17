from typing import Union, Optional, List, Tuple
import logging
import os
from os.path import join, isdir, dirname, isfile
import xarray as xr
import xugrid as xu
import geopandas as gpd
from shapely.geometry import box

from ..raster import GEO_MAP_COORD
from .model_api import Model

__all__ = ["MeshModel"]
logger = logging.getLogger(__name__)


class MeshMixin(object):
    # placeholders
    # We cannot initialize an empty xu.UgridDataArray
    _API = {
        "mesh": Union[xu.UgridDataArray, xu.UgridDataset],
    }

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._mesh = None

    @property
    def mesh(self) -> Union[xu.UgridDataArray, xu.UgridDataset]:
        """Model mesh data. Returns a xarray.Dataset."""
        # XU grid data type Xarray dataset with xu sampling.
        if self._mesh is None:
            if self._read:
                self.read_mesh()
        return self._mesh

    def set_mesh(
        self,
        data: Union[xu.UgridDataArray, xu.UgridDataset],
        name: Optional[str] = None,
    ) -> None:
        """Add data to mesh.

        All layers of mesh have identical spatial coordinates in Ugrid conventions.

        Parameters
        ----------
        data: xugrid.UgridDataArray or xugrid.UgridDataset
            new layer to add to mesh
        name: str, optional
            Name of new object layer, this is used to overwrite the name of a UgridDataArray.
        """
        if not isinstance(data, (xu.UgridDataArray, xu.UgridDataset)):
            raise ValueError(
                "New mesh data in set_mesh should be of type xu.UgridDataArray or xu.UgridDataset"
            )
        if isinstance(data, xu.UgridDataArray):
            if name is not None:
                data.name = data
            elif data.name is None:
                raise ValueError(
                    f"Cannot set mesh from {str(type(data).__name__)} without a name."
                )
            data = data.to_dataset()
        if self._mesh is None:  # NOTE: mesh is initialized with None
            self._mesh = data
        else:
            for dvar in data.data_vars:
                if dvar in self._mesh:
                    self.logger.warning(f"Replacing mesh parameter: {dvar}")
                self._mesh[dvar] = data[dvar]

    def read_mesh(self, fn: str = "mesh/mesh.nc", **kwargs) -> None:
        """Read model mesh data at <root>/<fn> and add to mesh property

        key-word arguments are passed to :py:func:`xr.open_dataset`

        Parameters
        ----------
        fn : str, optional
            filename relative to model root, by default 'mesh/mesh.nc'
        """
        for ds in self._read_nc(fn, **kwargs).values():
            uds = xu.UgridDataset(ds)
            if ds.rio.crs is not None:  # parse crs
                uds.ugrid.grid.set_crs(ds.rio.crs)
                uds = uds.drop_vars(GEO_MAP_COORD, errors="ignore")
            self.set_mesh(uds)

    def write_mesh(self, fn: str = "mesh/mesh.nc", **kwargs) -> None:
        """Write model grid data to netcdf file at <root>/<fn>

        key-word arguments are passed to :py:meth:`xarray.Dataset.ugrid.to_netcdf`

        Parameters
        ----------
        fn : str, optional
            filename relative to model root, by default 'grid/grid.nc'
        """
        if not self._write:
            raise IOError("Model opened in read-only mode")
        elif self._mesh is None:
            self.logger.warning("No mesh to write - Exiting")
            return
        # filename
        _fn = join(self.root, fn)
        if not isdir(dirname(_fn)):
            os.makedirs(dirname(_fn))
        self.logger.debug(f"Writing file {fn}")
        # ds_new = xu.UgridDataset(grid=ds_out.ugrid.grid) # bug in xugrid?
        ds_out = self.mesh.ugrid.to_dataset()
        if self.mesh.ugrid.crs is not None:
            # save crs to spatial_ref coordinate
            ds_out = ds_out.rio.write_crs(self.mesh.ugrid.crs)
        ds_out.to_netcdf(_fn, **kwargs)


class MeshModel(MeshMixin, Model):

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
            "mesh",
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
            By default ['config', 'maps', 'mesh', 'geoms', 'forcing', 'states', 'results']
        """
        super().read(components=components)

    def write(
        self,
        components: List = ["config", "mesh", "geoms", "forcing", "states"],
    ) -> None:
        """Write the complete model schematization and configuration to model files.

        Parameters
        ----------
        components : List, optional
            List of model components to write, each should have an associated write_<component> method.
            By default ['config', 'maps', 'mesh', 'geoms', 'forcing', 'states']
        """
        super().write(components=components)

    # MeshModel specific methods

    # MeshModel properties
    @property
    def bounds(self) -> Tuple:
        """Returns model mesh bounds."""
        if self._mesh is not None:
            return self._mesh.ugrid.grid.bounds

    @property
    def region(self) -> gpd.GeoDataFrame:
        """Returns geometry of region of the model area of interest."""
        region = gpd.GeoDataFrame()
        if "region" in self.geoms:
            region = self.geoms["region"]
        elif self.mesh is not None:
            crs = self.mesh.ugrid.crs
            if crs is None and hasattr(crs, "to_epsg"):
                crs = crs.to_epsg()  # not all CRS have an EPSG code
            region = gpd.GeoDataFrame(geometry=[box(*self.bounds)], crs=crs)
        return region
