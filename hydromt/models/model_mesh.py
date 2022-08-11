from typing import Union, Optional, List
import logging
from os.path import join, isfile
import xarray as xr
import xugrid as xu
import geopandas as gpd
from shapely.geometry import box

from .model_api import Model, _check_data

__all__ = ["MeshModel", "MeshMixin"]
logger = logging.getLogger(__name__)


class MeshMixin(object):
    # placeholders
    # xr.Dataset if empty, else xu.UgridDataset representation of all static mesh variables at the same resolution and bounds
    _mesh = xr.Dataset()

    def read_mesh(self):
        """Read mesh at <root/?/> and parse to xugrid Dataset"""
        if not self._write:
            # start fresh in read-only mode
            self._mesh = xr.Dataset()
        if isfile(
            join(self.root, "mesh", "mesh.nc")
        ):  # Change of file not implemented yet
            self._mesh = xu.open_dataset(join(self.root, "mesh", "mesh.nc"))

    def write_mesh(self):
        """Write grid at <root/?/> in xugrid UgridDataset"""
        if not self._write:
            raise IOError("Model opened in read-only mode")
        elif not self._mesh:
            self.logger.warning("No mesh to write - Exiting")
            return
        # filename
        fn_default = join(self.root, "mesh", "mesh.nc")
        self.logger.info(f"Write mesh to {self.root}")
        ds_out = self.mesh
        # ds_new = xu.UgridDataset(grid=ds_out.ugrid.grid) # bug in xugrid?
        ds_out.ugrid.to_netcdf(fn_default)

    def set_mesh(
        self,
        data: Union[xu.UgridDataArray, xu.UgridDataset],
        name: Optional[str] = None,
    ):
        """Add data to mesh object.

        All layers of mesh have identical spatial coordinates in Ugrid conventions.

        Parameters
        ----------
        data: xugrid.UgridDataArray or xugrid.UgridDataset
            new layer to add to mesh
        name: str, optional
            Name of new object layer, this is used to overwrite the name of a UgridDataArray.
        """
        if not isinstance(data, xu.UgridDataArray) and not isinstance(
            data, xu.UgridDataset
        ):
            raise ValueError(
                "New mesh data in set_mesh should be of type xu.UgridDataArray or xu.UgridDataset"
            )
        if isinstance(data, xu.UgridDataArray):
            if name is not None:
                data.name = data
            elif data.name is None:
                raise ValueError(
                    f"Cannot set mesh from a data of type {str(type(data).__name__)}"
                )
            data = data.to_dataset()
        for dvar in data.data_vars:
            if dvar in self._mesh:
                if self._read:
                    self.logger.warning(f"Replacing mesh parameter: {dvar}")
                self._mesh[dvar] = data[dvar]


class MeshModel(Model, MeshMixin):
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

    def read(self):
        """Method to read the complete model schematization and configuration from file."""
        super().read()
        self.read_mesh()

    def write(self):
        """Method to write the complete model schematization and configuration to file."""
        super().write()
        self.write_mesh()

    # MeshModel specific methods

    # MeshModel properties
    @property
    def mesh(self):
        """xarray.Dataset representation of all mesh parameters"""
        # XU grid data type Xarray dataset with xu sampling.
        if len(self._mesh) == 0:
            if self._read:
                self.read_mesh()
        return self._mesh

    @property
    def bounds(self) -> tuple:
        """Returns model bounds."""
        return self.mesh.ugrid.grid.bounds

    @property
    def region(self) -> gpd.GeoDataFrame:
        """Returns geometry of region of the model area of interest."""
        region = gpd.GeoDataFrame()
        if "region" in self.geoms:
            region = self.geoms["region"]
        elif len(self.mesh) > 0:
            crs = self.mesh.ugrid.crs
            if crs is None and crs.to_epsg() is not None:
                crs = crs.to_epsg()  # not all CRS have an EPSG code
            region = gpd.GeoDataFrame(geometry=[box(*self.bounds)], crs=crs)
        return region

    def _test_model_api(self) -> List:
        """Test compliance with HYdroMT MeshModel API.

        Returns
        -------
        non_compliant: list
            List of objects that are non-compliant with the model API structure.
        """
        non_compliant = super()._test_model_api()
        # Mesh instance
        if len(self.mesh) == 0 and not isinstance(self.mesh, xr.Dataset):
            non_compliant.append("mesh")
        if len(self.mesh) > 0 and not isinstance(self.mesh, xu.UgridDataset):
            non_compliant.append("mesh")

        return non_compliant
