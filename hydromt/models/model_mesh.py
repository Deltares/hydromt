import pytest
import sys, os
from os.path import join, isfile
from .model_api import Model
import xarray as xr
import xugrid as xu
import numpy as np
import geopandas as gpd
from shapely.geometry import box

from typing import Tuple, Union, Optional, List

import logging
import os

__all__ = ["MeshModel"]
logger = logging.getLogger(__name__)


class MeshModel(Model):
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

        # placeholders
        self._mesh = None  # xu.Dataset() does not work for now

    def read(self):
        """Method to read the complete model schematization and configuration from file."""
        super().read()
        self.read_mesh()
        # Other specifics to MeshModel...

    def write(self):
        """Method to write the complete model schematization and configuration to file."""
        super().write()
        self.write_mesh()
        # Other specifics to MeshModel...

    @property
    def mesh(self):
        """xarray.Dataset representation of all mesh parameters"""
        # XU grid data type Xarray dataset with xu sampling.
        if self._mesh is None:
            if self._read:
                self.read_mesh()
        return self._mesh

    def read_mesh(self):
        """Read mesh at <root/?/> and parse to xugrid Dataset"""
        if not self._write:
            # start fresh in read-only mode
            self._mesh = xu.Dataset()
        if isfile(
            join(self.root, "mesh", "mesh.nc")
        ):  # Change of file not implemented yet
            self._mesh = xu.open_dataset(join(self.root, "mesh", "mesh.nc"))

    def write_mesh(self):
        """Write grid at <root/?/> in xugrid Dataset"""
        if not self._write:
            raise IOError("Model opened in read-only mode")
        elif not self._mesh:
            self.logger.warning("No mesh to write - Exiting")
            return
        # filename
        fn_default = join(self.root, "mesh", "mesh.nc")
        self.logger.info(f"Write mesh to {self.root}")
        ds_out = self.mesh
        ds_new = xu.UgridDataset(grid=ds_out.ugrid.grid)
        ds_new.to_netcdf(fn_default)

    def set_mesh(
        self,
        data: Union[xu.UgridDataArray, xu.UgridDataset],
        name: Optional[str] = None,
    ):
        """Add data to mesh object.

        -Describe specifics of the mesh object (for example related to the network and ugrid representation)

        Parameters
        ----------
        data: xugrid.UgridDataArray or xugrid.UgridDataset
            new layer to add to mesh
        name: str, optional
            Name of new object layer, this is used to overwrite the name of a DataArray
            or to select a variable from a Dataset.
        """
        if name is None:
            if isinstance(data, xu.UgridDataArray) and data.name is not None:
                name = data.name
            elif not isinstance(data, xu.UgridDataset):
                raise ValueError("Setting a mesh parameter requires a name")
        elif name is not None and isinstance(data, xu.UgridDataset):
            data_vars = list(data.data_vars)
            if len(data_vars) == 1 and name not in data_vars:
                data = data.rename_vars({data_vars[0]: name})
            elif name not in data_vars:
                raise ValueError("Name not found in DataSet")
            else:
                data = data[[name]]
        if isinstance(data, xu.UgridDataArray):
            data.name = name
            data = data.to_dataset()
        if self._mesh is None:  # new data
            self._mesh = data
        else:
            for dvar in data.data_vars.keys():
                if dvar in self._mesh:
                    if self._read:
                        self.logger.warning(f"Replacing mesh parameter: {dvar}")
                self._mesh[dvar] = data[dvar]

    # Possible other properties: related to the network and ugrid representation
    # ....

    def test_subclass_mesh(self):
        """Test compliance to model Mesh instances.

        Returns
        -------
        non_compliant: list
            List of objects that are non-compliant with the model API structure.
        """
        non_compliant = []
        # Mesh instance
        if self.mesh is not None:
            if not isinstance(self.mesh, xu.UgridDataset):
                non_compliant.append("mesh")

        return non_compliant
