"""Implementations for model mesh workloads."""
import logging
import os
from os.path import dirname, isdir, join
from pathlib import Path
from typing import Dict, List, Optional, Union

import geopandas as gpd
import numpy as np
import pandas as pd
import xarray as xr
import xugrid as xu
from pyproj import CRS
from shapely.geometry import box

from .. import workflows
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

    ## general setup methods
    def setup_mesh2d_from_rasterdataset(
        self,
        raster_fn: Union[str, Path, xr.DataArray, xr.Dataset],
        grid_name: Optional[str] = "mesh2d",
        variables: Optional[list] = None,
        fill_method: Optional[str] = None,
        resampling_method: Optional[str] = "mean",
        all_touched: Optional[bool] = True,
        rename: Optional[Dict] = dict(),
    ) -> List[str]:
        """HYDROMT CORE METHOD: Add data variable(s) from ``raster_fn`` to 2D ``grid_name`` in mesh object.

        Raster data is interpolated to the mesh ``grid_name`` using the ``resampling_method``.
        If raster is a dataset, all variables will be added unless ``variables`` list
        is specified.

        Adds model layers:

        * **raster.name** mesh: data from raster_fn

        Parameters
        ----------
        raster_fn: str, Path, xr.DataArray, xr.Dataset
            Data catalog key, path to raster file or raster xarray data object.
        grid_name: str, optional
            Name of the mesh grid to add the data to. By default 'mesh2d'.
        variables: list, optional
            List of variables to add to mesh from raster_fn. By default all.
        fill_method : str, optional
            If specified, fills no data values using fill_nodata method.
            Available methods are {'linear', 'nearest', 'cubic', 'rio_idw'}.
        resampling_method: str, optional
            Method to sample from raster data to mesh. By default mean. Options include
            {'count', 'min', 'max', 'sum', 'mean', 'std', 'median', 'q##'}.
        all_touched : bool, optional
            If True, all pixels touched by geometries will used to define the sample.
            If False, only pixels whose center is within the geometry or that are
            selected by Bresenham's line algorithm will be used. By default True.
        rename: dict, optional
            Dictionary to rename variable names in raster_fn before adding to mesh
            {'name_in_raster_fn': 'name_in_mesh'}. By default empty.

        Returns
        -------
        list
            List of variables added to mesh.
        """  # noqa: E501
        self.logger.info(f"Preparing mesh data from raster source {raster_fn}")
        # Check if grid name in self.mesh
        if grid_name not in self.mesh_names:
            raise ValueError(f"Grid name {grid_name} not in mesh ({self.mesh_names}).")
        # Read raster data and select variables
        ds = self.data_catalog.get_rasterdataset(
            raster_fn, bbox=self.bounds[grid_name], buffer=2, variables=variables
        )
        if isinstance(ds, xr.DataArray):
            ds = ds.to_dataset()

        if fill_method is not None:
            ds = ds.raster.interpolate_na(method=fill_method)

        # Convert mesh grid as geodataframe for sampling
        # Reprojection happens to gdf inside of zonal_stats method
        ds_sample = ds.raster.zonal_stats(
            gdf=self.mesh_gdf[grid_name],
            stats=resampling_method,
            all_touched=all_touched,
        )
        # Rename variables
        rm_dict = {f"{var}_{resampling_method}": var for var in ds.data_vars}
        ds_sample = ds_sample.rename(rm_dict).rename(rename)
        # Convert to UgridDataset
        uds_sample = xu.UgridDataset(ds_sample, grids=self.mesh_grids[grid_name])

        self.set_mesh(uds_sample, grid_name=grid_name, overwrite_grid=False)

        return list(ds_sample.data_vars.keys())

    def setup_mesh2d_from_raster_reclass(
        self,
        raster_fn: Union[str, Path, xr.DataArray],
        reclass_table_fn: Union[str, Path, pd.DataFrame],
        reclass_variables: list,
        grid_name: Optional[str] = "mesh2d",
        variable: Optional[str] = None,
        fill_nodata: Optional[str] = None,
        resampling_method: Optional[Union[str, list]] = "mean",
        all_touched: Optional[bool] = True,
        rename: Optional[Dict] = dict(),
        **kwargs,
    ) -> List[str]:
        """HYDROMT CORE METHOD: Add data variable(s) to 2D ``grid_name`` in mesh object by reclassifying the data in ``raster_fn`` based on ``reclass_table_fn``.

        The reclassified raster data
        are subsequently interpolated to the mesh using `resampling_method`.

        Adds model layers:

        * **reclass_variables** mesh: reclassified raster data interpolated to the
            model mesh

        Parameters
        ----------
        raster_fn : str, Path, xr.DataArray
            Data catalog key, path to the raster file, or raster xarray data object.
            Should be a DataArray. If not, use the `variable` argument for selection.
        reclass_table_fn : str, Path, pd.DataFrame
            Data catalog key, path to the tabular data file, or tabular pandas dataframe
            object for the reclassification table of `raster_fn`.
        reclass_variables : list
            List of reclass_variables from the reclass_table_fn table to add to the
            mesh. The index column should match values in raster_fn.
        grid_name : str, optional
            Name of the mesh grid to add the data to. By default 'mesh2d'.
        variable : str, optional
            Name of the raster dataset variable to use. This is only required when
            reading datasets with multiple variables. By default, None.
        fill_nodata : str, optional
            If specified, fills nodata values in `raster_fn` using the `fill_nodata`
            method before reclassifying. Available methods are
            {'linear', 'nearest', 'cubic', 'rio_idw'}.
        resampling_method : str or list, optional
            Method to sample from raster data to the mesh. Can be a list per variable
            in `reclass_variables` or a single method for all. By default, 'mean' is
            used for all `reclass_variables`. Options include {'count', 'min', 'max',
            'sum', 'mean', 'std', 'median', 'q##'}.
        all_touched : bool, optional
            If True, all pixels touched by geometries will be used to define the sample.
            If False, only pixels whose center is within the geometry or that are
            selected by Bresenham's line algorithm will be used. By default, True.
        rename : dict, optional
            Dictionary to rename variable names in `reclass_variables` before adding
            them to the mesh. The dictionary should have the form
            {'name_in_reclass_table': 'name_in_mesh'}. By default, an empty dictionary.
        **kwargs : dict
            Additional keyword arguments to be passed to the raster dataset
            retrieval method.

        Returns
        -------
        variable_names : List[str]
            List of added variable names in the mesh.

        Raises
        ------
        ValueError
            If `raster_fn` is not a single variable raster.
        """  # noqa: E501
        self.logger.info(
            f"Preparing mesh data by reclassifying the data in {raster_fn} "
            f"based on {reclass_table_fn}."
        )
        # Check if grid name in self.mesh
        if grid_name not in self.mesh_names:
            raise ValueError(f"Grid name {grid_name} not in mesh ({self.mesh_names}).")
        # Read raster data and mapping table
        da = self.data_catalog.get_rasterdataset(
            raster_fn,
            bbox=self.bounds[grid_name],
            buffer=2,
            variables=variable,
            **kwargs,
        )
        if not isinstance(da, xr.DataArray):
            raise ValueError(
                f"raster_fn {raster_fn} should be a single variable raster. "
                "Please select one using the 'variable' argument"
            )
        df_vars = self.data_catalog.get_dataframe(
            reclass_table_fn, variables=reclass_variables
        )

        if fill_nodata is not None:
            da = da.raster.interpolate_na(method=fill_nodata)

        # Mapping function
        ds_vars = da.raster.reclassify(reclass_table=df_vars, method="exact")

        # Convert mesh grid as geodataframe for sampling
        # Reprojection happens to gdf inside of zonal_stats method
        ds_sample = ds_vars.raster.zonal_stats(
            gdf=self.mesh_gdf[grid_name],
            stats=np.unique(np.atleast_1d(resampling_method)),
            all_touched=all_touched,
        )
        # Rename variables
        if isinstance(resampling_method, str):
            resampling_method = np.repeat(resampling_method, len(reclass_variables))
        rm_dict = {
            f"{var}_{mtd}": var
            for var, mtd in zip(reclass_variables, resampling_method)
        }
        ds_sample = ds_sample.rename(rm_dict).rename(rename)
        ds_sample = ds_sample[reclass_variables]
        # Convert to UgridDataset
        uds_sample = xu.UgridDataset(ds_sample, grids=self.mesh_grids[grid_name])

        self.set_mesh(uds_sample, grid_name=grid_name, overwrite_grid=False)

        return list(ds_sample.data_vars.keys())

    @property
    def mesh(self) -> Union[xu.UgridDataArray, xu.UgridDataset]:
        """
        Model static mesh data. It returns a xugrid.UgridDataset.

        Mesh can contain several grids (1D, 2D, 3D) defined according
        to UGRID conventions. To extract a specific grid, use get_mesh
        method.
        """
        # XU grid data type Xarray dataset with xu sampling.
        if self._mesh is None and self._read:
            self.read_mesh()
        return self._mesh

    def set_mesh(
        self,
        data: Union[xu.UgridDataArray, xu.UgridDataset],
        name: Optional[str] = None,
        grid_name: Optional[str] = None,
        overwrite_grid: Optional[bool] = False,
    ) -> None:
        """Add data to mesh.

        All layers of mesh have identical spatial coordinates in Ugrid conventions.
        Also updates self.region if grid_name is new or overwrite_grid is True.

        Parameters
        ----------
        data: xugrid.UgridDataArray or xugrid.UgridDataset
            new layer to add to mesh, should contain only one grid topology.
        name: str, optional
            Name of new object layer, this is used to overwrite the name of
            a UgridDataArray.
        grid_name: str, optional
            Name of the mesh grid to add data to. If None, inferred from data.
            Can be used for renaming the grid.
        overwrite_grid: bool, optional
            If True, overwrite the grid with the same name as the grid in self.mesh.
        """
        # Check if new grid_name
        if grid_name not in self.mesh_names:
            new_grid = True
        else:
            new_grid = False

        # Checks on data
        if not isinstance(data, (xu.UgridDataArray, xu.UgridDataset)):
            raise ValueError(
                "New mesh data in set_mesh should be of type xu.UgridDataArray"
                " or xu.UgridDataset"
            )
        if isinstance(data, xu.UgridDataArray):
            if name is not None:
                data = data.rename(name)
            elif data.name is None:
                raise ValueError(
                    f"Cannot set mesh from {str(type(data).__name__)} without a name."
                )
            data = data.to_dataset()

        # Checks on grid topology
        # TODO: check if we support setting multiple grids at once. For now just one
        if len(data.ugrid.grids) > 1:
            raise ValueError(
                "set_mesh methods only supports adding data to one grid at a time."
            )
        if grid_name is None:
            grid_name = data.ugrid.grid.name
        elif grid_name != data.ugrid.grid.name:
            data = workflows.rename_mesh(data, name=grid_name)

        # Adding to mesh
        if self.mesh is None:  # NOTE: mesh is initialized with None
            # Check on crs
            if not data.ugrid.grid.crs:
                raise ValueError("Data should have CRS.")
            self._mesh = data
        else:
            # Check on crs
            if not data.ugrid.grid.crs == self.crs:
                raise ValueError("Data and self.mesh should have the same CRS.")
            # Save crs as it will be lost when converting to xarray
            crs = self.crs
            # Check on new grid topology
            if grid_name in self.mesh_names:
                # check if the two grids are the same
                if (
                    not self.mesh_grids[grid_name]
                    .to_dataset()
                    .equals(data.ugrid.grid.to_dataset())
                ):
                    if not overwrite_grid:
                        raise ValueError(
                            f"Grid {grid_name} already exists in mesh"
                            " and has a different topology. "
                            "Use overwrite_grid=True to overwrite the grid"
                            " topology and related data."
                        )
                    else:
                        # Remove grid and all corresponding data variables from mesh
                        self.logger.warning(
                            f"Overwriting grid {grid_name} and the corresponding"
                            " data variables in mesh."
                        )
                        grids = [
                            self.mesh_datasets[g].ugrid.to_dataset(
                                optional_attributes=True
                            )
                            for g in self.mesh_names
                            if g != grid_name
                        ]
                        # Re-define _mesh
                        grids = xr.merge(grids)
                        self._mesh = xu.UgridDataset(grids)
            # Check again mesh_names, could have changed if overwrite_grid=True
            if grid_name in self.mesh_names:
                for dvar in data.data_vars:
                    if dvar in self._mesh:
                        self.logger.warning(f"Replacing mesh parameter: {dvar}")
                    self._mesh[dvar] = data[dvar]
            else:
                # We are potentially adding a new grid without any data variables
                self._mesh = xu.UgridDataset(
                    xr.merge(
                        [
                            self.mesh.ugrid.to_dataset(optional_attributes=True),
                            data.ugrid.to_dataset(optional_attributes=True),
                        ]
                    )
                )
            # Restore crs
            for grid in self._mesh.ugrid.grids:
                grid.set_crs(crs)

        # update related geoms if necessary: region
        if overwrite_grid or new_grid:
            # add / updates region
            if "region" in self.geoms:
                self._geoms.pop("region", None)
            self.region

    def get_mesh(
        self, grid_name: str, include_data: bool = False
    ) -> Union[xu.UgridDataArray, xu.UgridDataset]:
        """
        Return a specific grid topology from mesh based on grid_name.

        If include_data is True, the data variables for that specific
        grid are also included.

        Parameters
        ----------
        grid_name : str
            Name of the grid to return.
        include_data : bool, optional
            If True, also include data variables, by default False.

        Returns
        -------
        uds: Union[xu.UgridDataArray, xu.UgridDataset]
            Grid topology with or without data variables.
        """
        if self.mesh is None:
            raise ValueError("Mesh is not set, please use set_mesh first.")
        if grid_name not in self.mesh_names:
            raise ValueError(f"Grid {grid_name} not found in mesh.")
        if include_data:
            grid = self.mesh_grids[grid_name]
            uds = xu.UgridDataset(grid.to_dataset(optional_attributes=True))
            uds.ugrid.grid.set_crs(grid.crs)
            # Look for data_vars that are defined on grid_name
            for var in self.mesh.data_vars:
                if hasattr(self.mesh[var], "ugrid"):
                    if self.mesh[var].ugrid.grid.name == grid_name:
                        uds[var] = self.mesh[var]
                # additionnal topology properties
                elif var.startswith(grid_name):
                    uds[var] = self.mesh[var]
                # else is global property (not grid specific)

            return uds

        else:
            return self.mesh_grids[grid_name]

    def read_mesh(
        self, fn: str = "mesh/mesh.nc", crs: Union[CRS, int] = None, **kwargs
    ) -> None:
        """Read model mesh data at <root>/<fn> and add to mesh property.

        key-word arguments are passed to :py:meth:`~hydromt.models.Model.read_nc`

        Parameters
        ----------
        fn : str, optional
            filename relative to model root, by default 'mesh/mesh.nc'
        crs : CRS or int, optional
            Coordinate Reference System (CRS) object or EPSG code representing the
            spatial reference system of the mesh file. Only used if the CRS is not
            found when reading the mesh file.
        **kwargs : dict
            Additional keyword arguments to be passed to the `read_nc` method.
        """
        self._assert_read_mode
        ds = xr.merge(self.read_nc(fn, **kwargs).values())
        uds = xu.UgridDataset(ds)
        if ds.rio.crs is not None:  # parse crs
            uds.ugrid.set_crs(ds.raster.crs)
            uds = uds.drop_vars(GEO_MAP_COORD, errors="ignore")
        else:
            if not crs:
                raise ValueError(
                    "no crs is found in the file nor passed to the reader."
                )
            else:
                uds.ugrid.set_crs(crs)
                self.logger.info(
                    "no crs is found in the file, assigning from user input."
                )
        self._mesh = uds

    def write_mesh(
        self,
        fn: str = "mesh/mesh.nc",
        write_optional_ugrid_attributes: bool = True,
        **kwargs,
    ) -> None:
        """Write model grid data to a netCDF file at <root>/<fn>.

        Keyword arguments are passed to :py:meth:`xarray.Dataset.to_netcdf`.

        Parameters
        ----------
        fn : str, optional
            Filename relative to the model root directory, by default 'grid/grid.nc'.
        write_optional_ugrid_attributes : bool, optional
            If True, write optional ugrid attributes to the netCDF file, by default
            True.
        **kwargs : dict
            Additional keyword arguments to be passed to the
            `xarray.Dataset.to_netcdf` method.
        """
        if self.mesh is None:
            self.logger.debug("No mesh data found, skip writing.")
            return
        self._assert_write_mode
        # filename
        _fn = join(self.root, fn)
        if not isdir(dirname(_fn)):
            os.makedirs(dirname(_fn))
        self.logger.debug(f"Writing file {fn}")
        ds_out = self.mesh.ugrid.to_dataset(
            optional_attributes=write_optional_ugrid_attributes,
        )
        if self.crs is not None:
            # save crs to spatial_ref coordinate
            ds_out = ds_out.rio.write_crs(self.crs)
        ds_out.to_netcdf(_fn, **kwargs)

    # Other mesh properties
    @property
    def mesh_grids(self) -> Dict[str, Union[xu.Ugrid1d, xu.Ugrid2d]]:
        """Dictionnary of grid names and Ugrid topologies in mesh."""
        grids = dict()
        if self.mesh is not None:
            for grid in self.mesh.ugrid.grids:
                grids[grid.name] = grid

        return grids

    @property
    def mesh_datasets(self) -> Dict[str, xu.UgridDataset]:
        """Dictionnary of grid names and corresponding UgridDataset topology and data variables in mesh."""  # noqa: E501
        datasets = dict()
        if self.mesh is not None:
            for grid in self.mesh.ugrid.grids:
                datasets[grid.name] = self.get_mesh(
                    grid_name=grid.name, include_data=True
                )

        return datasets

    @property
    def mesh_names(self) -> List[str]:
        """List of grid names in mesh."""
        if self.mesh is not None:
            return [grid.name for grid in self.mesh.ugrid.grids]
        else:
            return []

    @property
    def mesh_gdf(self) -> Dict[str, gpd.GeoDataFrame]:
        """Returns dict of geometry of grids in mesh as a gpd.GeoDataFrame."""
        mesh_gdf = dict()
        if self.mesh is not None:
            for k, grid in self.mesh_grids.items():
                if grid.topology_dimension == 1:
                    dim = grid.edge_dimension
                elif grid.topology_dimension == 2:
                    dim = grid.face_dimension
                gdf = gpd.GeoDataFrame(
                    index=grid.to_dataset()[dim].values.astype(str),
                    geometry=grid.to_shapely(dim),
                )
                mesh_gdf[k] = gdf.set_crs(grid.crs)

        return mesh_gdf


class MeshModel(MeshMixin, Model):

    """Model class Mesh Model for mesh models in HydroMT.

    Uses xugrid for working with unstructured grids, for data and topology stored
    according to UGRID conventions.

    See also: xugrid
    """

    _CLI_ARGS = {"region": "setup_mesh2d", "res": "setup_mesh2d"}
    _NAME = "mesh_model"

    def __init__(
        self,
        root: str = None,
        mode: str = "w",
        config_fn: str = None,
        data_libs: List[str] = None,
        logger=logger,
    ):
        """Initialize a MeshModel for models with an unstructured grid."""
        super().__init__(
            root=root,
            mode=mode,
            config_fn=config_fn,
            data_libs=data_libs,
            logger=logger,
        )

    ## general setup methods
    def setup_mesh2d(
        self,
        region: dict,
        res: Optional[float] = None,
        crs: int = None,
        grid_name: str = "mesh2d",
    ) -> xu.UgridDataset:
        """HYDROMT CORE METHOD: Create an 2D unstructured mesh or reads an existing 2D mesh according UGRID conventions.

        Grids are read according to UGRID conventions. An 2D unstructured mesh
        will be created as 2D rectangular grid from a geometry (geom_fn) or bbox.
        If an existing 2D mesh is given, then no new mesh will be generated but an extent
        can be extracted using the `bounds` argument of region.

        Note Only existing meshed with only 2D grid can be read.

        Adds/Updates model layers:

        * **grid_name** mesh topology: add grid_name 2D topology to mesh object

        Parameters
        ----------
        region : dict
            Dictionary describing region of interest, bounds can be provided for type 'mesh'.
            In case of 'mesh', if the file includes several grids, the specific 2D grid can
            be selected using the 'grid_name' argument.
            CRS for 'bbox' and 'bounds' should be 4326; e.g.:

            * {'bbox': [xmin, ymin, xmax, ymax]}

            * {'geom': 'path/to/polygon_geometry'}

            * {'mesh': 'path/to/2dmesh_file'}

            * {'mesh': 'path/to/mesh_file', 'grid_name': 'mesh2d', 'bounds': [xmin, ymin, xmax, ymax]}
        res: float
            Resolution used to generate 2D mesh [unit of the CRS], required if region
            is not based on 'mesh'.
        crs : EPSG code, int, optional
            Optional EPSG code of the model or "utm" to let hydromt find the closest projected CRS.
            If None using the one from region, and else 4326.
        grid_name : str, optional
            Name of the 2D grid in mesh, by default "mesh2d".

        Returns
        -------
        mesh2d : xu.UgridDataset
            Generated mesh2d.

        """  # noqa: E501
        self.logger.info("Preparing 2D mesh.")

        # Create mesh2d
        mesh2d = workflows.create_mesh2d(
            region=region,
            res=res,
            crs=crs,
            logger=self.logger,
        )
        # Add mesh2d to self.mesh
        self.set_mesh(mesh2d, grid_name=grid_name)

        # This setup method returns mesh2d so that it can be wrapped for models
        # which require more information
        return mesh2d

    ## I/O
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
            List of model components to read, each should have an associated
            read_<component> method. By default ['config', 'maps', 'mesh',
            'geoms', 'forcing', 'states', 'results']
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
            List of model components to write, each should have an
            associated write_<component> method. By default ['config', 'maps',
            'mesh', 'geoms', 'forcing', 'states']
        """
        super().write(components=components)

    # MeshModel specific methods

    # MeshModel properties
    @property
    def bounds(self) -> Dict:
        """Returns model mesh bounds."""
        if self.mesh is not None:
            return self.mesh.ugrid.bounds

    @property
    def crs(self) -> CRS:
        """Returns model mesh crs."""
        if self.mesh is not None:
            grid_crs = self.mesh.ugrid.crs
            # Check if all the same
            crs = None
            for k, v in grid_crs.items():
                if crs is None:
                    crs = v
                if v == crs:
                    continue
                else:
                    raise ValueError(
                        f"Mesh crs is not uniform, please check {grid_crs}"
                    )
            return crs
        else:
            return None

    @property
    def region(self) -> gpd.GeoDataFrame:
        """Returns geometry of region of the model area of interest based on mesh total bounds."""  # noqa: E501
        region = gpd.GeoDataFrame()
        if "region" in self.geoms:
            region = self.geoms["region"]
        elif self.mesh is not None:
            region = gpd.GeoDataFrame(
                geometry=[box(*self.mesh.ugrid.total_bounds)], crs=self.crs
            )
            self.set_geoms(region, name="region")
        return region
