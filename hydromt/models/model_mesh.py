"""Implementations for model mesh workloads."""
import logging
import os
from os.path import dirname, isdir, isfile, join
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import geopandas as gpd
import numpy as np
import pandas as pd
import xarray as xr
import xugrid as xu
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
    def setup_mesh_from_rasterdataset(
        self,
        raster_fn: Union[str, Path, xr.DataArray, xr.Dataset],
        variables: Optional[list] = None,
        fill_method: Optional[str] = None,
        resampling_method: Optional[str] = "mean",
        all_touched: Optional[bool] = True,
        rename: Optional[Dict] = dict(),
    ) -> List[str]:
        """HYDROMT CORE METHOD: Add data variable(s) from ``raster_fn`` to mesh object.

        Raster data is interpolated to the mesh grid using the ``resampling_method``.
        If raster is a dataset, all variables will be added unless ``variables`` list
        is specified.

        Adds model layers:

        * **raster.name** mesh: data from raster_fn

        Parameters
        ----------
        raster_fn: str, Path, xr.DataArray, xr.Dataset
            Data catalog key, path to raster file or raster xarray data object.
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
        """
        self.logger.info(f"Preparing mesh data from raster source {raster_fn}")
        # Read raster data and select variables
        ds = self.data_catalog.get_rasterdataset(
            raster_fn, geom=self.region, buffer=2, variables=variables
        )
        if isinstance(ds, xr.DataArray):
            ds = ds.to_dataset()

        if fill_method is not None:
            ds = ds.raster.interpolate_na(method=fill_method)

        # Convert mesh grid as geodataframe for sampling
        # Reprojection happens to gdf inside of zonal_stats method
        ds_sample = ds.raster.zonal_stats(
            gdf=self.mesh_gdf, stats=resampling_method, all_touched=all_touched
        )
        # Rename variables
        rm_dict = {f"{var}_{resampling_method}": var for var in ds.data_vars}
        ds_sample = ds_sample.rename(rm_dict).rename(rename)
        # Convert to UgridDataset
        uds_sample = xu.UgridDataset(ds_sample, grids=self.mesh.ugrid.grid)

        self.set_mesh(uds_sample)

        return list(ds_sample.data_vars.keys())

    def setup_mesh_from_raster_reclass(
        self,
        raster_fn: Union[str, Path, xr.DataArray],
        reclass_table_fn: Union[str, Path, pd.DataFrame],
        reclass_variables: list,
        variable: Optional[str] = None,
        fill_nodata: Optional[str] = None,
        resampling_method: Optional[Union[str, list]] = "mean",
        all_touched: Optional[bool] = True,
        rename: Optional[Dict] = dict(),
        **kwargs,
    ) -> List[str]:
        """HYDROMT CORE METHOD: Add data variable(s) to mesh object by reclassifying the data in ``raster_fn`` based on ``reclass_table_fn``.

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
        # Read raster data and mapping table
        da = self.data_catalog.get_rasterdataset(
            raster_fn, geom=self.region, buffer=2, variables=variable, **kwargs
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
            gdf=self.mesh_gdf,
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
        uds_sample = xu.UgridDataset(ds_sample, grids=self.mesh.ugrid.grid)

        self.set_mesh(uds_sample)

        return list(ds_sample.data_vars.keys())

    @property
    def mesh(self) -> Union[xu.UgridDataArray, xu.UgridDataset]:
        """Model static mesh data. Returns a xarray.Dataset."""
        # XU grid data type Xarray dataset with xu sampling.
        if self._mesh is None and self._read:
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
            Name of new object layer, this is used to overwrite the name of
            a UgridDataArray.
        """
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
        if self._mesh is None:  # NOTE: mesh is initialized with None
            self._mesh = data
        else:
            for dvar in data.data_vars:
                if dvar in self._mesh:
                    self.logger.warning(f"Replacing mesh parameter: {dvar}")
                self._mesh[dvar] = data[dvar]

    def read_mesh(self, fn: str = "mesh/mesh.nc", **kwargs) -> None:
        """Read model mesh data at <root>/<fn> and add to mesh property.

        key-word arguments are passed to :py:func:`xr.open_dataset`

        Parameters
        ----------
        fn : str, optional
            filename relative to model root, by default 'mesh/mesh.nc'
        **kwargs : dict
            Additional keyword arguments to be passed to the `_read_nc` method.
        """
        self._assert_read_mode
        for ds in self._read_nc(fn, **kwargs).values():
            uds = xu.UgridDataset(ds)
            if ds.rio.crs is not None:  # parse crs
                uds.ugrid.grid.set_crs(ds.raster.crs)
                uds = uds.drop_vars(GEO_MAP_COORD, errors="ignore")
            self.set_mesh(uds)

    def write_mesh(self, fn: str = "mesh/mesh.nc", **kwargs) -> None:
        """Write model grid data to a netCDF file at <root>/<fn>.

        Keyword arguments are passed to :py:meth:`xarray.Dataset.ugrid.to_netcdf`.

        Parameters
        ----------
        fn : str, optional
            Filename relative to the model root directory, by default 'grid/grid.nc'.
        **kwargs : dict
            Additional keyword arguments to be passed to the
            `xarray.Dataset.ugrid.to_netcdf` method.
        """
        if self._mesh is None:
            self.logger.debug("No mesh data found, skip writing.")
            return
        self._assert_write_mode
        # filename
        _fn = join(self.root, fn)
        if not isdir(dirname(_fn)):
            os.makedirs(dirname(_fn))
        self.logger.debug(f"Writing file {fn}")
        ds_out = self.mesh.ugrid.to_dataset()
        if self.mesh.ugrid.grid.crs is not None:
            # save crs to spatial_ref coordinate
            ds_out = ds_out.rio.write_crs(self.mesh.ugrid.grid.crs)
        ds_out.to_netcdf(_fn, **kwargs)


class MeshModel(MeshMixin, Model):

    """Model class Mesh Model for mesh models in HydroMT."""

    _CLI_ARGS = {"region": "setup_mesh", "res": "setup_mesh"}
    _NAME = "mesh_model"

    def __init__(
        self,
        root: str = None,
        mode: str = "w",
        config_fn: str = None,
        data_libs: List[str] = None,
        logger=logger,
    ):
        """Initialize a MeshModel for distributed models with an unstructured grid."""
        super().__init__(
            root=root,
            mode=mode,
            config_fn=config_fn,
            data_libs=data_libs,
            logger=logger,
        )

    ## general setup methods
    def setup_mesh(
        self,
        region: dict,
        res: Optional[float] = None,
        crs: int = None,
    ) -> xu.UgridDataset:
        """HYDROMT CORE METHOD: Create an 2D unstructured mesh or reads an existing 2D mesh according UGRID conventions.

        Grids are read according to UGRID conventions. An 2D unstructured mesh
        will be created as 2D rectangular grid from a geometry (geom_fn) or bbox.
        If an existing 2D mesh is given, then no new mesh will be generated

        Note Only existing meshed with only 2D grid can be read.
        #FIXME: read existing 1D2D network file and extract 2D part.

        Adds/Updates model layers:

        * **mesh** mesh topology: add mesh topology to mesh object

        Parameters
        ----------
        region : dict
            Dictionary describing region of interest, e.g.:

            * {'bbox': [xmin, ymin, xmax, ymax]}

            * {'geom': 'path/to/polygon_geometry'}

            * {'mesh': 'path/to/2dmesh_file'}
        res: float
            Resolution used to generate 2D mesh [unit of the CRS], required if region
            is not based on 'mesh'.
        crs : EPSG code, int, optional
            Optional EPSG code of the model. If None using the one from region,
            and else 4326.

        Returns
        -------
        mesh2d : xu.UgridDataset
            Generated mesh2d.

        """  # noqa: E501
        self.logger.info("Preparing 2D mesh.")

        if "mesh" not in region:
            if not isinstance(res, (int, float)):
                raise ValueError("res argument required")
            kind, region = workflows.parse_region(region, logger=self.logger)
            if kind == "bbox":
                bbox = region["bbox"]
                geom = gpd.GeoDataFrame(geometry=[box(*bbox)], crs=4326)
            elif kind == "geom":
                geom = region["geom"]
                if geom.crs is None:
                    raise ValueError('Model region "geom" has no CRS')
            else:
                raise ValueError(
                    f"Region for mesh must of kind [bbox, geom, mesh], kind {kind} "
                    "not understood."
                )
            if crs is not None:
                geom = geom.to_crs(crs)
            # Generate grid based on res for region bbox
            xmin, ymin, xmax, ymax = geom.total_bounds
            # note we flood the number of faces within bounds
            ncol = int((xmax - xmin) // res)
            nrow = int((ymax - ymin) // res)
            dx, dy = res, -res
            faces = []
            for i in range(nrow):
                top = ymax + i * dy
                bottom = ymax + (i + 1) * dy
                for j in range(ncol):
                    left = xmin + j * dx
                    right = xmin + (j + 1) * dx
                    faces.append(box(left, bottom, right, top))
            grid = gpd.GeoDataFrame(geometry=faces, crs=geom.crs)
            # If needed clip to geom
            if kind != "bbox":
                # TODO: grid.intersects(geom) does not seem to work ?
                grid = grid.loc[
                    gpd.sjoin(
                        grid, geom, how="left", predicate="intersects"
                    ).index_right.notna()
                ].reset_index()
            # Create mesh from grid
            grid.index.name = "mesh2d_nFaces"
            mesh2d = xu.UgridDataset.from_geodataframe(grid)
            mesh2d.ugrid.grid.set_crs(grid.crs)

        else:
            mesh2d_fn = region["mesh"]
            if isinstance(mesh2d_fn, (str, Path)) and isfile(mesh2d_fn):
                self.logger.info("An existing 2D grid is used to prepare 2D mesh.")

                ds = xr.open_dataset(mesh2d_fn, mask_and_scale=False)
            elif isinstance(mesh2d_fn, xr.Dataset):
                ds = mesh2d_fn
            else:
                raise ValueError(
                    f"Region 'mesh' file {mesh2d_fn} not found, please check"
                )
            topologies = [
                k for k in ds.data_vars if ds[k].attrs.get("cf_role") == "mesh_topology"
            ]
            for topology in topologies:
                topodim = ds[topology].attrs["topology_dimension"]
                if topodim != 2:  # chek if 2d mesh file else throw error
                    raise NotImplementedError(
                        f"{mesh2d_fn} cannot be opened. Please check if the existing"
                        " grid is an 2D mesh and not 1D2D mesh. "
                        " This option is not yet available for 1D2D meshes."
                    )

            # Continues with a 2D grid
            mesh2d = xu.UgridDataset(ds)
            # Check crs and reproject to model crs
            if crs is None:
                crs = 4326
            if ds.rio.crs is not None:  # parse crs
                mesh2d.ugrid.grid.set_crs(ds.raster.crs)
            else:
                # Assume model crs
                self.logger.warning(
                    f"Mesh data from {mesh2d_fn} doesn't have a CRS."
                    f" Assuming crs option {crs}"
                )
                mesh2d.ugrid.grid.set_crs(crs)
            mesh2d = mesh2d.drop_vars(GEO_MAP_COORD, errors="ignore")

        # Reproject to user crs option if needed
        if mesh2d.ugrid.grid.crs != crs and crs is not None:
            self.logger.info(f"Reprojecting mesh to crs {crs}")
            mesh2d.ugrid.grid.to_crs(self.crs)

        self.set_mesh(mesh2d)

        # This setup method returns region so that it can be wrapped for models
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
            crs = self.mesh.ugrid.grid.crs
            if crs is None and hasattr(crs, "to_epsg"):
                crs = crs.to_epsg()  # not all CRS have an EPSG code
            region = gpd.GeoDataFrame(geometry=[box(*self.bounds)], crs=crs)
        return region

    @property
    def mesh_gdf(self) -> gpd.GeoDataFrame:
        """Returns geometry of mesh as a gpd.GeoDataFrame."""
        if self._mesh is not None:
            name = [n for n in self.mesh.data_vars][0]  # works better on a DataArray
            return self._mesh[name].ugrid.to_geodataframe()
