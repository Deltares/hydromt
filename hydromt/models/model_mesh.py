from typing import Union, Optional, List, Tuple
import logging
import os
from os.path import join, isdir, dirname, isfile
from pathlib import Path
import numpy as np
import xarray as xr
import xugrid as xu
import geopandas as gpd
from shapely.geometry import box, Polygon

from ..raster import GEO_MAP_COORD
from .model_api import Model
from .. import workflows

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
    def setup_mesh_from_raster(
        self,
        raster_fn: str,
        variables: Optional[list] = None,
        fill_method: Optional[str] = None,
        resampling_method: Optional[str] = "mean",
        all_touched: Optional[bool] = True,
    ) -> None:
        """
        This component adds data variable(s) from ``raster_fn`` to mesh object.

        Raster data is interpolated to the mesh grid using the ``resampling_method``.
        If raster is a dataset, all variables will be added unless ``variables`` list is specified.

        Adds model layers:

        * **raster.name** mesh: data from raster_fn

        Parameters
        ----------
        raster_fn: str
            Source name of raster data in data_catalog.
        variables: list, optional
            List of variables to add to mesh from raster_fn. By default all.
        fill_method : str, optional
            If specified, fills no data values using fill_nodata method. AVailable methods
            are {'linear', 'nearest', 'cubic', 'rio_idw'}.
        resampling_method: str, optional
            Method to sample from raster data to mesh. By default mean. Options include
            {'count', 'min', 'max', 'sum', 'mean', 'std', 'median', 'q##'}.
        all_touched : bool, optional
            If True, all pixels touched by geometries will used to define the sample.
            If False, only pixels whose center is within the geometry or that are
            selected by Bresenham's line algorithm will be used. By default True.
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
        ds_sample = ds_sample.rename(rm_dict)
        # Convert to UgridDataset
        uds_sample = xu.UgridDataset(ds_sample, grids=self.mesh.ugrid.grid)

        self.set_mesh(uds_sample)

    def setup_mesh_from_rastermapping(
        self,
        raster_fn: str,
        raster_mapping_fn: str,
        mapping_variables: list,
        fill_nodata: Optional[str] = None,
        resampling_method: Optional[Union[str, list]] = "mean",
        all_touched: Optional[bool] = True,
        **kwargs,
    ) -> None:
        """
        This component adds data variable(s) to mesh object by combining values in ``raster_mapping_fn`` to spatial layer ``raster_fn``.

        The ``mapping_variables`` rasters are first created by mapping variables values from ``raster_mapping_fn`` to value in the
        ``raster_fn`` grid.
        Mapping variables data are then interpolated to the mesh grid using ``resampling_method``.

        Adds model layers:

        * **mapping_variables** mesh: data from raster_mapping_fn spatially ditributed with raster_fn

        Parameters
        ----------
        raster_fn: str
            Source name of raster data in data_catalog. Should be a DataArray. Else use **kwargs to select variables/time_tuple in
            hydromt.data_catalog.get_rasterdataset method
        raster_mapping_fn: str
            Source name of mapping table of raster_fn in data_catalog.
        mapping_variables: list
            List of mapping_variables from rasert_mapping_fn table to add to mesh. Index column should match values in raster_fn.
        fill_nodata : str, optional
            If specified, fills no data values using fill_nodata method. AVailable methods
            are {'linear', 'nearest', 'cubic', 'rio_idw'}.
        resampling_method: str/list, optional
            Method to sample from raster data to mesh. Can be a list per variable in ``mapping_variables`` or a
            single method for all. By default mean for all mapping_variables. Options include
            {'count', 'min', 'max', 'sum', 'mean', 'std', 'median', 'q##'}.
        all_touched : bool, optional
            If True, all pixels touched by geometries will used to define the sample.
            If False, only pixels whose center is within the geometry or that are
            selected by Bresenham's line algorithm will be used. By default True.
        """
        self.logger.info(
            f"Preparing mesh data from mapping {mapping_variables} values in {raster_mapping_fn} to raster source {raster_fn}"
        )
        # Read raster data and mapping table
        da = self.data_catalog.get_rasterdataset(
            raster_fn, geom=self.region, buffer=2, **kwargs
        )
        if not isinstance(da, xr.DataArray):
            raise ValueError(
                f"raster_fn {raster_fn} for mapping should be a single variable. Please select one using 'variable' argument in setup_auxmaps_from_rastermapping"
            )
        df_vars = self.data_catalog.get_dataframe(
            raster_mapping_fn, variables=mapping_variables
        )

        if fill_nodata is not None:
            ds = ds.raster.interpolate_na(method=fill_nodata)

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
            resampling_method = np.repeat(resampling_method, len(mapping_variables))
        rm_dict = {
            f"{var}_{mtd}": var
            for var, mtd in zip(mapping_variables, resampling_method)
        }
        ds_sample = ds_sample.rename(rm_dict)
        ds_sample = ds_sample[mapping_variables]
        # Convert to UgridDataset
        uds_sample = xu.UgridDataset(ds_sample, grids=self.mesh.ugrid.grid)

        self.set_mesh(uds_sample)

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
        if self.mesh.ugrid.grid.crs is not None:
            # save crs to spatial_ref coordinate
            ds_out = ds_out.rio.write_crs(self.mesh.ugrid.grid.crs)
        ds_out.to_netcdf(_fn, **kwargs)


class MeshModel(MeshMixin, Model):

    _CLI_ARGS = {"region": "setup_mesh"}

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

    ## general setup methods
    def setup_mesh(
        self,
        region: dict,
        crs: int = None,
        res: float = 100.0,
    ) -> xu.UgridDataset:
        """Creates an 2D unstructured mesh or reads an existing 2D mesh according UGRID conventions.
        An 2D unstructured mesh will be created as 2D rectangular grid from a geometry (geom_fn) or bbox. If an existing
        2D mesh is given, then no new mesh will be generated

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
        crs : EPSG code, int, optional
            Optional EPSG code of the model. If None using the one from region, and else 4326.
        resolution: float, optional
            Resolution used to generate 2D mesh. By default a value of 100 m is applied.

        Returns
        -------
        mesh2d : xu.UgridDataset
            Generated mesh2d.

        """
        self.logger.info(f"Preparing 2D mesh.")

        if "mesh" not in region:
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
                    f"Region for mesh must of kind [bbox, geom, mesh], kind {kind} not understood."
                )
            # Generate grid based on res for region bbox
            xmin, ymin, xmax, ymax = geom.total_bounds
            length = (xmax - xmin) / res
            wide = (ymax - ymin) / res
            cols = list(np.arange(xmin, xmax + wide, wide))
            rows = list(np.arange(ymin, ymax + length, length))
            polygons = []
            for x in cols[:-1]:
                for y in rows[:-1]:
                    polygons.append(
                        Polygon(
                            [
                                (x, y),
                                (x + wide, y),
                                (x + wide, y + length),
                                (x, y + length),
                            ]
                        )
                    )
            grid = gpd.GeoDataFrame({"geometry": polygons}, crs=geom.crs)
            # If needed clip to geom
            if kind != "bbox":
                grid = grid.overlay(geom, how="intersection").explode().reset_index()
            # Create mesh from grid
            grid.index.name = "mesh2d_nFaces"
            mesh2d = xu.UgridDataset.from_geodataframe(grid)
            mesh2d.ugrid.grid.set_crs(grid.crs)

        else:
            mesh2d_fn = region["mesh"]
            if isinstance(mesh2d_fn, (str, Path)) and isfile(mesh2d_fn):
                self.logger.info(f"An existing 2D grid is used to prepare 2D mesh.")

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
                        f"{mesh2d_fn} cannot be opened. Please check if the existing grid is "
                        f"an 2D mesh and not 1D2D mesh. This option is not yet available for 1D2D meshes."
                    )

            # Continues with a 2D grid
            mesh2d = xu.UgridDataset(ds)
            # Check crs and reproject to model crs
            if crs is None:
                crs = 4326
            if ds.rio.crs is not None:  # parse crs
                mesh2d.ugrid.grid.set_crs(ds.rio.crs)
            else:
                # Assume model crs
                self.logger.warning(
                    f"Mesh data from {mesh2d_fn} doesn't have a CRS. Assuming crs option {crs}"
                )
                mesh2d.ugrid.grid.set_crs(crs)
            mesh2d = mesh2d.drop_vars(GEO_MAP_COORD, errors="ignore")

        # Reproject to user crs option if needed
        if mesh2d.ugrid.grid.crs != crs and crs is not None:
            self.logger.info(f"Reprojecting mesh to crs {crs}")
            mesh2d.ugrid.grid.to_crs(self.crs)

        self.set_mesh(mesh2d)

        # This setup method returns region so that it can be wrapped for models which require
        # more information
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
            crs = self.mesh.ugrid.grid.crs
            if crs is None and hasattr(crs, "to_epsg"):
                crs = crs.to_epsg()  # not all CRS have an EPSG code
            region = gpd.GeoDataFrame(geometry=[box(*self.bounds)], crs=crs)
        return region

    @property
    def mesh_gdf(self) -> gpd.GeoDataFrame:
        """Returns geometry of mesh as a gpd.GeoDataFrame"""
        if self._mesh is not None:
            name = [n for n in self.mesh.data_vars][0]  # works better on a DataArray
            return self._mesh[name].ugrid.to_geodataframe()
