"""Mesh Component."""

import os
from os.path import dirname, isdir, join
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple, Union, cast

import geopandas as gpd
import pandas as pd
import xarray as xr
import xugrid as xu
from pyproj import CRS
from shapely.geometry import box

from hydromt import hydromt_step
from hydromt.components.base import ModelComponent
from hydromt.components.spatial import SpatialModelComponent
from hydromt.gis.raster import GEO_MAP_COORD
from hydromt.io.readers import read_nc
from hydromt.workflows.mesh import (
    create_mesh2d_from_region,
    mesh2d_from_raster_reclass,
    mesh2d_from_rasterdataset,
)

if TYPE_CHECKING:
    from hydromt.models import Model

__all__ = ["MeshComponent"]


class MeshComponent(SpatialModelComponent):
    """ModelComponent class for mesh components.

    This class is used to manage unstructured mesh data in a model. The mesh component
    data stored in the ``data`` property is a xugrid.UgridDataset object.
    """

    def __init__(
        self,
        model: "Model",
        *,
        filename: str = "mesh/mesh.nc",
        region_component: Optional[str] = None,
        region_filename: str = "mesh/mesh_region.geojson",
    ):
        """
        Initialize a MeshComponent.

        Parameters
        ----------
        model: Model
            HydroMT model instance
        filename: str
            The path to use for reading and writing of component data by default.
            by default "mesh/mesh.nc".
        region_component: str, optional
            The name of the region component to use as reference for this component's
            region. If None, the region will be set to the total bounds of the mesh.
            Note that the create method only works if the region_component is None.
            For add_data_from_* methods, the other region_component should be a
            reference to another mesh component for correct reprojection.
        region_filename: str
            The path to use for reading and writing of the region data by default.
            by default "mesh/mesh_region.geojson".
        """
        super().__init__(
            model,
            region_component=region_component,
            region_filename=region_filename,
        )
        self._data: Optional[xu.UgridDataset] = None
        self._filename: str = filename

    def set(
        self,
        data: Union[xu.UgridDataArray, xu.UgridDataset],
        *,
        name: Optional[str] = None,
        grid_name: Optional[str] = None,
        overwrite_grid: bool = False,
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
        self._initialize()
        # Checks on data
        data = _check_UGrid(data, name)

        # Checks on grid topology
        # TODO: check if we support setting multiple grids at once. For now just one
        if len(data.ugrid.grids) > 1:
            raise ValueError(
                "set_mesh methods only supports adding data to one grid at a time."
            )
        if grid_name is None:
            grid_name = data.ugrid.grid.name
        elif grid_name != data.ugrid.grid.name:
            data = data.ugrid.rename({data.ugrid.grid.name: grid_name})
        self._add_mesh(data=data, grid_name=grid_name, overwrite_grid=overwrite_grid)

    @hydromt_step
    def write(
        self,
        filename: Optional[str] = None,
        *,
        region_options: Optional[Dict] = None,
        write_optional_ugrid_attributes: bool = False,
        **kwargs,
    ) -> None:
        """Write model grid data to a netCDF file at <root>/<filename>.

        Keyword arguments are passed to :py:meth:`xarray.Dataset.to_netcdf`.

        Parameters
        ----------
        filename : str, optional
            Filename relative to the model root directory, by default 'grid/grid.nc'.
        write_optional_ugrid_attributes : bool, optional
            If True, write optional ugrid attributes to the netCDF file, by default
            True.
        region_options : dict, optional
            Options to pass to the write_region method.
            Can contain `filename`, `to_wgs84`, and anything that will be passed to `GeoDataFrame.to_file`.
            If `filename` is not provided, `self.region_filename` will be used.
        **kwargs : dict
            Additional keyword arguments to be passed to the
            `xarray.Dataset.to_netcdf` method.
        """
        self.root._assert_write_mode()
        region_options = region_options or {}
        self.write_region(**region_options)

        if len(self.data) < 1:
            self.logger.debug("No mesh data found, skip writing.")
            return

        # filename
        filename = filename or str(self._filename)
        _filename = join(self.root.path, filename)
        if not isdir(dirname(_filename)):
            os.makedirs(dirname(_filename), exist_ok=True)
        self.logger.debug(f"Writing file {filename}")
        ds_out = self.data.ugrid.to_dataset(
            optional_attributes=write_optional_ugrid_attributes,
        )
        if self.crs is not None:
            # save crs to spatial_ref coordinate
            ds_out = ds_out.rio.write_crs(self.crs)
        ds_out.to_netcdf(_filename, **kwargs)

    @hydromt_step
    def read(
        self,
        filename: Optional[str] = None,
        *,
        crs: Optional[Union[CRS, int]] = None,
        **kwargs,
    ) -> None:
        """Read model mesh data at <root>/<filename> and add to mesh property.

        key-word arguments are passed to :py:meth:`~hydromt.models.Model.read_nc`

        Parameters
        ----------
        filename : str, optional
            filename relative to model root, by default 'mesh/mesh.nc'
        crs : CRS or int, optional
            Coordinate Reference System (CRS) object or EPSG code representing the
            spatial reference system of the mesh file. Only used if the CRS is not
            found when reading the mesh file.
        **kwargs : dict
            Additional keyword arguments to be passed to the `read_nc` method.
        """
        self.root._assert_read_mode()
        self._initialize(skip_read=True)

        filename = filename or str(self._filename)
        files = read_nc(
            filename,
            root=self.root.path,
            single_var_as_array=False,
            logger=self.logger,
            **kwargs,
        ).values()
        if len(files) > 0:
            ds = xr.merge(files)
            if ds.rio.crs is not None:  # parse crs
                crs = ds.raster.crs
                ds = ds.drop_vars(GEO_MAP_COORD, errors="ignore")
                uds = xu.UgridDataset(ds)
            else:
                if not crs:
                    raise ValueError(
                        "no crs is found in the file nor passed to the reader."
                    )
                else:
                    uds = xu.UgridDataset(ds)
                    self.logger.info(
                        "no crs is found in the file, assigning from user input."
                    )
            # Reading ugrid data adds nNodes coordinates to grid and makes it not
            # possible to test two equal grids for equality
            if f"{uds.grid.name}_nNodes" in uds.grid.to_dataset():
                uds = xu.UgridDataset(
                    uds.ugrid.to_dataset().drop_vars(f"{uds.grid.name}_nNodes")
                )
            uds.ugrid.set_crs(crs)

            self._data = uds

    @hydromt_step
    def create_2d_from_region(
        self,
        region: Dict[str, Any],
        *,
        res: Optional[float] = None,
        crs: Optional[int] = None,
        region_crs: int = 4326,
        grid_name: str = "mesh2d",
        align: bool = True,
    ) -> xu.UgridDataset:
        """HYDROMT CORE METHOD: Create an 2D unstructured mesh or reads an existing 2D mesh according UGRID conventions.

        Grids are read according to UGRID conventions. An 2D unstructured mesh
        will be created as 2D rectangular grid from a geometry (geom_filename) or bbox.
        If an existing 2D mesh is given, then no new mesh will be generated but an extent
        can be extracted using the `bounds` argument of region.

        Note Only existing meshed with only 2D grid can be read.

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

        res : float, optional
            Resolution used to generate 2D mesh [unit of the CRS], required if region
            is not based on 'mesh'.
        crs : int, optional
            Optional EPSG code of the model.
            If None using the one from region, and else 4326.
        region_crs : int, optional
            EPSG code of the region geometry, by default None. Only applies if region is
            of kind 'bbox'or if geom crs is not defined in the file itself.
        align : bool, default True
            Align the mesh to the resolution.
            Required for 'bbox' and 'geom' region types.
        grid_name : str, optional
            Name of the 2D grid in the mesh, by default 'mesh2d'.

        Returns
        -------
        mesh2d : xu.UgridDataset
            Generated mesh2d.
        """
        self.logger.info("Preparing 2D mesh.")

        # Check if this component's region is a reference to another component
        if self._region_component is not None:
            raise ValueError(
                "Region is a reference to another component. Cannot create grid."
            )

        mesh2d = create_mesh2d_from_region(
            region,
            res=res,
            crs=crs,
            region_crs=region_crs,
            align=align,
            logger=self.logger,
            data_catalog=self.data_catalog,
        )
        self.set(mesh2d, grid_name=grid_name)
        return mesh2d

    @property
    def data(self) -> Union[xu.UgridDataArray, xu.UgridDataset]:
        """
        Model static mesh data. It returns a xugrid.UgridDataset.

        Mesh can contain several grids (1D, 2D, 3D) defined according
        to UGRID conventions. To extract a specific grid, use get_mesh
        method.
        """
        # XU grid data type Xarray dataset with xu sampling.
        if self._data is None:
            self._initialize()
        return self._data

    def _initialize(self, skip_read: bool = False) -> None:
        """Initialize mesh object."""
        if self._data is None:
            self._data = xu.UgridDataset(xr.Dataset())
            if self.root.is_reading_mode() and not skip_read:
                self.read()

    @property
    def crs(self) -> Optional[CRS]:
        """Returns model mesh crs."""
        if len(self.data) > 0:
            for _, v in self.data.ugrid.crs.items():
                return v
        return None

    @property
    def bounds(self) -> Optional[Tuple[float, float, float, float]]:
        """Returns model mesh bounds."""
        if len(self.data) > 0:
            return self.data.ugrid.bounds
        return None

    @property
    def _region_data(self) -> Optional[gpd.GeoDataFrame]:
        """Return mesh total_bounds as a geodataframe."""
        if len(self.data) > 0:
            region = gpd.GeoDataFrame(
                geometry=[box(*self.data.ugrid.total_bounds)], crs=self.crs
            )
            return region
        return None

    @property
    def mesh_names(self) -> List[str]:
        """List of grid names in mesh."""
        if len(self.data.grids) > 0:
            return [grid.name for grid in self.data.ugrid.grids]
        else:
            return []

    @property
    def mesh_grids(self) -> Dict[str, Union[xu.Ugrid1d, xu.Ugrid2d]]:
        """Dictionary of grid names and Ugrid topologies in mesh."""
        grids = dict()
        if len(self.data.grids) > 0:
            for grid in self.data.ugrid.grids:
                grids[grid.name] = grid

        return grids

    @property
    def mesh_datasets(self) -> Dict[str, xu.UgridDataset]:
        """Dictionnary of grid names and corresponding UgridDataset topology and data variables in mesh."""  # noqa: E501
        datasets = dict()
        if len(self.data) > 0:
            for grid in self.data.ugrid.grids:
                datasets[grid.name] = self.get_mesh(
                    grid_name=grid.name, include_data=True
                )

        return datasets

    @property
    def mesh_gdf(self) -> Dict[str, gpd.GeoDataFrame]:
        """Returns dict of geometry of grids in mesh as a gpd.GeoDataFrame."""
        mesh_gdf = dict()
        if len(self.data.grids) > 0:
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

    def get_mesh(
        self, grid_name: str, include_data: bool = False
    ) -> Union[xu.Ugrid1d, xu.Ugrid2d, xu.UgridDataArray, xu.UgridDataset]:
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
        uds: Union[xu.Ugrid1d, xu.Ugrid2d, xu.UgridDataArray, xu.UgridDataset]
            Grid topology with or without data variables.
        """
        if len(self.data) == 0:
            raise ValueError("Mesh is not set, please use set_mesh first.")
        if grid_name not in self.mesh_names:
            raise ValueError(f"Grid {grid_name} not found in mesh.")
        if include_data:
            # Look for data_vars that are defined on grid_name
            variables = []
            for var in self.data.data_vars:
                if hasattr(self.data[var], "ugrid"):
                    if self.data[var].ugrid.grid.name != grid_name:
                        variables.append(var)
                # additional topology properties
                elif not var.startswith(grid_name):
                    variables.append(var)
                # else is global property (not grid specific)
            if variables and len(variables) < len(self.data.data_vars):
                uds = self.data.drop_vars(variables)
                # Drop coords as well
                drop_coords = [c for c in uds.coords if not c.startswith(grid_name)]
                uds = uds.drop_vars(drop_coords)
            elif variables and len(variables) == len(self.data.data_vars):
                grid = self.mesh_grids[grid_name]
                uds = xu.UgridDataset(grid.to_dataset(optional_attributes=True))
                uds.ugrid.grid.set_crs(grid.crs)
            else:
                uds = self.data.copy()

            return uds

        else:
            return self.mesh_grids[grid_name]

    @hydromt_step
    def add_2d_data_from_rasterdataset(
        self,
        raster_filename: Union[str, Path, xr.DataArray, xr.Dataset],
        *,
        grid_name: str = "mesh2d",
        variables: Optional[list] = None,
        fill_method: Optional[str] = None,
        resampling_method: Optional[Union[str, List]] = "centroid",
        rename: Optional[Dict] = None,
    ) -> List[str]:
        """HYDROMT CORE METHOD: Add data variable(s) from ``raster_filename`` to 2D ``grid_name`` in mesh object.

        Raster data is interpolated to the mesh ``grid_name`` using the ``resampling_method``.
        If raster is a dataset, all variables will be added unless ``variables`` list
        is specified.

        Adds model layers:

        * **raster.name** mesh: data from raster_filename

        Parameters
        ----------
        raster_filename: str, Path, xr.DataArray, xr.Dataset
            Data catalog key, path to raster file or raster xarray data object.
        grid_name: str
            Name of the mesh grid to add the data to. By default 'mesh2d'.
        variables: list, optional
            List of variables to add to mesh from raster_filename. By default all.
        fill_method : str, optional
            If specified, fills no data values using fill_nodata method.
            Available methods are {'linear', 'nearest', 'cubic', 'rio_idw'}.
        resampling_method: str, list, optional
            Method to sample from raster data to mesh. By default mean. Options include
            {"centroid", "barycentric", "mean", "harmonic_mean", "geometric_mean", "sum",
            "minimum", "maximum", "mode", "median", "max_overlap"}. If centroid, will use
            :py:meth:`xugrid.CentroidLocatorRegridder` method. If barycentric, will use
            :py:meth:`xugrid.BarycentricInterpolator` method. If any other, will use
            :py:meth:`xugrid.OverlapRegridder` method.
            Can provide a list corresponding to ``variables``.
        rename: dict, optional
            Dictionary to rename variable names in raster_filename before adding to mesh
            {'name_in_raster_filename': 'name_in_mesh'}. By default empty.

        Returns
        -------
        list
            List of variables added to mesh.
        """  # noqa: E501
        self.logger.info(f"Preparing mesh data from raster source {raster_filename}")
        # Get the grid from the mesh or the reference one
        mesh_like = self._get_mesh_grid_data(grid_name=grid_name)

        # Read raster data and select variables
        bounds = self._get_mesh_gdf_data(grid_name).to_crs(4326).total_bounds
        ds = self.data_catalog.get_rasterdataset(
            raster_filename,
            bbox=bounds,
            buffer=2,
            variables=variables,
            single_var_as_array=False,
        )

        uds_sample = mesh2d_from_rasterdataset(
            ds=ds,
            mesh2d=mesh_like,
            variables=variables,
            fill_method=fill_method,
            resampling_method=resampling_method,
            rename=rename,
            logger=self.logger,
        )

        self.set(uds_sample, grid_name=grid_name, overwrite_grid=False)

        return list(uds_sample.data_vars.keys())

    @hydromt_step
    def add_2d_data_from_raster_reclass(
        self,
        raster_filename: Union[str, Path, xr.DataArray],
        reclass_table_filename: Union[str, Path, pd.DataFrame],
        reclass_variables: list,
        grid_name: str = "mesh2d",
        variable: Optional[str] = None,
        fill_method: Optional[str] = None,
        resampling_method: Optional[Union[str, list]] = "centroid",
        rename: Optional[Dict] = None,
        **kwargs,
    ) -> List[str]:
        """HYDROMT CORE METHOD: Add data variable(s) to 2D ``grid_name`` in mesh object by reclassifying the data in ``raster_filename`` based on ``reclass_table_filename``.

        The reclassified raster data
        are subsequently interpolated to the mesh using `resampling_method`.

        Adds model layers:

        * **reclass_variables** mesh: reclassified raster data interpolated to the
            model mesh

        Parameters
        ----------
        raster_filename : str, Path, xr.DataArray
            Data catalog key, path to the raster file, or raster xarray data object.
            Should be a DataArray. If not, use the `variable` argument for selection.
        reclass_table_filename : str, Path, pd.DataFrame
            Data catalog key, path to the tabular data file, or tabular pandas dataframe
            object for the reclassification table of `raster_filename`.
        reclass_variables : list
            List of reclass_variables from the reclass_table_filename table to add to the
            mesh. The index column should match values in raster_filename.
        grid_name : str, optional
            Name of the mesh grid to add the data to. By default 'mesh2d'.
        variable : str, optional
            Name of the raster dataset variable to use. This is only required when
            reading datasets with multiple variables. By default, None.
        fill_method : str, optional
            If specified, fills nodata values in `raster_filename` using the `fill_method`
            method before reclassifying. Available methods are
            {'linear', 'nearest', 'cubic', 'rio_idw'}.
        resampling_method : str or list, optional
            Method to sample from raster data to mesh. By default mean. Options include
            {"centroid", "barycentric", "mean", "harmonic_mean", "geometric_mean", "sum",
            "minimum", "maximum", "mode", "median", "max_overlap"}. If centroid, will use
            :py:meth:`xugrid.CentroidLocatorRegridder` method. If barycentric, will use
            :py:meth:`xugrid.BarycentricInterpolator` method. If any other, will use
            :py:meth:`xugrid.OverlapRegridder` method.
            Can provide a list corresponding to ``reclass_variables``.
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
            If `raster_filename` is not a single variable raster.
        """  # noqa: E501
        self.logger.info(
            f"Preparing mesh data by reclassifying the data in {raster_filename} "
            f"based on {reclass_table_filename}."
        )
        # Get the grid from the mesh or the reference one
        mesh_like = self._get_mesh_grid_data(grid_name=grid_name)
        # Read raster data and mapping table
        bounds = self._get_mesh_gdf_data(grid_name).to_crs(4326).total_bounds
        da = self.data_catalog.get_rasterdataset(
            raster_filename,
            bbox=bounds,
            buffer=2,
            variables=variable,
            **kwargs,
        )
        if not isinstance(da, xr.DataArray):
            raise ValueError(
                f"raster_filename {raster_filename} should be a single variable raster. "
                "Please select one using the 'variable' argument"
            )
        df_vars = self.data_catalog.get_dataframe(
            reclass_table_filename, variables=reclass_variables
        )

        uds_sample = mesh2d_from_raster_reclass(
            da=da,
            df_vars=df_vars,
            mesh2d=mesh_like,
            reclass_variables=reclass_variables,
            fill_method=fill_method,
            resampling_method=resampling_method,
            rename=rename,
            logger=self.logger,
        )

        self.set(uds_sample, grid_name=grid_name, overwrite_grid=False)

        return list(uds_sample.data_vars.keys())

    def _add_mesh(
        self, data: xu.UgridDataset, grid_name: str, overwrite_grid: bool
    ) -> Optional[CRS]:
        if len(self.data) == 0:
            # Check on crs
            if not data.ugrid.grid.crs:
                raise ValueError("Data should have CRS.")
            crs = data.ugrid.grid.crs  # Save crs
            # Needed for grid equality checking when adding new data
            data = xu.UgridDataset(data.ugrid.to_dataset())
            data.grid.set_crs(crs)
            self._data = data
            return None
        else:
            # Check on crs
            if not data.ugrid.grid.crs == self.crs:
                raise ValueError("Data and Mesh should have the same CRS.")
            # Save crs as it will be lost when converting to xarray
            crs = self.crs
            # Check on new grid topology
            if grid_name in self.mesh_names:
                # This makes sure the data has the same coordinates as the existing data
                # check if the two grids are the same
                data = xu.UgridDataset(
                    data.ugrid.to_dataset()
                )  # add nFaces coordinates to grid
                if not self._grid_is_equal(grid_name, data):
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
                        # Re-define _data
                        grids = xr.merge(objects=grids)
                        self._data = xu.UgridDataset(grids)
            # Check again mesh_names, could have changed if overwrite_grid=True
            if grid_name in self.mesh_names:
                grids = [
                    self.mesh_datasets[g].ugrid.to_dataset(optional_attributes=True)
                    for g in self.mesh_names
                ]
                grids = xr.merge(objects=grids)
                for dvar in data.data_vars:
                    if dvar in self._data:
                        self.logger.warning(f"Replacing mesh parameter: {dvar}")
                    # The xugrid check on grid equal does not work properly compared to
                    # our _grid_is_equal method. Add to xarray Dataset and convert back
                    grids[dvar] = data.ugrid.to_dataset()[dvar]
                    # self._data[dvar] = data[dvar]
                self._data = xu.UgridDataset(grids)
            else:
                # We are potentially adding a new grid without any data variables
                self._data = xu.UgridDataset(
                    xr.merge(
                        [
                            self.data.ugrid.to_dataset(optional_attributes=True),
                            data.ugrid.to_dataset(optional_attributes=True),
                        ]
                    )
                )
            if crs:  # Restore crs
                for grid in self.data.ugrid.grids:
                    grid.set_crs(crs)
            return None

    def _grid_is_equal(self, grid_name: str, data: xu.UgridDataset) -> bool:
        return (
            self.mesh_grids[grid_name]
            .to_dataset(optional_attributes=True)
            .equals(data.grid.to_dataset(optional_attributes=True))
        )

    def _get_mesh_grid_data(self, grid_name: str) -> Union[xu.Ugrid1d, xu.Ugrid2d]:
        if self._region_component is not None:
            reference_component = self.model.get_component(self._region_component)
            self._check_mesh_component(grid_name, reference_component)
            mesh_component = cast(MeshComponent, reference_component)
            return mesh_component.mesh_grids[grid_name]
        if self.data is None:
            raise ValueError("No mesh data available.")
        if grid_name not in self.mesh_names:
            raise ValueError(f"Grid {grid_name} not found in mesh.")
        return self.mesh_grids[grid_name]

    def _get_mesh_gdf_data(self, grid_name: str) -> gpd.GeoDataFrame:
        if self._region_component is not None:
            reference_component = self.model.get_component(self._region_component)
            self._check_mesh_component(grid_name, reference_component)
            mesh_component = cast(MeshComponent, reference_component)
            return mesh_component.mesh_gdf[grid_name]
        if self.data is None:
            raise ValueError("No mesh data available.")
        if grid_name not in self.mesh_names:
            raise ValueError("No region data available.")
        return self.mesh_gdf[grid_name]

    def _check_mesh_component(
        self, grid_name: str, reference_component: ModelComponent
    ):
        if not isinstance(reference_component, MeshComponent):
            raise ValueError(
                f"Referenced region component is not a MeshComponent: '{self._region_component}'."
            )
        if reference_component.data is None:
            raise ValueError(
                f"Unable to get mesh data from the referenced region component: '{self._region_component}'"
            )
        if grid_name not in reference_component.mesh_names:
            raise ValueError(
                f"Grid '{grid_name}' not found in mesh of '{self._region_component}'."
            )


def _check_UGrid(
    data: Union[xu.UgridDataArray, xu.UgridDataset], name: Optional[str]
) -> xu.UgridDataset:
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
        return data.to_dataset()
    return data
