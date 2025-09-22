"""Mesh Component."""

import logging
from typing import TYPE_CHECKING, Dict, List, Optional, Tuple, Union, cast

import geopandas as gpd
import xarray as xr
import xugrid as xu
from pyproj import CRS
from shapely.geometry import box

from hydromt.gis.raster import GEO_MAP_COORD
from hydromt.io.readers import open_ncs
from hydromt.model.components.base import ModelComponent
from hydromt.model.components.spatial import SpatialModelComponent
from hydromt.model.steps import hydromt_step

if TYPE_CHECKING:
    from hydromt.model import Model

__all__ = ["MeshComponent"]


logger = logging.getLogger(__name__)


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
        data = self._check_ugrid(data, name)

        # Checks on grid topology
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
        **kwargs : dict
            Additional keyword arguments to be passed to the
            `xarray.Dataset.to_netcdf` method.
        """
        self.root._assert_write_mode()

        if len(self.data) == 0:
            logger.info(
                f"{self.model.name}.{self.name_in_model}: No mesh data found, skip writing."
            )
            return

        filename = filename or self._filename
        full_path = self.root.path / filename
        logger.info(
            f"{self.model.name}.{self.name_in_model}: Writing mesh to {full_path}."
        )
        full_path.parent.mkdir(parents=True, exist_ok=True)

        ds_out = self.data.ugrid.to_dataset(
            optional_attributes=write_optional_ugrid_attributes,
        )
        if self.crs is not None:
            # save crs to spatial_ref coordinate
            ds_out = ds_out.rio.write_crs(self.crs)
        ds_out.to_netcdf(full_path, **kwargs)

    @hydromt_step
    def read(
        self,
        filename: Optional[str] = None,
        *,
        crs: Optional[Union[CRS, int]] = None,
        **kwargs,
    ) -> None:
        """Read model mesh data at <root>/<filename> and add to mesh property.

        key-word arguments are passed to :py:meth:`~hydromt.model.Model.open_nc`

        Parameters
        ----------
        filename : str, optional
            filename relative to model root, by default 'mesh/mesh.nc'
        crs : CRS or int, optional
            Coordinate Reference System (CRS) object or EPSG code representing the
            spatial reference system of the mesh file. Only used if the CRS is not
            found when reading the mesh file.
        **kwargs : dict
            Additional keyword arguments to be passed to the `open_nc` method.
        """
        self.root._assert_read_mode()
        self._initialize(skip_read=True)

        filename = filename or str(self._filename)
        files = open_ncs(filename, root=self.root.path, **kwargs).values()
        self._open_datasets.extend(files)
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
                    logger.info(
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
            return next(iter(self.data.ugrid.crs.values()))
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
            if data.ugrid.grid.crs != self.crs:
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
                        logger.warning(
                            f"Overwriting grid {grid_name} and the corresponding"
                            " data variables in mesh."
                        )
                        grids: List[xr.Dataset] = [
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
                grids: List[xr.Dataset] = [
                    self.mesh_datasets[g].ugrid.to_dataset(optional_attributes=True)
                    for g in self.mesh_names
                ]
                grids = xr.merge(objects=grids)
                for dvar in data.data_vars:
                    if dvar in self._data:
                        logger.warning(f"Replacing mesh parameter: {dvar}")
                    # The xugrid check on grid equal does not work properly compared to
                    # our _grid_is_equal method. Add to xarray Dataset and convert back
                    grids[dvar] = data.ugrid.to_dataset()[dvar]
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

    @staticmethod
    def _check_ugrid(
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
