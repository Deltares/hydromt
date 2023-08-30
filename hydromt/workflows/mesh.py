"""Implementation for mesh based workflows."""
import logging
from typing import Dict, Optional, Union

import geopandas as gpd
import xugrid as xu
from pyproj import CRS
from shapely.geometry import box
from xugrid.ugrid import conventions

from .. import gis_utils
from ..raster import GEO_MAP_COORD
from .basin_mask import parse_region

logger = logging.getLogger(__name__)

__all__ = [
    "create_mesh2d",
    "rename_mesh",
]


def create_mesh2d(
    region: Dict,
    res: Optional[float] = None,
    crs: Optional[int] = None,
    align: bool = True,
    logger=logger,
) -> xu.UgridDataset:
    """
    Create an 2D unstructured mesh or reads an existing 2D mesh.

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
        Dictionary describing region of interest, bounds can be provided
        for type 'mesh'. In case of 'mesh', if the file includes several
        grids, the specific 2D grid can be selected using the 'grid_name' argument.
        CRS for 'bbox' and 'bounds' should be 4326; e.g.:

        * {'bbox': [xmin, ymin, xmax, ymax]}

        * {'geom': 'path/to/polygon_geometry'}

        * {'mesh': 'path/to/2dmesh_file'}

        * {'mesh': 'path/to/2dmesh_file', 'grid_name': 'mesh2d', 'bounds': [xmin, ymin, xmax, ymax]}
    res: float
        Resolution used to generate 2D mesh [unit of the CRS], required if region
        is not based on 'mesh'.
    crs : EPSG code, int, optional
        Optional EPSG code of the model or "utm" to let hydromt find the closest
        projected CRS. If None using the one from region, and else 4326.
    align : bool, optional
        Align the mesh to the resolution, by default True.

    Returns
    -------
    mesh2d : xu.UgridDataset
        Generated mesh2d.
    """  # noqa: E501
    kind, region = parse_region(region, logger=logger)
    if kind != "mesh":
        if not isinstance(res, (int, float)):
            raise ValueError("res argument required for kind 'bbox', 'geom'")
        if kind == "bbox":
            bbox = region["bbox"]
            geom = gpd.GeoDataFrame(geometry=[box(*bbox)], crs=4326)
        elif kind == "geom":
            geom = region["geom"]
            if geom.crs is None:
                raise ValueError('Model region "geom" has no CRS')
        else:
            raise ValueError(
                "Region for mesh must be of kind [bbox, geom, mesh], "
                f"kind {kind} not understood."
            )
        # Parse crs and reproject geom if needed
        if crs is not None:
            crs = gis_utils.parse_crs(crs, bbox=geom.total_bounds)
            geom = geom.to_crs(crs)
        # Generate grid based on res for region bbox
        xmin, ymin, xmax, ymax = geom.total_bounds
        if align:
            xmin = round(xmin / res) * res
            ymin = round(ymin / res) * res
            xmax = round(xmax / res) * res
            ymax = round(ymax / res) * res
        # note we flood the number of faces within bounds
        ncol = round((xmax - xmin) / res)  # int((xmax - xmin) // res)
        nrow = round((ymax - ymin) / res)  # int((ymax - ymin) // res)
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
            grid = grid.loc[
                gpd.sjoin(
                    grid, geom, how="left", predicate="intersects"
                ).index_right.notna()
            ].reset_index()
        # Create mesh from grid
        grid.index.name = "mesh2d_nFaces"
        mesh2d = xu.UgridDataset.from_geodataframe(grid)
        mesh2d = mesh2d.ugrid.assign_face_coords()
        mesh2d.ugrid.grid.set_crs(grid.crs)

    else:
        # Read existing mesh
        uds = region["mesh"]

        # Select grid and/or check single grid
        grids = dict()
        for grid in uds.grids:
            grids[grid.name] = grid
        # Select specific topology if given
        if "grid_name" in region:
            if region["grid_name"] not in grids:
                raise ValueError(
                    f"Mesh file does not contain grid {region['grid_name']}."
                )
            grid = grids[region["grid_name"]]
        elif len(grids) == 1:
            grid = list(grids.values())[0]
        else:
            raise ValueError(
                "Mesh file contains several grids. "
                "Use grid_name argument of region to select a single one."
            )
        # Chek if 2d mesh file else throw error
        if grid.topology_dimension != 2:
            raise ValueError("Grid in mesh file for create_mesh2d is not 2D.")
        # Continues with a 2D grid
        mesh2d = xu.UgridDataset(grid.to_dataset())

        # Check crs and reproject to model crs
        grid_crs = grid.crs
        if crs is not None:
            bbox = None
            if grid_crs is not None:
                if grid_crs.to_epsg() == 4326:
                    bbox = mesh2d.ugrid.grid.bounds
            crs = gis_utils.parse_crs(crs, bbox=bbox)
        else:
            crs = CRS.from_user_input(4326)
        if grid_crs is not None:  # parse crs
            mesh2d.ugrid.grid.set_crs(grid_crs)
        else:
            # Assume model crs
            logger.warning(
                "Mesh data from mesh file doesn't have a CRS."
                f" Assuming crs option {crs}"
            )
            mesh2d.ugrid.grid.set_crs(crs)
        mesh2d = mesh2d.drop_vars(GEO_MAP_COORD, errors="ignore")

        # If bounds are provided in region, extract mesh for bounds
        if "bounds" in region:
            bounds = gpd.GeoDataFrame(geometry=[box(*region["bounds"])], crs=4326)
            xmin, ymin, xmax, ymax = bounds.to_crs(mesh2d.ugrid.grid.crs).total_bounds
            subset = mesh2d.ugrid.sel(y=slice(ymin, ymax), x=slice(xmin, xmax))
            # Check if still cells after clipping
            err = (
                "MeshDataset: No data within model region."
                "Check that bounds were given in the correct crs 4326"
            )
            subset = subset.ugrid.assign_node_coords()
            if subset.ugrid.grid.node_x.size == 0 or subset.ugrid.grid.node_y.size == 0:
                raise IndexError(err)
            mesh2d = subset

    # Reproject to user crs option if needed
    if mesh2d.ugrid.grid.crs != crs:
        logger.info(f"Reprojecting mesh to crs {crs}")
        mesh2d.ugrid.grid.to_crs(crs)

    return mesh2d


def rename_mesh(mesh: Union[xu.UgridDataArray, xu.UgridDataset], name: str):
    """
    Rename all grid variables in mesh according to UGRID conventions.

    Note: adapted from xugrid.ugrid.grid.rename to also work on
    UgridDataset and UgridDataArray
    """
    # Get the old and the new names. Their keys are the same.
    old_attrs = mesh.ugrid.grid._attrs
    new_attrs = conventions.default_topology_attrs(
        name, mesh.ugrid.grid.topology_dimension
    )

    # The attrs will have some roles joined together, e.g. node_coordinates
    # will contain x and y as "mesh2d_node_x mesh2d_node_y".
    name_dict = {mesh.ugrid.grid.name: name}
    skip = ("cf_role", "long_name", "topology_dimension")
    for key, value in old_attrs.items():
        if key in new_attrs and key not in skip:
            split_new = new_attrs[key].split()
            split_old = value.split()
            if len(split_new) != len(split_old):
                raise ValueError(
                    f"Number of entries does not match on {key}: "
                    f"{split_new} versus {split_old}"
                )
            for name_key, name_value in zip(split_old, split_new):
                name_dict[name_key] = name_value

    new = mesh.copy()
    new.ugrid.grid.name = name
    new.ugrid.grid._attrs = new_attrs
    new.ugrid.grid._indexes = {
        k: name_dict[v] for k, v in new.ugrid.grid._indexes.items()
    }

    to_rename = tuple(new.data_vars) + tuple(new.coords) + tuple(new.dims)
    new = new.rename({k: v for k, v in name_dict.items() if k in to_rename})

    return new
