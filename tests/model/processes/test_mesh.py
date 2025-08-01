from unittest.mock import MagicMock

import geopandas as gpd
import numpy as np
import pandas as pd
import pytest
import xarray as xr
import xugrid as xu
from pyproj import CRS
from shapely import box

from hydromt.model.processes.mesh import (
    create_mesh2d_from_geom,
    create_mesh2d_from_mesh,
    create_mesh2d_from_region,
    mesh2d_from_raster_reclass,
    mesh2d_from_rasterdataset,
)


def test_create_mesh2d_from_region():
    res = 500
    crs = 28992
    region = {"bbox": [120000.0, 450000.0, 122000.0, 452000.0]}

    mesh = create_mesh2d_from_region(region=region, res=res, crs=crs, region_crs=crs)

    assert mesh.sizes["mesh2d_nFaces"] > 0, "Mesh should contain at least one face"
    assert "mesh2d_face_x" in mesh.coords, "Missing x face coordinates"
    assert "mesh2d_face_y" in mesh.coords, "Missing y face coordinates"

    x = mesh.mesh2d_face_x.values
    y = mesh.mesh2d_face_y.values
    assert x.min() >= region["bbox"][0], "X min below region bound"
    assert x.max() <= region["bbox"][2], "X max above region bound"
    assert y.min() >= region["bbox"][1], "Y min below region bound"
    assert y.max() <= region["bbox"][3], "Y max above region bound"

    expected_ncol = (region["bbox"][2] - region["bbox"][0]) // res
    expected_nrow = (region["bbox"][3] - region["bbox"][1]) // res
    expected_faces = expected_ncol * expected_nrow
    assert mesh.sizes["mesh2d_nFaces"] == expected_faces, "Unexpected number of faces"


def test_mesh2d_from_rasterdataset():
    crs = 28992
    res = 100
    width, height = 5, 5  # 5x5 raster
    xmin, ymin = 100_000, 400_000
    xmax, ymax = xmin + width * res, ymin + height * res

    x = np.arange(xmin + res / 2, xmax, res)
    y = np.arange(ymin + res / 2, ymax, res)
    data = np.arange(width * height).reshape(height, width)
    geom = gpd.GeoDataFrame(geometry=[box(xmin, ymin, xmax, ymax)], crs=crs)
    mesh = xu.Ugrid2d.from_geodataframe(geom)

    da = xr.DataArray(
        data,
        coords={"y": y[::-1], "x": x},
        dims=("y", "x"),
        name="test_var",
    )
    da.rio.write_crs(CRS.from_epsg(crs), inplace=True)
    ds = da.to_dataset()

    uds_out = mesh2d_from_rasterdataset(
        ds=ds,
        mesh2d=mesh,
        resampling_method="centroid",
    )

    # Assert
    assert isinstance(uds_out, xu.UgridDataset)
    assert "test_var" in uds_out.data_vars
    assert uds_out.test_var.ugrid.grid.crs.to_epsg() == crs
    assert uds_out.test_var.notnull().all(), "Resampled data contains NaNs"
    assert uds_out.test_var.dims == ("mesh2d_nFaces",)
    assert uds_out.test_var.size == mesh.sizes["mesh2d_nFaces"], (
        "Unexpected number of faces in data"
    )


def test_mesh2d_from_rasterdataset_full():
    # Arrange
    crs = 28992
    res = 100
    width, height = 5, 5
    xmin, ymin = 100_000, 400_000
    xmax, ymax = xmin + width * res, ymin + height * res

    x = np.arange(xmin + res / 2, xmax, res)
    y = np.arange(ymin + res / 2, ymax, res)

    var1 = np.arange(width * height).reshape(height, width).astype(np.float32)
    var2 = np.full_like(var1, fill_value=np.nan)
    var2[2, 2] = 42  # single non-NaN value to test fill_method

    ds = xr.Dataset(
        {
            "a": (("y", "x"), var1),
            "b": (("y", "x"), var2),
        },
        coords={"x": x, "y": y[::-1]},
    )
    ds.rio.write_crs(CRS.from_epsg(crs), inplace=True)

    # Mesh covering same extent
    geom = gpd.GeoDataFrame(geometry=[box(xmin, ymin, xmax, ymax)], crs=crs)
    mesh = xu.Ugrid2d.from_geodataframe(geom)

    # Act
    out = mesh2d_from_rasterdataset(
        ds=ds,
        mesh2d=mesh,
        variables=["a", "b"],
        fill_method="nearest",
        resampling_method=["centroid", "centroid"],
        rename={"a": "resampled_a", "b": "resampled_b"},
    )

    # Assert: Output type and content
    assert isinstance(out, xu.UgridDataset)
    assert set(out.data_vars) == {"resampled_a", "resampled_b"}

    # Assert: No NaNs (after fill_method)
    assert out["resampled_a"].notnull().all()
    assert out["resampled_b"].notnull().all()

    # Assert: Output dimensions match mesh
    assert all(out[v].dims == ("mesh2d_nFaces",) for v in out.data_vars)
    assert all(out[v].size == mesh.sizes["mesh2d_nFaces"] for v in out.data_vars)

    # Assert: CRS matches
    assert out["resampled_a"].ugrid.grid.crs.to_epsg() == crs
    assert out["resampled_b"].ugrid.grid.crs.to_epsg() == crs

    # Assert: Values make sense (not all zero or identical unless expected)
    assert not np.allclose(out["resampled_a"].values, out["resampled_b"].values), (
        "Outputs should differ"
    )
    assert np.any(out["resampled_b"].values == 42), (
        "Filled value from 'b' not preserved"
    )

    # Assert: Rename worked correctly
    assert "a" not in out
    assert "b" not in out


def test_mesh2d_from_raster_reclass_full():
    # Arrange: basic raster setup
    res = 100
    crs = 28992
    width, height = 5, 5
    xmin, ymin = 100_000, 400_000
    xmax, ymax = xmin + width * res, ymin + height * res

    x = np.arange(xmin + res / 2, xmax, res)
    y = np.arange(ymin + res / 2, ymax, res)
    raster_values = np.array(
        [
            [1, 1, 2, 2, 3],
            [1, 2, 2, 3, 3],
            [1, 2, 3, 3, 3],
            [2, 2, 3, 3, 4],
            [2, 3, 3, 4, 4],
        ],
        dtype=np.float32,
    )

    da = xr.DataArray(
        data=raster_values,
        coords={"y": y[::-1], "x": x},
        dims=("y", "x"),
        name="land_cover",
    )
    da.rio.write_crs(crs, inplace=True)

    # Reclass table: land_cover -> landuse and roughness
    landuse_mapping = {1.0: "urban", 2.0: "agriculture", 3.0: "forest", 4.0: "water"}
    df_vars = pd.DataFrame(
        {
            "class": [1.0, 2.0, 3.0, 4.0],
            "landuse": list(landuse_mapping.keys()),
            "roughness_manning": [0.01, 0.03, 0.04, 0.05],
        }
    ).set_index("class")

    # Mesh from geometry covering same extent
    geom = gpd.GeoDataFrame(geometry=[box(xmin, ymin, xmax, ymax)], crs=crs)
    mesh = xu.Ugrid2d.from_geodataframe(geom)

    # Act
    reclass_vars = ["landuse", "roughness_manning"]
    rename_map = {"landuse": "lc_type", "roughness_manning": "manning_n"}

    out = mesh2d_from_raster_reclass(
        da=da,
        df_vars=df_vars,
        mesh2d=mesh,
        reclass_variables=reclass_vars,
        fill_method="nearest",  # coverage for fill_method
        resampling_method=["mode", "mean"],  # test method dispatch
        rename=rename_map,
    )

    # Assert: Output type
    assert isinstance(out, xu.UgridDataset)

    # Assert: Expected renamed variables
    assert set(out.data_vars) == {"lc_type", "manning_n"}

    # Assert: Output dimensions and sizes match mesh
    for var in out.data_vars:
        assert out[var].dims == ("mesh2d_nFaces",)
        assert out[var].size == mesh.sizes["mesh2d_nFaces"]

    # Assert: CRS is preserved
    assert out.grid.crs.to_epsg() == crs

    # Assert: Value checks
    lc_values = out["lc_type"].values
    n_values = out["manning_n"].values
    assert np.issubdtype(lc_values.dtype, np.floating)
    assert np.issubdtype(n_values.dtype, np.floating)

    # Assert: land cover reclassification results in known mapped types
    lc_values_str = np.vectorize(landuse_mapping.get)(out["lc_type"].values)
    assert all(val in landuse_mapping.values() for val in lc_values_str)

    # Assert: manning_n values are within expected reclassified bounds
    expected_min = df_vars["roughness_manning"].min()
    expected_max = df_vars["roughness_manning"].max()
    assert expected_min <= n_values.min() <= expected_max
    assert expected_min <= n_values.max() <= expected_max


def test_create_mesh2d_from_geom_clipped():
    # Arrange
    crs = 28992
    res = 100
    xmin, ymin = 100_000, 400_000
    xmax, ymax = 100_500, 400_500
    geom = gpd.GeoDataFrame(geometry=[box(xmin, ymin, xmax, ymax)], crs=crs)

    # Act
    mesh = create_mesh2d_from_geom(
        geom=geom,
        res=res,
        align=True,
        clip_to_geom=True,
    )

    # Assert: Output type and CRS
    assert isinstance(mesh, xu.UgridDataset), "Output is not a UgridDataset"
    assert mesh.grid.crs.to_epsg() == crs, "CRS mismatch"

    # Assert: Coordinates are present
    assert "mesh2d_face_x" in mesh.coords, "Missing face x coordinates"
    assert "mesh2d_face_y" in mesh.coords, "Missing face y coordinates"

    # Assert: Mesh bounds are within or equal to original geometry bounds
    x = mesh.mesh2d_face_x.values
    y = mesh.mesh2d_face_y.values
    assert x.min() >= xmin, "X min is below geometry bounds"
    assert x.max() <= xmax, "X max exceeds geometry bounds"
    assert y.min() >= ymin, "Y min is below geometry bounds"
    assert y.max() <= ymax, "Y max exceeds geometry bounds"

    # Assert: Clipping works (faces should be spatially within original geometry)
    mesh_geom = mesh.ugrid.to_geodataframe()
    assert mesh_geom.within(geom.geometry[0]).any(), "No mesh faces within geometry"
    assert mesh.sizes["mesh2d_nFaces"] > 0, "Mesh has no faces after clipping"


def test_create_mesh2d_from_mesh():
    # Arrange: Create a basic mesh with CRS and known bounds
    crs = 28992
    xmin, ymin = 100_000, 400_000
    xmax, ymax = 100_500, 400_500

    # Create a mesh from geometry
    geom = gpd.GeoDataFrame(geometry=[box(xmin, ymin, xmax, ymax)], crs=crs)
    mesh_orig = xu.Ugrid2d.from_geodataframe(geom)
    uds = xu.UgridDataset(mesh_orig.copy().to_dataset())

    # Assign a grid name for testing grid_name functionality
    mesh_orig.name = "test_grid"
    uds.grids[0].name = "test_grid"

    # Act: Without bounds
    mesh_out = create_mesh2d_from_mesh(
        uds=uds,
        grid_name="test_grid",
        crs=crs,
        bounds=None,
    )

    # Assert: Mesh is returned and has correct type and CRS
    assert isinstance(mesh_out, xu.UgridDataset), (
        "Returned object is not a UgridDataset"
    )
    assert mesh_out.grid.crs.to_epsg() == crs, "CRS was not set correctly"

    # Assert: Coordinates and dimensions
    assert "mesh2d_node_x" in mesh_out.coords, "Missing x node coordinates"
    assert "mesh2d_node_y" in mesh_out.coords, "Missing y node coordinates"
    assert mesh_out.sizes["mesh2d_nNodes"] > 0, "Mesh should contain nodes"

    # Act: With bounds in EPSG:4326, check for proper clipping
    # Convert bounds to WGS84 for input to function
    bounds_geom = gpd.GeoDataFrame(geometry=[box(xmin, ymin, xmax, ymax)], crs=crs)
    bounds_wgs84 = bounds_geom.to_crs(4326).total_bounds

    mesh_clipped = create_mesh2d_from_mesh(
        uds=uds,
        grid_name="test_grid",
        crs=crs,
        bounds=tuple(bounds_wgs84),
    )

    # Assert: Mesh is clipped correctly
    mesh_gdf = mesh_clipped.ugrid.to_geodataframe()
    bbox = box(xmin, ymin, xmax, ymax)
    assert mesh_gdf.geometry.within(bbox).any(), (
        "Clipped mesh is not within expected bounds"
    )

    # Assert: Clipped mesh has fewer or equal nNodes
    assert mesh_clipped.sizes["mesh2d_nNodes"] <= mesh_out.sizes["mesh2d_nNodes"], (
        "Clipped mesh has more nodes than original"
    )

    # Act & Assert: Missing grid_name error
    uds_multi = xu.UgridDataset(mesh_orig.to_dataset())
    uds_multi.grids[0].name = "grid_1"
    mesh2 = xu.Ugrid2d.from_geodataframe(geom)
    mesh2.name = "grid_2"
    uds_multi.grids.append(mesh2)

    with pytest.raises(ValueError, match="Mesh file contains several grids"):
        create_mesh2d_from_mesh(uds=uds_multi, grid_name=None, crs=crs)

    # Act & Assert: Invalid grid name
    with pytest.raises(ValueError, match="Mesh file does not contain grid"):
        create_mesh2d_from_mesh(uds=uds, grid_name="nonexistent", crs=crs)

    # Act & Assert: Non-2D topology
    mock_1d_grid = MagicMock(spec=xu.Ugrid2d)
    mock_1d_grid.name = "test_grid"
    mock_1d_grid.topology_dimension = 1
    mock_1d_grid.to_dataset.return_value = xr.Dataset()
    mock_1d_grid.crs = CRS.from_epsg(crs)

    mock_uds = MagicMock()
    mock_uds.grids = [mock_1d_grid]

    with pytest.raises(
        ValueError, match="Grid in mesh file for create_mesh2d is not 2D"
    ):
        create_mesh2d_from_mesh(uds=mock_uds, grid_name="test_grid", crs=crs)
