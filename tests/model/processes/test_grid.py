from pathlib import Path

import geopandas as gpd
import numpy as np
import pytest
import xarray as xr
from shapely.geometry import box

from hydromt.data_catalog.data_catalog import DataCatalog
from hydromt.model.model import Model
from hydromt.model.processes.grid import (
    create_grid_from_region,
    grid_from_constant,
    grid_from_geodataframe,
    grid_from_raster_reclass,
    grid_from_rasterdataset,
)


@pytest.mark.integration
def test_create_grid_from_region(data_catalog: DataCatalog):
    ds = create_grid_from_region(
        region={"subbasin": [12.319, 46.320], "uparea": 50},
        res=1000,
        crs="utm",
        data_catalog=data_catalog,
        hydrography_path="merit_hydro",
        basin_index_path="merit_hydro_index",
    )
    assert not np.all(ds.raster.mask_nodata().values is True)
    assert ds.raster.shape == (47, 61)


def test_create_grid_from_region_bbox_rotated():
    da = create_grid_from_region(
        region={"bbox": [12.65, 45.50, 12.85, 45.60]},
        res=0.05,
        crs=4326,
        region_crs=4326,
        rotated=True,
        add_mask=True,
    )

    assert "xc" in da.coords
    assert da.raster.y_dim == "y"
    assert np.isclose(da.raster.res[0], 0.05)


def test_create_grid_from_region_bbox():
    bbox = [12.05, 45.30, 12.85, 45.65]

    da = create_grid_from_region(
        region={"bbox": bbox},
        res=0.05,
        add_mask=True,
        align=True,
    )

    assert da.raster.dims == ("y", "x")
    assert da.raster.shape == (7, 16)
    assert np.all(np.round(da.raster.bounds, 2) == bbox)


def test_create_grid_from_region_raise_errors():
    # Wrong region kind
    with pytest.raises(ValueError, match="Region for grid must be of kind"):
        create_grid_from_region(region={"vector_model": "test_model"})

    # bbox
    bbox = [12.05, 45.30, 12.85, 45.65]
    with pytest.raises(
        ValueError, match="res argument required for kind 'bbox', 'geom'"
    ):
        create_grid_from_region(region={"bbox": bbox})


def test_grid_from_constant(demda):
    demda.name = "demda"
    da = grid_from_constant(grid_like=demda, constant=0.01, name="demda")

    assert da.name == "demda"
    assert da.shape == demda.shape
    assert np.all(da == 0.01)


def test_grid_from_rasterdataset(demda):
    demda.name = "demda"
    demda = demda.to_dataset()
    rename = {"demda": "new_var"}
    ds = grid_from_rasterdataset(
        grid_like=demda,
        ds=demda,
        variables=["variable1", "variable2"],
        fill_method="mean",
        reproject_method="nearest",
        mask_name="mask",
        rename=rename,
    )

    assert all(ds == demda)
    dvars_expected = [
        rename[var] if var in rename else var for var in demda.data_vars.keys()
    ]
    assert "demda" in demda.data_vars.keys()
    assert all([x in ds.data_vars.keys() for x in dvars_expected])


def test_grid_from_raster_reclass(demda: xr.DataArray, tmp_path: Path, data_dir: Path):
    demda.name = "name"
    reclass_variables = ["roughness_manning"]
    model = Model(root=tmp_path, data_libs=["artifact_data"], mode="w")

    raster_data = "vito_2015"
    reclass_table_data = data_dir / "vito_mapping.csv"
    variable = "roughness_manning"

    # Read raster data and remapping table
    da = model.data_catalog.get_rasterdataset(
        raster_data,
        geom=model.region,
        buffer=2,
        variables=variable,
    )

    df_vars = model.data_catalog.get_dataframe(
        reclass_table_data, variables=reclass_variables
    )

    ds = grid_from_raster_reclass(
        grid_like=demda,
        da=da,
        fill_method="nearest",
        reproject_method=["average"],
        reclass_table=df_vars,
        reclass_variables=reclass_variables,
    )

    assert all(ds == demda)
    assert all([x in ds.data_vars.keys() for x in reclass_variables])


def test_grid_from_geodataframe(
    data_catalog: DataCatalog,
    demda: xr.DataArray,
):
    demda.name = "name"
    gdf = data_catalog.get_geodataframe("hydro_lakes")
    variables = ["waterbody_id", "Depth_avg"]
    nodata = [-1, -999.0]
    rename = {
        "waterbody_id": "lake_id",
        "Depth_avg": "lake_depth",
    }

    ds = grid_from_geodataframe(
        gdf=gdf,
        grid_like=demda.to_dataset(),
        variables=variables,
        nodata=nodata,
        rasterize_method="value",
        rename=rename,
    )

    assert all(ds == demda)
    expected_vars = rename.values()
    assert all([x in ds.data_vars.keys() for x in expected_vars])

    gdf_no_overlap = gpd.GeoDataFrame(
        data={
            col: [gdf.iloc[0][col]] for col in gdf.columns if col != gdf.geometry.name
        },
        geometry=[
            box(
                ds.x.values.max() + 100,
                ds.y.values.max() + 100,
                ds.x.values.max() + 110,
                ds.y.values.max() + 110,
            )
        ],
        crs=ds.raster.crs,
    )
    result = grid_from_geodataframe(
        gdf=gdf_no_overlap,
        grid_like=demda.to_dataset(),
        variables=variables,
        nodata=nodata,
        rasterize_method="value",
        rename=rename,
    )

    assert result.raster.shape == demda.raster.shape
    assert result.raster.dims == demda.raster.dims

    for old, new in rename.items():
        assert old not in result.data_vars
        assert new in result.data_vars

        # assert that all values are nodata
        assert (
            result.data_vars[new].values == nodata[list(expected_vars).index(new)]
        ).all()
