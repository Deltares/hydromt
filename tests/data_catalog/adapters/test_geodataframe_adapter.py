import geopandas as gpd
import pytest

from hydromt.data_catalog.adapters.geodataframe import GeoDataFrameAdapter
from hydromt.error import NoDataException
from hydromt.typing import SourceMetadata


class TestGeodataFrameAdapter:
    def test_transform_empty_gdf(
        self,
        geodf: gpd.GeoDataFrame,
    ):
        adapter = GeoDataFrameAdapter()

        empty_gdf = geodf.iloc[0:0]

        with pytest.raises(NoDataException, match="No data was read from source"):
            adapter.transform(
                empty_gdf,
                metadata=SourceMetadata(),
            )

    def test_set_crs(
        self,
        geodf: gpd.GeoDataFrame,
        caplog: pytest.LogCaptureFixture,
    ):
        adapter = GeoDataFrameAdapter()

        gdf_no_crs = geodf.copy()

        gdf_with_crs = adapter._set_crs(gdf_no_crs, crs=4326)

        assert gdf_with_crs.crs.to_epsg() == 4326

        gdf_no_crs.set_crs(None, allow_override=True, inplace=True)
        with pytest.raises(
            ValueError, match="GeoDataFrame: CRS not defined in data catalog or data."
        ):
            adapter._set_crs(gdf_no_crs, crs=None)
        caplog.set_level("WARNING")
        adapter._set_crs(geodf, crs=3857)
        assert (
            "GeoDataFrame : CRS from data catalog does not match CRS of"
            " data. The original CRS will be used. Please check your data catalog."
        ) in caplog.text
