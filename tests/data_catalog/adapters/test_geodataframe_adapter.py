import geopandas as gpd
import numpy as np
import pytest
from shapely import Point

from hydromt.data_catalog.adapters.geodataframe import GeoDataFrameAdapter
from hydromt.error import NoDataException, NoDataStrategy
from hydromt.typing import SourceMetadata


class TestGeodataFrameAdapter:
    def test_transform_empty_gdf(self, geodf: gpd.GeoDataFrame, mocker):
        adapter = GeoDataFrameAdapter()

        empty_gdf = geodf.iloc[0:0]
        mocker.patch.object(
            adapter,
            "_set_nodata",
            return_value=empty_gdf,
        )
        with pytest.raises(
            NoDataException, match="GeoDataFrame has no data after masking"
        ):
            adapter.transform(
                empty_gdf,
                mask=empty_gdf,
                metadata=SourceMetadata(),
                handle_nodata=NoDataStrategy.RAISE,
            )

    def test_set_crs(
        self,
        geodf: gpd.GeoDataFrame,
        caplog: pytest.LogCaptureFixture,
    ):
        adapter = GeoDataFrameAdapter()

        gdf_no_crs = geodf.copy()
        gdf_no_crs.set_crs(None, allow_override=True, inplace=True)
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

    def test__set_nodata(self):
        adapter = GeoDataFrameAdapter()
        gdf = gpd.GeoDataFrame(
            {
                "int_col": [1, -99, 3],
                "geometry": [Point(0, 0), Point(1, 1), Point(2, 2)],
            }
        )
        metadata = SourceMetadata(nodata=-99)

        result = adapter._set_nodata(gdf, metadata)

        assert result["int_col"].dtype == float
        assert np.isnan(result["int_col"].iloc[1])
        assert result["int_col"].iloc[0] == 1.0
        assert result["int_col"].iloc[2] == 3.0
