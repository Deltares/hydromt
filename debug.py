# import hydromt and setup logging
import geopandas as gpd
import pandas as pd
import hydromt
from hydromt.log import setuplog
import numpy as np

logger = setuplog("read vector data", log_level=10)

data_catalog = hydromt.DataCatalog(logger=logger, data_libs=["artifact_data"])
gdf = data_catalog.get_geodataframe("gadm_level3")
gdf_subset = data_catalog.get_geodataframe(
    "gadm_level3", bbox=gdf[:5].total_bounds, variables=["GID_0", "NAME_3"]
)
fn = "tmpdir/xy.csv"
df = pd.DataFrame(
    columns=["x_centroid", "y"],
    data=np.vstack([gdf_subset.centroid.x, gdf_subset.centroid.y]).T,
)
df["name"] = gdf_subset["NAME_3"]
df.to_csv(fn)  # write to file

data_source = {
    "GADM_level3_centroids": {
        "path": fn,
        "data_type": "GeoDataFrame",
        "driver": "vector_table",
        "crs": 4326,
        "driver_kwargs": {"x_dim": "x_centroid"},
    }
}
data_catalog.from_dict(data_source)
data_catalog["GADM_level3_centroids"]
breakpoint()
data_catalog.get_geodataframe("GADM_level3_centroids")
