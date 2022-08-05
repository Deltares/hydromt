# -*- coding: utf-8 -*-
"""Script to test a data catalog (not part of CI)

To test the full catalog run: 
  python test_catalog.py <catalog filename>

To test specific sources run:
  python test_catalog.py <catalog filename> <source_name1>,<source_name2>
"""

import sys
import os
import geopandas as gpd
import time
from hydromt import DataCatalog
from hydromt.data_adapter import (
    RasterDatasetAdapter,
    GeoDataFrameAdapter,
    GeoDatasetAdapter,
)
from hydromt.log import setuplog

if __name__ == "__main__":
    logger = setuplog("test_catalog", log_level=10)

    # read data catalog
    fn = sys.argv[1]
    if not os.path.isabs(fn):
        fn = os.path.join(os.path.dirname(__file__), fn)
    cat = DataCatalog(fn, logger=logger)

    # get names of data sources to test; by default all
    if len(sys.argv) == 3:
        sources = sys.argv[2].split(",")
    else:
        sources = cat.sources

    for name in sources:
        source = cat.sources[name]
        kwargs = {}
        # TODO: add available time slice to catalog to limit reading ..
        try:
            t0 = time.time()
            if isinstance(source, RasterDatasetAdapter):
                ds = cat.get_rasterdataset(name)
                # test if correct spatial dims
                ds.raster.set_spatial_dims()
                # load slice of data
                if ds.raster.dim0 is not None:
                    ds = ds.isel({ds.raster.dim0: -1})
                ds[-100:, -100:].load()  # always 2D data
            elif isinstance(source, GeoDataFrameAdapter):
                if source.driver == "vector":
                    kwargs.update(rows=10)  # test for small slice
                gdf = cat.get_geodataframe(name, **kwargs)
                assert isinstance(gdf, gpd.GeodataFrame)
            elif isinstance(source, GeoDatasetAdapter):
                ds = cat.get_geodataset(name)
                # test if correct spatial dims
                ds.vector.set_spatial_dims()
                # load slice of data
                if coord in ds.coords:
                    if coord == ds.vector.index_dim:
                        continue
                    ds = ds.isel({coord: -1})
                ds[-100:].load()  # always 1D data
            dt = time.time() - t0
            logger.info(f"{name}: PASSED ({dt:0.1f} sec)")
        except Exception as e:
            logger.error(f"{name}: FAILED")
            logger.error(e)
