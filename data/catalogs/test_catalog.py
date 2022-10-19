# -*- coding: utf-8 -*-
"""Script to test a data catalog (not part of CI)

To test the full catalog run: 
  python test_catalog.py <catalog filename>

To test specific sources run:
  python test_catalog.py <catalog filename> <source_name1>,<source_name2>
"""

import sys
from typing import Union, Optional, Callable, Any
import os
import geopandas as gpd
import time
from hydromt import DataCatalog
from hydromt.log import setuplog
from hydromt.data_adapter import (
    RasterDatasetAdapter,
    GeoDataFrameAdapter,
    GeoDatasetAdapter,
)

# from https://www.fuzzingbook.org/html/Timeout.html
class GenericTimeout:
    """Execute a code block raising a timeout."""

    def __init__(self, timeout: Union[int, float]) -> None:
        """
        Constructor. Interrupt execution after `timeout` seconds.
        """

        self.seconds_before_timeout = timeout
        self.original_trace_function: Optional[Callable] = None
        self.end_time: Optional[float] = None

    def check_time(self, frame, event: str, arg) -> Callable:
        """Tracing function"""
        if self.original_trace_function is not None:
            self.original_trace_function(frame, event, arg)

        current_time = time.time()
        if self.end_time and current_time >= self.end_time:
            raise TimeoutError(
                f"Timeout after {self.seconds_before_timeout:.0f} seconds"
            )

        return self.check_time

    def __enter__(self) -> Any:
        """Begin of `with` block"""
        start_time = time.time()
        self.end_time = start_time + self.seconds_before_timeout

        self.original_trace_function = sys.gettrace()
        sys.settrace(self.check_time)
        return self

    def __exit__(self, exc_type, exc_value, tb) -> Optional[bool]:
        """End of `with` block"""
        self.cancel()
        return None

    def cancel(self) -> None:
        """Cancel timeout"""
        sys.settrace(self.original_trace_function)


def test_source(data_cat, name, **kwargs):
    # TODO: add available time slice to catalog to limit reading ..
    source = data_cat[name]
    bbox = [-1, 11.5, 0, 12.5]
    if isinstance(source, RasterDatasetAdapter):
        ds = data_cat.get_rasterdataset(name, bbox=bbox, buffer=500, **kwargs)
        # load slice of data
        if ds.raster.dim0 is not None:
            ds = ds.isel({ds.raster.dim0: -1}).squeeze()
        ds.isel({ds.raster.x_dim: slice(-50), ds.raster.y_dim: slice(-50)}).load()
    elif isinstance(source, GeoDataFrameAdapter):
        gdf = data_cat.get_geodataframe(name, bbox=bbox, **kwargs)
        assert isinstance(gdf, gpd.GeoDataFrame)
    elif isinstance(source, GeoDatasetAdapter):
        ds = data_cat.get_geodataset(name, bbox=bbox, **kwargs)
        # load slice of data
        for coord in ds.coords:  # shouldn't that be for instead of if?
            if coord == ds.vector.index_dim:
                ds = ds.isel({coord: slice(-50)})
            else:
                ds = ds.isel({coord: -1}).squeeze()
        ds = ds.load()  # not used if outside loop, musn't it be inside ine
    return


if __name__ == "__main__":
    logger = setuplog(
        "test_catalog", path="./test_catalog.log", append=False, log_level=10
    )
    timeout_sec = 60

    # read data catalog
    fn = sys.argv[1]
    if not os.path.isabs(fn):
        fn = os.path.join(os.path.dirname(__file__), fn)
    data_cat = DataCatalog(fn, logger=logger)

    # get names of data sources to test; by default all
    if len(sys.argv) == 3:
        sources = sys.argv[2].split(",")
    else:
        sources = data_cat.sources

    # test sources
    failed = []
    for name in sources:

        try:
            with GenericTimeout(timeout_sec):
                t0 = time.time()
                test_source(data_cat, name)
                dt = time.time() - t0
                logger.info(f"{name}: PASSED ({dt:0.1f} sec)")
        except Exception as e:
            logger.error(f"{name}: FAILED")
            logger.error(e)
            failed.append(name)

    if len(failed) > 0:
        logger.error(f"{len(failed)}/{len(sources)} data sources failed: {failed}")
    else:
        logger.info(f"{len(sources)} sources succesful")
