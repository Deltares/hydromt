"""DataSource class for the GeoDataset type."""

from datetime import datetime
from logging import Logger, getLogger
from os.path import basename, splitext
from typing import Any, ClassVar, Dict, List, Literal, Optional, Union, cast

import pandas as pd
import xarray as xr
from fsspec import filesystem
from pydantic import Field
from pyproj import CRS
from pyproj.exceptions import CRSError
from pystac import Asset as StacAsset
from pystac import Catalog as StacCatalog
from pystac import Item as StacItem
from pystac import MediaType

from hydromt._typing import (
    Bbox,
    ErrorHandleMethod,
    Geom,
    NoDataStrategy,
    StrPath,
    TimeRange,
    TotalBounds,
)
from hydromt._typing.type_def import GeomBuffer, Predicate
from hydromt.data_catalog.adapters.geodataset import GeoDatasetAdapter
from hydromt.data_catalog.drivers.geodataset.geodataset_driver import GeoDatasetDriver
from hydromt.data_catalog.sources.data_source import DataSource
from hydromt.gis.utils import parse_geom_bbox_buffer

logger: Logger = getLogger(__name__)


class GeoDatasetSource(DataSource):
    """DataSource class for the GeoDatasetSource type."""

    data_type: ClassVar[Literal["GeoDataset"]] = "GeoDataset"
    driver: GeoDatasetDriver
    data_adapter: GeoDatasetAdapter = Field(default_factory=GeoDatasetAdapter)

    def read_data(
        self,
        *,
        mask: Optional[Geom] = None,
        predicate: Predicate = "intersects",
        variables: Optional[List[str]] = None,
        time_range: Optional[TimeRange] = None,
        single_var_as_array: bool = True,
        handle_nodata: NoDataStrategy = NoDataStrategy.RAISE,
        logger: Logger = logger,
    ) -> Optional[Union[xr.Dataset, xr.DataArray]]:
        """
        Read data from this source.

        Args:
        """
        self.mark_as_used()

        # Transform time_range and variables to match the data source
        tr = self.data_adapter.to_source_timerange(time_range)
        vrs = self.data_adapter.to_source_variables(variables)

        ds: Optional[xr.Dataset] = self.driver.read(
            self.full_uri,
            time_range=tr,
            variables=vrs,
            metadata=self.metadata,
            handle_nodata=handle_nodata,
        )
        return self.data_adapter.transform(
            ds,
            self.metadata,
            predicate=predicate,
            single_var_as_array=single_var_as_array,
            mask=mask,
            variables=variables,
            time_range=time_range,
            handle_nodata=handle_nodata,
            logger=logger,
        )

    def to_file(
        self,
        file_path: StrPath,
        *,
        driver_override: Optional[GeoDatasetDriver] = None,
        bbox: Optional[Bbox] = None,
        mask: Optional[Geom] = None,
        buffer: GeomBuffer = 0,
        predicate: Predicate = "intersects",
        variables: Optional[List[str]] = None,
        time_range: Optional[TimeRange] = None,
        single_var_as_array: bool = True,
        handle_nodata: NoDataStrategy = NoDataStrategy.RAISE,
        logger: Logger = logger,
        **kwargs,
    ) -> Optional["GeoDatasetSource"]:
        """
        Write the GeoDatasetSource to a local file.

        args:
        """
        if not self.driver.supports_writing:
            raise RuntimeError(
                f"driver {self.driver.__class__.__name__} does not support writing. please use a differnt driver "
            )
        if bbox is not None or (mask is not None and buffer > 0):
            mask = parse_geom_bbox_buffer(mask, bbox, buffer)
        ds: Optional[Union[xr.Dataset, xr.DataArray]] = self.read_data(
            mask=mask,
            predicate=predicate,
            variables=variables,
            single_var_as_array=single_var_as_array,
            time_range=time_range,
            handle_nodata=handle_nodata,
            logger=logger,
        )
        if ds is None:  # handle_nodata == ignore
            return None

        # update driver based on local path
        update: Dict[str, Any] = {"uri": file_path}

        if driver_override:
            driver: GeoDatasetDriver = driver_override
        else:
            # use local filesystem
            driver: GeoDatasetDriver = self.driver.model_copy(
                update={"filesystem": filesystem("local")}
            )
        update.update({"driver": driver})

        driver.write(
            file_path,
            ds,
            **kwargs,
        )

        return self.model_copy(update=update)

    def get_bbox(self, crs: Optional[CRS] = None, detect: bool = True) -> TotalBounds:
        """Return the bounding box and espg code of the dataset.

        if the bounding box is not set and detect is True,
        :py:meth:`hydromt.GeoDatasetAdapter.detect_bbox` will be used to detect it.

        Parameters
        ----------
        detect: bool, Optional
            whether to detect the bounding box if it is not set. If False, and it's not
            set None will be returned.

        Returns
        -------
        bbox: Tuple[np.float64,np.float64,np.float64,np.float64]
            the bounding box coordinates of the data. coordinates are returned as
            [xmin,ymin,xmax,ymax]
        crs: int
            The ESPG code of the CRS of the coordinates returned in bbox
        """
        bbox = self.metadata.extent.get("bbox", None)
        crs = cast(int, crs)
        if bbox is None and detect:
            bbox, crs = self.detect_bbox()

        return bbox, crs

    def get_time_range(
        self,
        detect: bool = True,
    ) -> TimeRange:
        """Detect the time range of the dataset.

        if the time range is not set and detect is True,
        :py:meth:`hydromt.GeoDatasetAdapter.detect_time_range` will be used
        to detect it.


        Parameters
        ----------
        detect: bool, Optional
            whether to detect the time range if it is not set. If False, and it's not
            set None will be returned.

        Returns
        -------
        range: Tuple[np.datetime64, np.datetime64]
            A tuple containing the start and end of the time dimension. Range is
            inclusive on both sides.
        """
        time_range = self.metadata.extent.get("time_range", None)
        if time_range is None and detect:
            time_range = self.detect_time_range()

        return time_range

    def detect_bbox(
        self,
        ds: Optional[xr.Dataset] = None,
    ) -> TotalBounds:
        """Detect the bounding box and crs of the dataset.

        If no dataset is provided, it will be fetched according to the settings in the
        adapter. also see :py:meth:`hydromt.GeoDatasetAdapter.get_data`. the
        coordinates are in the CRS of the dataset itself, which is also returned
        alongside the coordinates.


        Parameters
        ----------
        ds: xr.Dataset, xr.DataArray, Optional
            the dataset to detect the bounding box of.
            If none is provided, :py:meth:`hydromt.GeoDatasetAdapter.get_data`
            will be used to fetch the it before detecting.

        Returns
        -------
        bbox: Tuple[np.float64,np.float64,np.float64,np.float64]
            the bounding box coordinates of the data. coordinates are returned as
            [xmin,ymin,xmax,ymax]
        crs: int
            The ESPG code of the CRS of the coordinates returned in bbox
        """
        if ds is None:
            ds = self.read_data()
        crs = ds.vector.crs.to_epsg()
        bounds = ds.vector.bounds

        return bounds, crs

    def detect_time_range(
        self,
        ds: Optional[xr.Dataset] = None,
    ) -> TimeRange:
        """Detect the temporal range of the dataset.

        If no dataset is provided, it will be fetched accodring to the settings in the
        addapter. also see :py:meth:`hydromt.GeoDatasetAdapter.get_data`.

        Parameters
        ----------
        ds: xr.Dataset, xr.DataArray, Optional
            the dataset to detect the time range of. It must have a time dimentsion set.
            If none is provided, :py:meth:`hydromt.GeoDatasetAdapter.get_data`
            will be used to fetch the it before detecting.

        Returns
        -------
        range: Tuple[np.datetime64, np.datetime64]
            A tuple containing the start and end of the time dimension. Range is
            inclusive on both sides.
        """
        if ds is None:
            ds = self.read_data()
        return (
            ds[ds.vector.time_dim].min().values,
            ds[ds.vector.time_dim].max().values,
        )

    def to_stac_catalog(
        self,
        on_error: ErrorHandleMethod = ErrorHandleMethod.COERCE,
    ) -> Optional[StacCatalog]:
        """
        Convert a geodataset into a STAC Catalog representation.

        The collection will contain an asset for each of the associated files.


        Parameters
        ----------
        - on_error (str, optional): The error handling strategy.
          Options are: "raise" to raise an error on failure, "skip" to skip
          the dataset on failure, and "coerce" (default) to set default
          values on failure.

        Returns
        -------
        - Optional[StacCatalog]: The STAC Catalog representation of the dataset, or
          None if the dataset was skipped.
        """
        try:
            bbox, crs = self.get_bbox(detect=True)
            bbox = list(bbox)
            start_dt, end_dt = self.get_time_range(detect=True)
            start_dt = pd.to_datetime(start_dt)
            end_dt = pd.to_datetime(end_dt)
            props = {**self.metadata.model_dump(exclude_none=True), "crs": crs}
            ext = splitext(self.uri)[-1]
            if ext in [".nc", ".vrt"]:
                media_type = MediaType.HDF5
            else:
                raise RuntimeError(
                    f"Unknown extension: {ext} cannot determine media type"
                )
        except (IndexError, KeyError, CRSError) as e:
            if on_error == ErrorHandleMethod.SKIP:
                logger.warning(
                    "Skipping {name} during stac conversion because"
                    "because detecting spacial extent failed."
                )
                return
            elif on_error == ErrorHandleMethod.COERCE:
                bbox = [0.0, 0.0, 0.0, 0.0]
                props = self.metadata
                start_dt = datetime(1, 1, 1)
                end_dt = datetime(1, 1, 1)
                media_type = MediaType.JSON
            else:
                raise e

        stac_catalog = StacCatalog(
            self.name,
            description=self.name,
        )
        stac_item = StacItem(
            self.name,
            geometry=None,
            bbox=bbox,
            properties=props,
            datetime=None,
            start_datetime=start_dt,
            end_datetime=end_dt,
        )
        stac_asset = StacAsset(str(self.uri), media_type=media_type)
        base_name = basename(self.uri)
        stac_item.add_asset(base_name, stac_asset)

        stac_catalog.add_item(stac_item)
        return stac_catalog
