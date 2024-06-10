"""DataSource class for the Dataset type."""

from datetime import datetime
from logging import Logger, getLogger
from os.path import basename, splitext
from typing import Any, ClassVar, Dict, List, Literal, Optional, Union

import pandas as pd
import xarray as xr
from fsspec import filesystem
from pydantic import Field
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
    ZoomLevel,
)
from hydromt.data_catalog.adapters.dataset import DatasetAdapter
from hydromt.data_catalog.drivers import DatasetDriver
from hydromt.data_catalog.sources.data_source import DataSource
from hydromt.gis.gis_utils import parse_geom_bbox_buffer

logger: Logger = getLogger(__name__)


class DatasetSource(DataSource):
    """DataSource class for the Dataset type."""

    data_type: ClassVar[Literal["Dataset"]] = "Dataset"
    _fallback_driver_read: ClassVar[str] = "dataset_xarray"
    _fallback_driver_write: ClassVar[str] = "dataset_xarray"
    driver: DatasetDriver
    data_adapter: DatasetAdapter = Field(default_factory=DatasetAdapter)
    zoom_levels: Optional[Dict[int, float]] = None

    def read_data(
        self,
        *,
        bbox: Optional[Bbox] = None,
        mask: Optional[Geom] = None,
        buffer: float = 0,
        variables: Optional[List[str]] = None,
        time_range: Optional[TimeRange] = None,
        handle_nodata: NoDataStrategy = NoDataStrategy.RAISE,
        logger: Logger = logger,
    ) -> Optional[xr.Dataset]:
        """
        Read data from this source.

        Args:
        """
        self._used = True
        if bbox is not None or (mask is not None and buffer > 0):
            mask = parse_geom_bbox_buffer(mask, bbox, buffer)

        # Transform time_range and variables to match the data source
        tr = self.data_adapter.to_source_timerange(time_range)
        vrs = self.data_adapter.to_source_variables(variables)

        ds: xr.Dataset = self.driver.read(
            self.full_uri,
            mask=mask,
            time_range=tr,
            variables=vrs,
            metadata=self.metadata,
            logger=logger,
            handle_nodata=handle_nodata,
        )
        return self.data_adapter.transform(
            ds,
            self.metadata,
            mask=mask,
            variables=variables,
            time_range=time_range,
            logger=logger,
        )

    def to_file(
        self,
        file_path: StrPath,
        *,
        driver_override: Optional[DatasetDriver] = None,
        bbox: Optional[Bbox] = None,
        mask: Optional[Geom] = None,
        buffer: float = 0.0,
        time_range: Optional[TimeRange] = None,
        zoom_level: Optional[ZoomLevel] = None,
        handle_nodata: NoDataStrategy = NoDataStrategy.RAISE,
        logger: Logger = logger,
        **kwargs,
    ) -> "DatasetSource":
        """
        Write the DatasetSource to a local file.

        args:
        """
        if driver_override is None and not self.driver.supports_writing:
            # default to fallback driver
            driver: DatasetDriver = DatasetDriver.model_validate(
                self._fallback_driver_write
            )
        elif driver_override:
            if not driver_override.supports_writing:
                raise RuntimeError(
                    f"driver: '{driver_override.name}' does not support writing data."
                )
            driver: DatasetDriver = driver_override
        else:
            # use local filesystem
            driver: DatasetDriver = self.driver.model_copy(
                update={"filesystem": filesystem("local")}
            )

        ds: Optional[xr.Dataset] = self.read_data(
            time_range=time_range,
            handle_nodata=handle_nodata,
            logger=logger,
        )
        if ds is None:  # handle_nodata == ignore
            return None

        # update driver based on local path
        update: Dict[str, Any] = {"uri": file_path, "root": None, "driver": driver}

        driver.write(
            file_path,
            ds,
            **kwargs,
        )

        return self.model_copy(update=update)

    def get_time_range(
        self,
        ds: Union[xr.DataArray, xr.Dataset] = None,
    ) -> TimeRange:
        """Get the temporal range of a dataset.

        Parameters
        ----------
        ds : Optional[xr.DataArray  |  xr.Dataset]
            The dataset to detect the time range of. It must have a time dimension set.
            If none is provided, :py:meth:`hydromt.DatasetAdapter.get_data`
            will be used to fetch the it before detecting.


        Returns
        -------
        range: Tuple[np.datetime64, np.datetime64]
            A tuple containing the start and end of the time dimension. Range is
            inclusive on both sides.
        """
        if ds is None:
            ds = self.get_data()

        try:
            return (ds.time[0].values, ds.time[-1].values)
        except AttributeError:
            raise AttributeError("Dataset has no dimension called 'time'")

    def to_stac_catalog(
        self,
        on_error: ErrorHandleMethod = ErrorHandleMethod.COERCE,
    ) -> Optional[StacCatalog]:
        """
        Convert a dataset into a STAC Catalog representation.

        The collection will contain an asset for each of the associated files.


        Parameters
        ----------
        - on_error (str, optional): The error handling strategy.
          Options are: "raise" to raise an error on failure, "skip" to skip the
          dataset on failure, and "coerce" (default) to set default values on failure.

        Returns
        -------
        - Optional[StacCatalog]: The STAC Catalog representation of the dataset, or None
          if the dataset was skipped.
        """
        try:
            bbox, crs = self.get_bbox(detect=True)
            bbox = list(bbox)
            start_dt, end_dt = self.get_time_range(detect=True)
            start_dt = pd.to_datetime(start_dt)
            end_dt = pd.to_datetime(end_dt)
            props = {**self.metadata.model_dump(exclude_none=True), "crs": crs}
            ext = splitext(self.full_uri)[-1]
            if ext == ".nc" or ext == ".vrt":
                media_type = MediaType.HDF5
            elif ext == ".tiff":
                media_type = MediaType.TIFF
            elif ext == ".cog":
                media_type = MediaType.COG
            elif ext == ".png":
                media_type = MediaType.PNG
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
                props = self.data_adapter.meta
                start_dt = datetime(1, 1, 1)
                end_dt = datetime(1, 1, 1)
                media_type = MediaType.JSON
            else:
                raise e

        else:
            # else makes type checkers a bit happier
            stac_catalog = StacCatalog(
                self.name,
                description=self.name,
            )
            stac_item = StacItem(
                self.name,
                geometry=None,
                bbox=list(bbox),
                properties=props,
                datetime=None,
                start_datetime=start_dt,
                end_datetime=end_dt,
            )
            stac_asset = StacAsset(str(self.full_uri), media_type=media_type)
            base_name = basename(self.full_uri)
            stac_item.add_asset(base_name, stac_asset)

            stac_catalog.add_item(stac_item)
            return stac_catalog
