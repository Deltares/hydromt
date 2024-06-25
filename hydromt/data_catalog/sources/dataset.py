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
    ErrorHandleMethod,
    NoDataStrategy,
    StrPath,
    TimeRange,
)
from hydromt.data_catalog.adapters.dataset import DatasetAdapter
from hydromt.data_catalog.drivers import DatasetDriver
from hydromt.data_catalog.sources.data_source import DataSource

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
        variables: Optional[List[str]] = None,
        time_range: Optional[TimeRange] = None,
        handle_nodata: NoDataStrategy = NoDataStrategy.RAISE,
        single_var_as_array: bool = True,
    ) -> Union[xr.Dataset, xr.DataArray]:
        """
        Read data from this source.

        Args:
        """
        self._used = True

        # Transform time_range and variables to match the data source
        tr = self.data_adapter.to_source_timerange(time_range)
        vrs = self.data_adapter.to_source_variables(variables)

        ds: xr.Dataset = self.driver.read(
            self.full_uri,
            time_range=tr,
            variables=vrs,
            metadata=self.metadata,
            handle_nodata=handle_nodata,
        )
        return self.data_adapter.transform(
            ds,
            self.metadata,
            variables=variables,
            time_range=time_range,
            single_var_as_array=single_var_as_array,
        )

    def to_file(
        self,
        file_path: StrPath,
        *,
        driver_override: Optional[DatasetDriver] = None,
        time_range: Optional[TimeRange] = None,
        handle_nodata: NoDataStrategy = NoDataStrategy.RAISE,
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

    def detect_time_range(
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
            ds = self.read_data()

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
            start_dt, end_dt = self.get_time_range(detect=True)
            start_dt = pd.to_datetime(start_dt)
            end_dt = pd.to_datetime(end_dt)
            props = {**self.metadata.model_dump(exclude_none=True, exclude_unset=True)}
            ext = splitext(self.full_uri)[-1]
            if ext == ".nc":
                media_type = MediaType.HDF5
            elif ext == ".zarr":
                raise ValueError("STAC does not support zarr datasets")
            else:
                raise ValueError(
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
                props = self.metadata.model_dump(exclude_none=True, exclude_unset=True)
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
            bbox=None,
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
