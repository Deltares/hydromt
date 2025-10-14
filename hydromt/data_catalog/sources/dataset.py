"""DataSource class for the Dataset type."""

import logging
from os.path import basename, splitext
from typing import Any, ClassVar, Dict, List, Literal, Optional, Union

import xarray as xr
from pydantic import Field
from pyproj.exceptions import CRSError
from pystac import Asset as StacAsset
from pystac import Catalog as StacCatalog
from pystac import Item as StacItem
from pystac import MediaType

from hydromt.data_catalog.adapters.dataset import DatasetAdapter
from hydromt.data_catalog.drivers import DatasetDriver
from hydromt.data_catalog.sources.data_source import DataSource
from hydromt.error import NoDataStrategy
from hydromt.typing import (
    StrPath,
    TimeRange,
)
from hydromt.typing.fsspec_types import FSSpecFileSystem

logger = logging.getLogger(__name__)


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
        """Use the resolver, driver, and data adapter to read and harmonize the data.

        Parameters
        ----------
        variables : Optional[List[str]], optional
            Names of variables to return, or all if None, by default None
            variables queried for, by default None
        time_range : Optional[TimeRange], optional
            left-inclusive start end time of the data, by default None
        handle_nodata : NoDataStrategy, optional
            how to react when no data is found, by default NoDataStrategy.RAISE
        single_var_as_array : bool, optional
            _description_, by default True

        Returns
        -------
        Union[xr.Dataset, xr.DataArray]
            harmonized data
        """
        self._mark_as_used()
        self._log_start_read_data()

        # Transform time_range and variables to match the data source
        tr = self.data_adapter._to_source_timerange(time_range)
        vrs = self.data_adapter._to_source_variables(variables)

        uris: List[str] = self.uri_resolver.resolve(
            self.full_uri,
            time_range=tr,
            variables=vrs,
            handle_nodata=handle_nodata,
        )

        ds: xr.Dataset = self.driver.read(
            uris,
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
                update={"filesystem": FSSpecFileSystem.create("local")}
            )

        ds: Optional[xr.Dataset] = self.read_data(
            time_range=time_range,
            handle_nodata=handle_nodata,
        )
        if ds is None:  # handle_nodata == ignore
            return None

        # driver can return different path if file ext changes
        dest_path: str = driver.write(
            file_path,
            ds,
            **kwargs,
        )

        # update driver based on local path
        update: Dict[str, Any] = {"uri": dest_path, "root": None, "driver": driver}

        return self.model_copy(update=update)

    def _detect_time_range(
        self,
        *,
        strict: bool = False,
        ds: Union[xr.DataArray, xr.Dataset] = None,
    ) -> TimeRange | None:
        """Get the temporal range of a dataset.

        Parameters
        ----------
        ds : Optional[xr.DataArray  |  xr.Dataset]
            The dataset to detect the time range of. It must have a time dimension set.
            If none is provided, :py:meth:`hydromt.DatasetAdapter.get_data`
            will be used to fetch the it before detecting.

        Returns
        -------
        range: TimeRange, optional
            Instance containing the start and end of the time dimension. Range is
            inclusive on both sides. None if the dataset has no time dimension.
        """
        if ds is None:
            ds = self.read_data()

        try:
            return TimeRange(
                start=ds.time[0].values,
                end=ds.time[-1].values,
            )
        except AttributeError:
            raise AttributeError("Dataset has no dimension called 'time'")

    def to_stac_catalog(
        self,
        handle_nodata: NoDataStrategy = NoDataStrategy.IGNORE,
    ) -> Optional[StacCatalog]:
        """
        Convert a dataset into a STAC Catalog representation.

        The collection will contain an asset for each of the associated files.


        Parameters
        ----------
        - handle_nodata (str, optional): The error handling strategy.
          Options are: "raise" to raise an error on failure, "skip" to skip the
          dataset on failure.

        Returns
        -------
        - Optional[StacCatalog]: The STAC Catalog representation of the dataset, or None
          if the dataset was skipped.
        """
        try:
            time_range = self.get_time_range(detect=True)
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
            if handle_nodata == NoDataStrategy.IGNORE:
                logger.warning(
                    "Skipping {name} during stac conversion because"
                    "because detecting spacial extent failed."
                )
                return
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
            start_datetime=time_range.start,
            end_datetime=time_range.end,
        )
        stac_asset = StacAsset(str(self.full_uri), media_type=media_type)
        base_name = basename(self.full_uri)
        stac_item.add_asset(base_name, stac_asset)

        stac_catalog.add_item(stac_item)
        return stac_catalog
