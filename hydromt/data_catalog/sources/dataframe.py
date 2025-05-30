"""DataSource class for the DataFrame type."""

from logging import Logger, getLogger
from typing import Any, ClassVar, Dict, List, Literal, Optional

import pandas as pd
from fsspec import filesystem
from pydantic import Field
from pystac import Catalog as StacCatalog

from hydromt._typing import (
    NoDataStrategy,
    StrPath,
    TimeRange,
)
from hydromt.data_catalog.adapters import DataFrameAdapter
from hydromt.data_catalog.drivers import DataFrameDriver
from hydromt.data_catalog.sources import DataSource

logger: Logger = getLogger(__name__)


class DataFrameSource(DataSource):
    """
    DataSource for DataFrames.

    Reads and validates DataCatalog entries.
    """

    data_type: ClassVar[Literal["DataFrame"]] = "DataFrame"
    _fallback_driver_read: ClassVar[str] = "pandas"
    _fallback_driver_write: ClassVar[str] = "pandas"
    driver: DataFrameDriver
    data_adapter: DataFrameAdapter = Field(default_factory=DataFrameAdapter)

    def read_data(
        self,
        *,
        variables: Optional[List[str]] = None,
        time_range: Optional[TimeRange] = None,
        predicate: str = "intersects",
        handle_nodata: NoDataStrategy = NoDataStrategy.RAISE,
    ) -> Optional[pd.DataFrame]:
        """Use the resolver, driver, and data adapter to read and harmonize the data."""
        self._mark_as_used()

        tr: TimeRange = self.data_adapter._to_source_timerange(time_range)
        vrs: Optional[List[str]] = self.data_adapter._to_source_variables(variables)

        uris: List[str] = self.uri_resolver.resolve(
            self.full_uri,
            variables=vrs,
            time_range=tr,
            handle_nodata=handle_nodata,
        )

        df: pd.DataFrame = self.driver.read(
            uris,
            variables=vrs,
            time_range=tr,
            handle_nodata=handle_nodata,
        )

        return self.data_adapter.transform(
            df,
            self.metadata,
            variables=variables,
            time_range=time_range,
            handle_nodata=handle_nodata,
        )

    def to_file(
        self,
        file_path: StrPath,
        *,
        driver_override: Optional[DataFrameDriver] = None,
        variables: Optional[List[str]] = None,
        time_range: Optional[TimeRange] = None,
        handle_nodata: NoDataStrategy = NoDataStrategy.RAISE,
        **kwargs,
    ) -> "DataFrameSource":
        """
        Write the DataFrameSource to a local file.

        args:
        """
        if not driver_override and not self.driver.supports_writing:
            # default to fallback driver
            driver = DataFrameDriver.model_validate(self._fallback_driver_write)
        elif driver_override:
            if not driver_override.supports_writing:
                raise RuntimeError(
                    f"driver: '{driver_override.name}' does not support writing data."
                )
            driver: DataFrameDriver = driver_override
        else:
            # use local filesystem
            driver: DataFrameDriver = self.driver.model_copy(
                update={"filesystem": filesystem("local")}
            )
        df: Optional[pd.DataFrame] = self.read_data(
            variables=variables, time_range=time_range, handle_nodata=handle_nodata
        )
        if df is None:  # handle_nodata == ignore
            return None

        # driver can return different path if file ext changes
        dest_path: str = driver.write(
            file_path,
            df,
            **kwargs,
        )

        # update source and its driver based on local path
        update: Dict[str, Any] = {"uri": dest_path, "root": None, "driver": driver}

        return self.model_copy(update=update)

    def to_stac_catalog(
        self,
        handle_nodata: NoDataStrategy = NoDataStrategy.IGNORE,
    ) -> Optional[StacCatalog]:
        """
        Convert a dataframe into a STAC Catalog representation.

        The collection will contain an asset for each of the associated files.


        Parameters
        ----------
        - handle_nodata (str, optional): The error handling strategy.
          Options are: "raise" to raise an error on failure, "skip" to skip the
          dataframe on failure, and "coerce" (default) to set default values on failure.

        Returns
        -------
        - Optional[StacCatalog]: The STAC Catalog representation of the dataframe, or
          None if the dataset was skipped.
        """
        if handle_nodata == NoDataStrategy.IGNORE:
            logger.warning(
                f"Skipping {self.name} during stac conversion because"
                "because detecting temporal extent failed."
            )
            return None
        else:
            raise NotImplementedError(
                "DataFrameSource does not support full stac conversion as it lacks"
                " spatio-temporal dimensions"
            )
