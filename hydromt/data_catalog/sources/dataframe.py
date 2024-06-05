"""DataSource class for the DataFrame type."""

from datetime import datetime
from logging import Logger, getLogger
from typing import Any, ClassVar, Dict, List, Literal, Optional

import pandas as pd
from fsspec import filesystem
from pydantic import Field
from pystac import Asset as StacAsset
from pystac import Catalog as StacCatalog
from pystac import Item as StacItem

from hydromt._typing import (
    ErrorHandleMethod,
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
        predicate: str = "intersects",  # TODO: https://github.com/Deltares/hydromt/issues/983
        handle_nodata: NoDataStrategy = NoDataStrategy.RAISE,
        logger: Logger = logger,
    ) -> Optional[pd.DataFrame]:
        """Use the driver and data adapter to read and harmonize the data."""
        self.mark_as_used()
        df: pd.DataFrame = self.driver.read(
            self.full_uri,
            variables=variables,
            time_range=time_range,
            metadata=self.metadata,
            handle_nodata=handle_nodata,
            logger=logger,
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
        logger: Logger = logger,
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
            variables=variables,
            time_range=time_range,
            handle_nodata=handle_nodata,
            logger=logger,
        )
        if df is None:  # handle_nodata == ignore
            return None

        # update source and its driver based on local path
        update: Dict[str, Any] = {"uri": file_path, "root": None, "driver": driver}

        driver.write(
            file_path,
            df,
            **kwargs,
        )

        return self.model_copy(update=update)

    def to_stac_catalog(
        self,
        on_error: ErrorHandleMethod = ErrorHandleMethod.COERCE,
    ) -> Optional[StacCatalog]:
        """
        Convert a dataframe into a STAC Catalog representation.

        The collection will contain an asset for each of the associated files.


        Parameters
        ----------
        - on_error (str, optional): The error handling strategy.
          Options are: "raise" to raise an error on failure, "skip" to skip the
          dataframe on failure, and "coerce" (default) to set default values on failure.

        Returns
        -------
        - Optional[StacCatalog]: The STAC Catalog representation of the dataframe, or
          None if the dataset was skipped.
        """
        if on_error == ErrorHandleMethod.SKIP:
            logger.warning(
                f"Skipping {self.name} during stac conversion because"
                "because detecting temporal extent failed."
            )
            return
        elif on_error == ErrorHandleMethod.COERCE:
            stac_catalog = StacCatalog(
                self.name,
                description=self.name,
            )
            stac_item = StacItem(
                self.name,
                geometry=None,
                bbox=[0, 0, 0, 0],
                properties=self.metadata.model_dump(),
                datetime=datetime(1, 1, 1),
            )
            stac_asset = StacAsset(self.full_uri)
            stac_item.add_asset("hydromt_path", stac_asset)

            stac_catalog.add_item(stac_item)
            return stac_catalog
        else:
            raise NotImplementedError(
                "DataFrameSource does not support full stac conversion as it lacks"
                " spatio-temporal dimensions"
            )
