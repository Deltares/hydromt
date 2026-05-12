"""URI Resolver responsible for finding the data using the URI in the Data Catalog."""

import logging
from abc import ABC, abstractmethod

from pydantic import ConfigDict, Field

from hydromt._abstract_base import AbstractBaseModel
from hydromt.error import NoDataStrategy
from hydromt.plugins import PLUGINS
from hydromt.typing import (
    FSSpecFileSystem,
    Geom,
    SourceMetadata,
    TimeRange,
    Zoom,
)

logger = logging.getLogger(__name__)


class URIResolver(AbstractBaseModel, ABC):
    """URI Resolver responsible for finding the data using the URI in the Data Catalog."""

    model_config = ConfigDict(extra="forbid")
    filesystem: FSSpecFileSystem = Field(default_factory=FSSpecFileSystem)
    options: dict = Field(default_factory=dict)

    @abstractmethod
    def resolve(
        self,
        uri: str,
        *,
        time_range: TimeRange | None = None,
        mask: Geom | None = None,
        variables: list[str] | None = None,
        zoom: Zoom | None = None,
        metadata: SourceMetadata | None = None,
        handle_nodata: NoDataStrategy = NoDataStrategy.RAISE,
    ) -> list[str]:
        """Resolve a single uri to multiple uris.

        Parameters
        ----------
        uri : str
            Unique Resource Identifier
        time_range : TimeRange | None, optional
            left-inclusive start end time of the data, by default None
        mask : Geom | None, optional
            A geometry defining the area of interest, by default None
        zoom : Zoom | None  , optional
            zoom of the dataset, by default None
        variables : list[str] | None, optional
            Names of variables to return, or all if None, by default None
        metadata: SourceMetadata | None, optional
            Metadata of DataSource.
        handle_nodata : NoDataStrategy, optional
            how to react when no data is found, by default NoDataStrategy.RAISE

        Returns
        -------
        list[str]
            a list of expanded uris

        Raises
        ------
        NoDataException
            when no data is found and `handle_nodata` is `NoDataStrategy.RAISE`
        """
        ...

    @classmethod
    def _load_plugins(cls):
        """Load URIResolver plugins."""
        plugins = PLUGINS.uri_resolver_plugins.keys()
        logger.debug(f"loaded {cls.__name__} plugins: {', '.join(plugins)}")
