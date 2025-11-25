"""URI Resolver responsible for finding the data using the URI in the Data Catalog."""

import logging
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional

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
    options: Dict[str, Any] = Field(default_factory=dict)

    @abstractmethod
    def resolve(
        self,
        uri: str,
        *,
        time_range: Optional[TimeRange] = None,
        mask: Optional[Geom] = None,
        variables: Optional[List[str]] = None,
        zoom: Optional[Zoom] = None,
        metadata: Optional[SourceMetadata] = None,
        handle_nodata: NoDataStrategy = NoDataStrategy.RAISE,
    ) -> List[str]:
        """Resolve a single uri to multiple uris.

        Parameters
        ----------
        uri : str
            Unique Resource Identifier
        time_range : Optional[TimeRange], optional
            left-inclusive start end time of the data, by default None
        mask : Optional[Geom], optional
            A geometry defining the area of interest, by default None
        zoom : Optional[Zoom], optional
            zoom of the dataset, by default None
        variables : Optional[List[str]], optional
            Names of variables to return, or all if None, by default None
        metadata: Optional[SourceMetadata], optional
            Metadata of DataSource.
        handle_nodata : NoDataStrategy, optional
            how to react when no data is found, by default NoDataStrategy.RAISE

        Returns
        -------
        List[str]
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
