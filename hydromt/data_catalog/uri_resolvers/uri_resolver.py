"""URI Resolver responsible for finding the data using the URI in the Data Catalog."""

from abc import ABC, abstractmethod
from logging import Logger, getLogger
from typing import Any, Dict, List, Optional

from fsspec.implementations.local import LocalFileSystem
from pydantic import ConfigDict, Field

from hydromt._typing import FS, Geom, NoDataStrategy, SourceMetadata, TimeRange, Zoom
from hydromt.abstract_base import AbstractBaseModel
from hydromt.plugins import PLUGINS

logger: Logger = getLogger(__name__)


class URIResolver(AbstractBaseModel, ABC):
    """URI Resolver responsible for finding the data using the URI in the Data Catalog."""

    model_config = ConfigDict(extra="forbid")
    filesystem: FS = Field(default_factory=LocalFileSystem)
    options: Dict[str, Any] = Field(default_factory=dict)

    @abstractmethod
    def resolve(
        self,
        uri: str,
        *,
        time_range: Optional[TimeRange] = None,
        mask: Optional[Geom] = None,
        variables: Optional[List[str]] = None,
        zoom_level: Optional[Zoom] = None,
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
        zoom_level : Optional[ZoomLevel], optional
            zoom_level of the dataset, by default None
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
    def load_plugins(cls):
        """Load URIResolver plugins."""
        plugins: Dict[str, URIResolver] = PLUGINS.uri_resolver_plugins
        logger.debug(f"loaded {cls.__name__} plugins: {plugins}")
