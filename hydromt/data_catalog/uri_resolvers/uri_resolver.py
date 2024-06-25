"""URI Resolver responsible for finding the data using the URI in the Data Catalog."""

from abc import ABC, abstractmethod
from logging import Logger, getLogger
from typing import Any, ClassVar, Dict, List, Optional

from fsspec import AbstractFileSystem
from pydantic import (
    BaseModel,
    ConfigDict,
    SerializerFunctionWrapHandler,
    model_serializer,
)

from hydromt._typing import Geom, NoDataStrategy, TimeRange, ZoomLevel

logger: Logger = getLogger(__name__)


class URIResolver(BaseModel, ABC):
    """URI Resolver responsible for finding the data using the URI in the Data Catalog."""

    model_config = ConfigDict(extra="forbid")
    name: ClassVar[str]

    @model_serializer(mode="wrap")
    def _serialize(self, nxt: SerializerFunctionWrapHandler) -> Any:
        """Also serialize name."""
        res = nxt(self)
        res["name"] = self.name
        return res

    @abstractmethod
    def resolve(
        self,
        uri: str,
        fs: AbstractFileSystem,
        *,
        time_range: Optional[TimeRange] = None,
        mask: Optional[Geom] = None,
        variables: Optional[List[str]] = None,
        zoom_level: Optional[ZoomLevel] = None,
        handle_nodata: NoDataStrategy = NoDataStrategy.RAISE,
        options: Optional[Dict[str, Any]] = None,
    ) -> List[str]:
        """Resolve a single uri to multiple uris.

        Parameters
        ----------
        uri : str
            Unique Resource Identifier
        fs : AbstractFileSystem
            fsspec filesystem used to resolve wildcards in the uri
        time_range : Optional[TimeRange], optional
            left-inclusive start end time of the data, by default None
        mask : Optional[Geom], optional
            A geometry defining the area of interest, by default None
        zoom_level : Optional[ZoomLevel], optional
            zoom_level of the dataset, by default None
        variables : Optional[List[str]], optional
            Names of variables to return, or all if None, by default None
        handle_nodata : NoDataStrategy, optional
            how to react when no data is found, by default NoDataStrategy.RAISE
        options : Optional[Dict[str, Any]], optional
            extra options for this resolver, by default None

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
