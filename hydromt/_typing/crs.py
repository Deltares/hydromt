from typing import Any, List, Optional, Union

from pydantic import PlainSerializer, PlainValidator
from pyproj import CRS as ProjCRS
from pyproj.exceptions import ProjError
from typing_extensions import Annotated


def _serialize_crs(crs: ProjCRS) -> Union[int, List[str], str]:
    # Try epsg authority first
    epsg: Optional[int] = crs.to_epsg()
    if not epsg:
        # Then try any authority
        auth: Optional[List[str, str]] = list(
            crs.to_authority()
        )  # cast to list to be serializable to yaml
        if not auth:
            # reserve wkt for last
            return crs.to_wkt()
        return auth
    return epsg


def _validate_crs(crs: Any) -> ProjCRS:
    try:
        return ProjCRS.from_user_input(crs)
    except ProjError:
        return ProjCRS.from_authority("ESRI", crs)  # fallback on ESRI


CRS = Annotated[ProjCRS, PlainValidator(_validate_crs), PlainSerializer(_serialize_crs)]
