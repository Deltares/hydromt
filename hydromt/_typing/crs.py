from typing import Any, Union

from pydantic import PlainSerializer, PlainValidator
from pyproj import CRS as ProjCRS
from pyproj.exceptions import CRSError
from typing_extensions import Annotated


def _serialize_crs(crs: ProjCRS) -> Union[str, None]:
    return crs.to_wkt()


def _validate_crs(crs: Any) -> ProjCRS:
    try:
        return ProjCRS(crs)
    except CRSError as e:
        if "Invalid projection" in str(e):
            try:
                return ProjCRS.from_authority(auth_name="ESRI", code=crs)
            except CRSError:
                pass  # handle previous exception
        raise e


CRS = Annotated[ProjCRS, PlainValidator(_validate_crs), PlainSerializer(_serialize_crs)]
