"""Abstract DataSource class."""

from abc import ABC
from copy import deepcopy
from logging import Logger, getLogger
from os.path import abspath, join
from pathlib import Path
from typing import Any, ClassVar, Dict, List, Optional, TypeVar, Union

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    PrivateAttr,
    SerializerFunctionWrapHandler,
    model_serializer,
    model_validator,
)

from hydromt._typing import DataType, SourceMetadata
from hydromt._utils.uris import is_valid_url
from hydromt.data_adapter.data_adapter_base import DataAdapterBase
from hydromt.drivers import BaseDriver

logger: Logger = getLogger(__name__)

T = TypeVar("T")


class DataSource(BaseModel, ABC):
    """
    A DataSource is a parsed section of a DataCatalog.

    The DataSource, specific for a data type within HydroMT, is responsible for
    validating the input from the DataCatalog, to
    ensure the workflow fails as early as possible. A DataSource has information on
    the driver that the data should be read with, and is responsible for initializing
    this driver.
    """

    model_config = ConfigDict(extra="forbid")

    _used: bool = PrivateAttr(default=False)

    name: str
    uri: str
    data_adapter: DataAdapterBase
    driver: BaseDriver
    data_type: ClassVar[DataType]
    root: Optional[str] = Field(
        default=None, exclude=True
    )  # root is already in the catalog.
    version: Optional[str] = Field(default=None)
    provider: Optional[str] = Field(default=None)
    metadata: SourceMetadata = Field(default_factory=SourceMetadata)

    def summary(self) -> Dict[str, Any]:
        """Return a summary of the DataSource."""
        summ: Dict[str, Any] = self.model_dump(include={"uri"})
        summ.update(
            {
                "data_type": self.__class__.data_type,
                "driver": self.driver.__repr_name__(),
                **self.metadata.model_dump(exclude_unset=True),
            }
        )
        return summ

    @model_validator(mode="before")
    @classmethod
    def _validate_data_type(cls, data: Any) -> Any:
        """Pydantic does not check class variables, so it is checked here."""
        if isinstance(data, dict):
            copy_data: dict = deepcopy(data)
            if data_type := copy_data.pop("data_type", None):
                if data_type != cls.data_type:
                    raise ValueError(f"'data_type' must be '{cls.data_type}'.")
        return copy_data

    @property
    def full_uri(self) -> str:
        """Join root with uri."""
        uri_is_url: bool = is_valid_url(self.uri)
        if uri_is_url:
            # uri is fully self-describing
            return self.uri
        if not uri_is_url and self.root:
            if is_valid_url(self.root):
                # use '/' to connect url parts
                return f"{self.root.rstrip('/')}/{self.uri}"
        # Local file, make absolute
        return _abs_path(self.root, self.uri)

    @model_serializer(mode="wrap")
    def _serialize(self, nxt: SerializerFunctionWrapHandler) -> Dict[str, Any]:
        """Serialize data_type."""
        res: Dict[str, Any] = nxt(self)
        res["data_type"] = self.data_type
        return res


def _abs_path(root: Union[Path, str], rel_path: Union[Path, str]) -> str:
    path = Path(str(rel_path))
    if not path.is_absolute():
        if root is not None:
            rel_path = join(root, rel_path)
        path = Path(abspath(rel_path))
    return str(path)


def get_nested_var(
    nested_var: List[str], nested_object: Any, default: Any = None
) -> Any:
    """Get nested variable during pydantic "before" validation."""
    if lenn := len(nested_var) > 0:
        var: str = nested_var.pop(0)
        prop: Union[BaseModel, Dict, None] = _get_property(nested_object, var, default)

        return get_nested_var(nested_var, prop, default)
    elif lenn == 0:  # Got the value we were looking for
        return nested_object


def set_nested_var(
    nested_var_keys: List[str],
    nested_object: Any,
    value: Any,
):
    """Set nested variable during pydantic "before" validation."""
    key: str = nested_var_keys.pop(0)
    if len(nested_var_keys) > 0:
        prop: Union[BaseModel, Dict, None] = _get_property(
            nested_object, key, default={}
        )
        # Then set result of get on larger object
        newprop = set_nested_var(nested_var_keys, prop, value)
        return _set_pydantic_or_dict_property(nested_object, key, newprop)
    else:
        return _set_pydantic_or_dict_property(nested_object, key, value)


def _get_property(obj: Any, key: str, default: T) -> Union[Dict, BaseModel, T]:
    if isinstance(obj, Dict):
        return obj.get(key, default)
    elif isinstance(obj, BaseModel):
        return getattr(obj, key)
    elif isinstance(obj, str):
        # shortcut for {"name": str}
        return {"name": obj}
    else:
        return default


def _set_pydantic_or_dict_property(
    obj: Any, key: str, value: Any
) -> Union[BaseModel, Dict]:
    if isinstance(obj, Dict):
        obj[key] = value
    elif isinstance(obj, BaseModel):
        setattr(obj, key, value)  # forces validation
    else:
        raise ValueError(
            f"Cannot set value '{value}' on object '{obj}' with key '{key}'."
        )
    return obj
