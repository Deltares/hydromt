"""Abstract DataSource class."""

from abc import ABC
from logging import Logger, getLogger
from os.path import abspath, join
from pathlib import Path
from typing import Any, ClassVar, Dict, List, Optional, TypeVar, Union

from pydantic import (
    BaseModel,
    Field,
    PrivateAttr,
    model_validator,
)

from hydromt._typing import DataType
from hydromt.data_adapter.caching import _uri_validator
from hydromt.data_adapter.data_adapter_base import DataAdapterBase
from hydromt.driver import BaseDriver

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

    _used: bool = PrivateAttr(default=False)

    name: str
    uri: str
    data_adapter: DataAdapterBase
    driver: BaseDriver
    data_type: ClassVar[DataType]
    root: Optional[str] = Field(default=None)
    version: Optional[str] = Field(default=None)
    provider: Optional[str] = Field(default=None)
    crs: Optional[int] = Field(default=None)
    extent: Dict[str, Any] = Field(default_factory=dict)

    def summary(self) -> Dict[str, Any]:
        """Return a summary of the DataSource."""
        summ: Dict[str, Any] = self.model_dump(include={"uri"})
        summ.update(
            {
                "data_type": self.__class__.data_type,
                "driver": self.driver.__repr_name__(),
                **self.data_adapter.meta,
            }
        )
        return summ

    # TODO: def to_file(self, **query_params) https://github.com/Deltares/hydromt/issues/840

    @model_validator(mode="before")
    @classmethod
    def _push_down_data_adapter_args(cls, data: Any):
        """Copy unit_add, unit_mult and rename to metadata_adapter.

        The metadata resolver should query the data with the right variables and
        the correct time variables, and this needs these arguments when used in
        conjunction with the DataAdapter. The DataSource is responsable to keep these
        up-to-date.
        """
        if isinstance(data, dict) or isinstance(data, BaseModel):
            try:
                for da_arg in ["unit_add", "unit_mult", "rename"]:
                    value: Any = get_nested_var(["data_adapter", da_arg], data, {})
                    set_nested_var(["driver", "metadata_resolver", da_arg], data, value)
            except ValueError:
                pass  # let pydantic handle any errors
        return data

    @model_validator(mode="before")
    @classmethod
    def _validate_data_type(cls, data: Any) -> Any:
        """Pydantic does not check class variables, so it is checked here."""
        if (
            isinstance(data, dict)
            and data.get("data_type")
            and data.get("data_type") != cls.data_type
        ):
            raise ValueError(f"'data_type' must be '{cls.data_type}'.")
        return data

    @model_validator(mode="after")
    def _validate_uri(self) -> str:
        """In case of a local path, add the root before it."""
        if not _uri_validator(self.uri):
            self.uri = _abs_path(self.root, self.uri)
        return self


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
