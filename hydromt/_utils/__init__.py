"""Utility functions for hydromt that have no other home."""

from hydromt._utils.caching import cache_vrt_tiles, copy_to_local
from hydromt._utils.dataset import (
    _rename_vars,
    _set_metadata,
    _shift_dataset_time,
    _single_var_as_array,
)
from hydromt._utils.deep_merge import _deep_merge
from hydromt._utils.dictionaries import _partition_dictionaries
from hydromt._utils.elevation import _elevation2rgba, _rgba2elevation
from hydromt._utils.log import initialize_logging, to_file
from hydromt._utils.nodata import _has_no_data, _set_raster_nodata, _set_vector_nodata
from hydromt._utils.path import _make_config_paths_absolute, _make_config_paths_relative
from hydromt._utils.rgetattr import _rgetattr
from hydromt._utils.steps_validator import _validate_steps
from hydromt._utils.temp_env import temp_env
from hydromt._utils.unused_kwargs import _warn_on_unused_kwargs
from hydromt._utils.uris import _is_valid_url, _strip_scheme

__all__ = [
    "cache_vrt_tiles",
    "copy_to_local",
    "_deep_merge",
    "_rename_vars",
    "_set_metadata",
    "initialize_logging",
    "to_file",
    "_shift_dataset_time",
    "_single_var_as_array",
    "_partition_dictionaries",
    "_elevation2rgba",
    "_rgba2elevation",
    "_has_no_data",
    "_set_raster_nodata",
    "_set_vector_nodata",
    "_rgetattr",
    "_validate_steps",
    "_warn_on_unused_kwargs",
    "_is_valid_url",
    "_strip_scheme",
    "_make_config_paths_absolute",
    "_make_config_paths_relative",
    "temp_env",
]


class _classproperty(property):
    def __get__(self, owner_self, owner_cls):
        return self.fget(owner_cls)
