"""Utility functions for hydromt that have no other home."""

from .caching import _cache_vrt_tiles, _copyfile
from .dataset import (
    _rename_vars,
    _set_metadata,
    _shift_dataset_time,
    _single_var_as_array,
    _slice_temporal_dimension,
)
from .dictionaries import _partition_dictionaries
from .elevation import _elevation2rgba, _rgba2elevation
from .nodata import _has_no_data, _set_raster_nodata, _set_vector_nodata
from .rgetattr import _rgetattr
from .steps_validator import _validate_steps
from .unused_kwargs import _warn_on_unused_kwargs
from .uris import _is_valid_url, _strip_scheme

__all__ = [
    "_cache_vrt_tiles",
    "_copyfile",
    "_rename_vars",
    "_set_metadata",
    "_shift_dataset_time",
    "_single_var_as_array",
    "_slice_temporal_dimension",
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
]


class _classproperty(property):
    def __get__(self, owner_self, owner_cls):
        return self.fget(owner_cls)
