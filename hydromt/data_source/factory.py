"""Factory function for DataSource."""
from inspect import isabstract
from typing import Any, Dict, Union

from hydromt._typing.type_def import DataType
from hydromt.data_source import DataSource, GeoDataFrameSource, RasterDatasetSource

# Map DataType to DataSource, need to add here when implementing a new Type
available_sources: Dict[DataType, DataSource] = {
    "RasterDataset": RasterDatasetSource,
    "GeoDataFrame": GeoDataFrameSource,
}


def create_source(data: Union[Dict[str, Any], DataSource]) -> DataSource:
    """Create a DataSource.

    Create a datasource from a dictionary, or another DataSource.
    """
    if isinstance(data, DataSource):
        if isabstract(DataSource):
            raise ValueError("DataSource is an Abstract Class")
        else:
            # Already is a subclass of DataSource
            return data

    elif isinstance(data, dict):
        if data_type := data.pop("data_type", None):
            if target_source := available_sources.get(data_type):
                return target_source.model_validate(data)

            raise ValueError(f"Unknown 'data_type': '{data_type}'")
        else:
            raise ValueError("DataSource needs 'data_type'.")
    else:
        raise ValueError(f"Invalid argument for creating DataSource: {data}")
