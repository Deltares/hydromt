from pathlib import Path
from typing import Any, Dict, Optional, Union

from pydantic import AnyUrl, BaseModel, ConfigDict
from pydantic_core import Url

from hydromt.typing import Bbox, Number, TimeRange


class SourceSpecDict(BaseModel):
    source: str
    provider: Optional[str] = None
    version: Optional[Union[str, int]] = None

    @staticmethod
    def from_dict(input_dict):
        return SourceSpecDict(**input_dict)


class Extent(BaseModel):
    time_range: TimeRange
    bbox: Bbox


class DataCatalogMetaData(BaseModel):
    root: Optional[Path] = None
    version: Optional[Union[str, int]] = None
    name: Optional[str] = None
    model_config = ConfigDict(extra="allow")

    @staticmethod
    def from_dict(input_dict):
        if input_dict is None:
            return DataCatalogMetaData()
        else:
            return DataCatalogMetaData(**input_dict)


class DataCatalogItemMetadata(BaseModel):
    category: Optional[str] = None
    paper_doi: Optional[str] = None
    paper_ref: Optional[str] = None
    source_license: Optional[str] = None
    source_url: Optional[AnyUrl] = None
    source_version: Optional[str] = None
    notes: Optional[str] = None

    @staticmethod
    def from_dict(input_dict):
        if input_dict is None:
            return DataCatalogItemMetadata()
        else:
            item_source_url = input_dict.pop("source_url", None)
            if item_source_url:
                source_url = Url(item_source_url)
            else:
                source_url = None
            return DataCatalogItemMetadata(**input_dict, source_url=item_source_url)


class DataCatalogItem(BaseModel):
    name: str
    data_type: str
    driver: str
    path: Path
    crs: Optional[int] = None
    filesystem: Optional[str] = None
    kwargs: Dict[str, Any] = {}
    storage_options: Dict[str, Any] = {}
    rename: Dict[str, str] = {}
    nodata: Optional[Number] = None
    meta: Optional[DataCatalogItemMetadata] = None
    unit_add: Optional[Dict[str, Number]] = None
    unit_mult: Optional[Dict[str, Number]] = None

    @staticmethod
    def from_dict(input_dict, name=None):
        dict_name = input_dict.pop("name", None)
        if name is None:
            entry_name = dict_name
        else:
            entry_name = name
        item_metadata = DataCatalogItemMetadata.from_dict(input_dict.pop("meta", {}))
        item_kwargs = input_dict.pop("kwargs", {})
        item_storage_options = input_dict.pop("storage_options", {})
        return DataCatalogItem(
            **input_dict,
            name=entry_name,
            kwargs=item_kwargs,
            storage_options=item_storage_options,
            meta=item_metadata,
        )


# class DataCatalogAlias(BaseModel):
#     alias: str

#     @staticmethod
#     def from_dict(input_dict):
#         return DataCatalogAlias(**input_dict)


class DataCatalogValidator(BaseModel):
    meta: Optional[DataCatalogMetaData] = None
    sources: Dict[str, DataCatalogItem] = {}
    # aliases: Dict[str, DataCatalogAlias] = {}
    aliases: Dict[str, str] = {}

    @staticmethod
    def from_dict(input_dict):
        if input_dict is None:
            return DataCatalogValidator()
        else:
            meta = input_dict.pop("meta", None)
            catalog_meta = DataCatalogMetaData.from_dict(meta)
            catalog_entries = {}
            catalog_aliases = {}
            for entry_name, entry_dict in input_dict.items():
                if "alias" in entry_dict.keys():
                    catalog_aliases[entry_name] = entry_dict["alias"]
                else:
                    catalog_entries[entry_name] = DataCatalogItem.from_dict(
                        entry_dict, name=entry_name
                    )

            for src, dst in catalog_aliases.items():
                assert (
                    dst in catalog_entries.keys()
                ), f"{src} references unfound entry {dst}"
            return DataCatalogValidator(
                meta=catalog_meta, sources=catalog_entries, aliases=catalog_aliases
            )
