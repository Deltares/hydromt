from os import PathLike

import geopandas as gpd
from hydromt.models._v1.model_region import ModelRegion

from hydromt.models._v1.model_root import ModelRoot
from ...data_catalog import DataCatalog
from typing import Literal, Optional
from ...typing import ModelModes


class Model:
    def __init__(
        self,
        root: PathLike,
        mode: ModelModes,
        data_catalog: Optional[DataCatalog],
        region: Optional[gpd.GeoDataFrame],
    ):
        self.root = ModelRoot(root, mode)
        self.setup_region(region)
        self.data_catalog = data_catalog
        self.set_mode(mode)

    def set_mode(self, mode: ModelModes) -> None:
        self.mode = mode

    def setup_region(self, region: Optional[gpd.GeoDataFrame]) -> None:
        if region is not None:
            self.region = ModelRegion(region)
        else:
            self.region = None
