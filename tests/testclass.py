# -*- coding: utf-8 -*-
"""Tests for the models.model_api of hydromt."""

import os
from os.path import join, isfile
import xarray as xr
import numpy as np
from affine import Affine
import logging

from pyflwdir import core_d8
import hydromt
from hydromt import raster
from hydromt.models import MODELS
from hydromt.models.model_api import Model

__all__ = ["TestModel"]

logger = logging.getLogger(__name__)


class TestModel(Model):
    _NAME = "testname"
    _CONF = "test.ini"
    _GEOMS = {}
    _MAPS = {"elevtn": "dem", "flwdir": "ldd", "basins": "basins", "mask": "mask"}
    _FOLDERS = ["data"]

    def __init__(
        self, root=None, mode="w", config_fn=None, data_libs=None, logger=logger
    ):
        super().__init__(
            root=root,
            mode=mode,
            config_fn=config_fn,
            data_libs=data_libs,
            logger=logger,
        )

    def setup_basemaps(self, region, res=0.5, crs=4326, add_geom=False):
        _maps = {
            "elevtn": {"func": _rand_float, "nodata": -9999.0},
            "flwdir": {"func": _rand_d8, "nodata": core_d8._mv},
            "mask": {"func": _rand_msk, "nodata": -1},
            "basins": {"func": _rand_msk, "nodata": -1},
        }
        ds_base = _create_staticmaps(_maps, region["bbox"], res)
        self.set_crs(crs)
        rmdict = {k: v for k, v in self._MAPS.items() if k in ds_base.data_vars}
        self.set_staticmaps(ds_base.rename(rmdict))
        if add_geom:
            self.set_staticgeoms(self.region, "region")

    def setup_param(self, name, value):
        nodatafloat = -999
        da_param = xr.where(self.staticmaps[self._MAPS["basins"]], value, nodatafloat)
        da_param.raster.set_nodata(nodatafloat)

        da_param = da_param.rename(name)
        self.set_staticmaps(da_param)

    def read_staticmaps(self):
        fn = join(self.root, "staticmaps.nc")
        if not self._write:
            self._staticmaps = xr.Dataset()
        if fn is not None and isfile(fn):
            self.logger.info(f"Read staticmaps from {fn}")
            ds = xr.open_dataset(fn, mask_and_scale=False).load()
            ds.close()
            self.set_staticmaps(ds)

    def write_staticmaps(self):
        if not self._write:
            raise IOError("Model opened in read-only mode")
        ds_out = self.staticmaps
        fn = join(self.root, "staticmaps.nc")
        self.logger.info(f"Write staticmaps to {fn}")
        ds_out.to_netcdf(fn)


def _rand_float(shape, dtype=np.float32):
    return np.random.rand(*shape).astype(dtype)


def _rand_d8(shape):
    d8 = core_d8._ds.ravel()
    return d8.flat[np.random.randint(d8.size, size=shape)]


def _rand_msk(shape):
    mks = np.array([0, 1, 2], dtype=np.int8)
    return mks[np.random.randint(mks.size, size=shape)]


def _create_staticmaps(_maps, bbox, res):
    w, s, e, n = bbox
    shape = (6, 10)
    transform = Affine(res, w, e, s, res, n)
    data_vars = {n: (_maps[n]["func"](shape), _maps[n]["nodata"]) for n in _maps}
    ds = raster.RasterDataset.from_numpy(data_vars=data_vars, transform=transform)
    ds.raster.set_crs(4326)
    return ds
