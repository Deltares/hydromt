# -*- coding: utf-8 -*-
"""Tests for the vector submodule."""

import pytest
import numpy as np
import xarray as xr

from hydromt import stats


def test_geo(obsda, simda):
    assert stats.bias(simda, obsda).values.round(2) == 5.0
    assert stats.percentual_bias(simda, obsda).values.round(2) == 0.1
    assert stats.nashsutcliffe(simda, obsda).values.round(2) == 0.97
    assert stats.lognashsutcliffe(simda, obsda).values.round(2) == 0.85
    assert stats.pearson_correlation(simda, obsda).values.round(2) == 1.0
    assert stats.spearman_rank_correlation(simda, obsda).values.round(2) == 1.0
    assert stats.kge(simda, obsda)["kge"].values.round(2) == 0.9
    assert stats.kge_2012(simda, obsda)["kge_2012"].values.round(2) == 0.86
    assert stats.kge_non_parametric(simda, obsda)["kge_np"].values.round(2) == 0.9
    assert (
        stats.kge_non_parametric_flood(simda, obsda)["kge_np_flood"].values.round(2)
        == 0.9
    )
    assert stats.rsquared(simda, obsda).values.round(2) == 1.0
    assert stats.mse(simda, obsda).values.round(2) == 9125.0
    assert stats.rmse(simda, obsda).values.round(2) == 95.52
