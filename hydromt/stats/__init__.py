# -*- coding: utf-8 -*-
"""HydroMT statistics."""

from hydromt.stats.design_events import get_peak_hydrographs
from hydromt.stats.extremes import (
    eva,
    fit_extremes,
    get_peaks,
    get_return_value,
)
from hydromt.stats.skills import (
    bias,
    kge,
    kge_2012,
    kge_non_parametric,
    kge_non_parametric_flood,
    lognashsutcliffe,
    mse,
    nashsutcliffe,
    pearson_correlation,
    percentual_bias,
    rmse,
    rsquared,
    rsr,
    spearman_rank_correlation,
    volumetric_error,
)

__all__ = [
    "bias",
    "kge",
    "kge_2012",
    "kge_non_parametric",
    "kge_non_parametric_flood",
    "lognashsutcliffe",
    "mse",
    "nashsutcliffe",
    "pearson_correlation",
    "percentual_bias",
    "rmse",
    "rsquared",
    "rsr",
    "spearman_rank_correlation",
    "volumetric_error",
    "eva",
    "fit_extremes",
    "get_peaks",
    "get_return_value",
    "get_peak_hydrographs",
]
