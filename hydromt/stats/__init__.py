# -*- coding: utf-8 -*-
"""HydroMT statistics."""

from .design_events import get_peak_hydrographs, hydrograph_1d
from .extremes import (
    eva,
    eva_block_maxima,
    eva_peaks_over_threshold,
    fit_extremes,
    get_peaks,
    get_return_value,
)
from .skills import (
    bias,
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
    "eva_block_maxima",
    "eva_peaks_over_threshold",
    "fit_extremes",
    "get_peaks",
    "get_return_value",
    "get_peak_hydrographs",
    "hydrograph_1d",
]
