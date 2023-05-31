# -*- coding: utf-8 -*-
"""HydroMT statistics."""

from .skills import (
    bias,
    kge,
    kge_2012,
    kge_non_parametric,
    kge_non_parametric_flood,
    pearson_correlation,
    percentual_bias,
    rmse,
    rsquared,
    spearman_rank_correlation,
)

__all__ = [
    "bias",
    "kge",
    "kge_2012",
    "kge_non_parametric",
    "kge_non_parametric_flood",
    "pearson_correlation",
    "percentual_bias",
    "rmse",
    "rsquared",
    "spearman_rank_correlation",
]
