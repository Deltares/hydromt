# -*- coding: utf-8 -*-
"""Tests for the vector submodule."""

import numpy as np

from hydromt.stats import skills


def test_skills(obsda):
    simda = obsda + 5.0
    assert np.isclose(skills.bias(simda, obsda).values, 5.0)
    assert np.isclose(skills.percentual_bias(simda, obsda).values, 10.3299)
    assert np.isclose(skills.volumetric_error(simda, obsda).values, 0.8967)
    assert np.isclose(skills.nashsutcliffe(simda, obsda).values, 0.97015)
    assert np.isclose(skills.lognashsutcliffe(simda, obsda).values, 0.8517)
    assert np.isclose(skills.pearson_correlation(simda, obsda).values, 1.0)
    assert np.isclose(skills.spearman_rank_correlation(simda, obsda).values, 1.0)
    assert np.isclose(skills.kge(simda, obsda)["kge"].values, 0.8967)
    assert np.isclose(skills.kge_2012(simda, obsda)["kge_2012"].values, 0.86058)
    assert np.isclose(skills.kge_non_parametric(simda, obsda)["kge_np"].values, 0.89390)
    assert np.isclose(
        skills.kge_non_parametric_flood(simda, obsda)["kge_np_flood"].values, 0.8939
    )
    assert np.isclose(skills.rsquared(simda, obsda).values, 1.0)
    assert np.isclose(skills.mse(simda, obsda).values, 25.0)
    assert np.isclose(skills.rmse(simda, obsda).values, 5.0)
    assert np.isclose(skills.rsr(simda, obsda).values, 0.1727659)
