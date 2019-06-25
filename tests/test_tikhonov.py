#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Tests for `tikhonov` package."""

import pytest
from collections import OrderedDict

import numpy as np
import symfit as sf
from symfit.core.fit_results import FitResults

from tikhonov import Regularizer
from tikhonov.models import FunctionalMorozovModel
from tikhonov.definitions import M_y, y, y_stdev, d, W_y, T_y

@pytest.fixture
def laplace_dataset():
    """Sample pytest fixture.

    See more at: http://doc.pytest.org/en/latest/fixture.html
    """
    t, f, s, F = sf.variables('t, f, s, F')
    model = sf.Model({f: t * sf.exp(- t)})
    laplace_model = sf.Model(
        {F: sf.laplace_transform(model[f], t, s, noconds=True)}
    )

    epsilon = 0.01  # 1 percent noise
    s_data = np.linspace(0, 10, 101)[1:]
    F_data = laplace_model(s=s_data).F
    F_sigma = epsilon * F_data
    np.random.seed(42)
    F_data = np.random.normal(F_data, F_sigma)
    # Reshape to matrices
    F_data = F_data[:, None]
    F_sigma = F_sigma[:, None]
    M_mat = 1 / (s_data[None, :] + s_data[:, None])
    delta = np.atleast_2d(np.linalg.norm(F_sigma)**2)
    return {M_y: M_mat, y: F_data, y_stdev: F_sigma, d: delta}


def test_model_selection_functional(laplace_dataset):
    """
    Test if the correct model is selected.
    """
    M_mat = laplace_dataset[M_y]
    y_mat = laplace_dataset[y]
    delta = laplace_dataset[d]

    with pytest.raises(TypeError):
        reg = Regularizer(data={M_y: M_mat, y: y_mat})
    reg = Regularizer(data={M_y: M_mat, y: y_mat, d: delta})
    assert reg.model is FunctionalMorozovModel

def test_regularizer_default_data(laplace_dataset):
    """
    Test if the data is set correctly
    """
    M_mat = laplace_dataset[M_y]
    y_mat = laplace_dataset[y]
    delta = laplace_dataset[d]

    data = {M_y: M_mat, y: y_mat, d: delta}
    reg = Regularizer(data=data)
    assert isinstance(reg.data, OrderedDict)
    assert y_stdev in reg.data
    assert reg.data[y_stdev].shape == (100, 1)
    assert T_y in reg.data
    assert reg.data[T_y].shape == M_mat.shape

def test_regularizer(laplace_dataset):
    """
    Test if we can perform a laplace transform!
    """
    M_mat = laplace_dataset[M_y]
    y_mat = laplace_dataset[y]
    delta = laplace_dataset[d]

    reg = Regularizer(data={M_y: M_mat, y: y_mat, d: delta})
    result = reg.execute()
    assert isinstance(result, FitResults)
    assert 'r' in result.model_ans._fields
    assert 'R_y' in result.model_ans._fields
    assert 'morozov' in result.model_ans._fields
    assert 'W_y' in result.model_ans._fields
    # Test if the default argument came through
    assert result.model_ans.W_y.shape == (100, 100)
    assert np.allclose(result.model_ans.W_y, np.eye(100))

def test_regularizer_multiple_datasets(laplace_dataset):
    """
    Test if we can perform a laplace transform!
    """
    N_sets = 3
    M_mat = laplace_dataset[M_y]
    # Multiply by the index to make each fit unique
    y_mat = np.hstack([i * laplace_dataset[y] for i in range(1, N_sets + 1)])
    y_stdev_mat = np.hstack([i * laplace_dataset[y_stdev] for i in range(1, N_sets + 1)])
    delta = np.hstack([i * laplace_dataset[d] for i in range(1, N_sets + 1)])
    assert M_mat.shape == (100, 100)
    assert y_mat.shape == (100, 3)
    assert y_stdev_mat.shape == (100, 3)
    assert delta.shape == (1, 3)

    reg = Regularizer(data={M_y: M_mat, y: y_mat, d: delta, y_stdev: y_stdev_mat})
    results = reg.execute()
    print(results[0])
    assert len(results) == N_sets
    for i in range(N_sets):
        assert isinstance(results[i], FitResults)

def test_generator(laplace_dataset):
    """
    Test if we can call this function as a generator instead.
    """
    M_mat = laplace_dataset[M_y]
    y_mat = laplace_dataset[y]
    delta = laplace_dataset[d]

    reg = Regularizer(data={M_y: M_mat, y: y_mat, d: delta})
    for fit_result in reg.generator_execute():
        assert isinstance(fit_result, FitResults)
