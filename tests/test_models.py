#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Tests for `tikhonov.models` package."""

import pytest

from tikhonov.models import FunctionalMorozovModel

@pytest.fixture
def response():
    """Sample pytest fixture.

    See more at: http://doc.pytest.org/en/latest/fixture.html
    """
    # import requests
    # return requests.get('https://github.com/audreyr/cookiecutter-pypackage')


def test_content(response):
    """Sample pytest test function with the pytest fixture as an argument."""
    # from bs4 import BeautifulSoup
    # assert 'GitHub' in BeautifulSoup(response.content).title.string

def test_FunctionalMorozovModel():
    model = FunctionalMorozovModel
    a, = model.params
    M_y, T_y, W_y, y = model.independent_vars
    d, = model.dependent_vars
    R_y, r = model.interdependent_vars
    assert set(model.optional_symbols.keys()) == {T_y, W_y}
    assert model.connectivity_mapping == {R_y: {M_y, W_y, a, T_y}, d: {r}, r: {R_y, y}}
