#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Tests for `tikhonov.models` package."""

import pytest

from tikhonov.models import FunctionalMorozovModel
from tikhonov.definitions import *

def test_FunctionalMorozovModel():
    model = FunctionalMorozovModel
    a, = model.params
    M_y, T_y, d, y, y_stdev = model.independent_vars
    morozov, = model.dependent_vars
    R_y, W_y, r = model.interdependent_vars
    assert set(model.optional_symbols.keys()) == {T_y, morozov, y_stdev}
    assert model.connectivity_mapping == {R_y: {M_y, W_y, a, T_y},
                                          morozov: {d, r},
                                          r: {R_y, y},
                                          W_y: {y_stdev}}
