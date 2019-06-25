"""
MatrixMorozovModel
==================

"""
import symfit as sf
import numpy as np
from sympy.core.numbers import Zero, One

from tikhonov.definitions import *

model_dict = {
    R_y: (T_y + M_y * W_y / a**2),
    r: - sf.Inverse(R_y) * y,
    morozov: - d.T * d + r.T * r,
    W_y: lambda y_stdev: np.diag(1 / np.atleast_1d(np.squeeze(y_stdev))**2)
}

FunctionalMorozovModel = sf.CallableNumericalModel(
    model_dict, connectivity_mapping={W_y: {y_stdev}}
)
FunctionalMorozovModel.optional_symbols = {T_y: I_y,
                                           y_stdev: One, morozov: Zero}

all_models = [FunctionalMorozovModel]
