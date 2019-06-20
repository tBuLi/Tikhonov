"""
MatrixMorozovModel
==================

"""

import symfit as sf

N_x, N_y = sf.symbols('N_x, N_y')
T_x = sf.MatrixSymbol('T_x', N_x, N_x)
T_y = sf.MatrixSymbol('T_y', N_y, N_y)
M_x = sf.MatrixSymbol('M_x', N_x, N_x)
M_y = sf.MatrixSymbol('M_y', N_y, N_y)
H_x = sf.MatrixSymbol('H_x', N_x, N_x)
H_y = sf.MatrixSymbol('H_y', N_y, N_y)
W_x = sf.MatrixSymbol('W_x', N_x, N_x)  # Weight matrix, i.e. inverse covariance
W_y = sf.MatrixSymbol('W_y', N_y, N_y)
R_x = sf.MatrixSymbol('R_x', N_x, N_x)  # Regularizer matrix
R_y = sf.MatrixSymbol('R_y', N_y, N_y)
I_x = sf.Identity(N_x)
I_y = sf.Identity(N_y)
A = sf.MatrixSymbol('A', N_y, N_x)
y = sf.MatrixSymbol('y', N_y, 1)
x = sf.MatrixSymbol('x', N_x, 1)
r = sf.MatrixSymbol('r', N_y, 1)
d = sf.MatrixSymbol('d', 1, 1)
a = sf.Parameter('alpha')

model_dict = {
    R_y: (T_y + M_y * W_y / a**2),
    r: - sf.Inverse(R_y) * y,
    d: r.T * r,
}
FunctionalMorozovModel = sf.CallableModel(model_dict)
FunctionalMorozovModel.optional_symbols = {T_y: I_y, W_y: I_y}

all_models = [FunctionalMorozovModel]
