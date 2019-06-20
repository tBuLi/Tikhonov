API Outline
-----------
At the heart of `Tikhonov` is the `Regularize` object. It can be initiated with
any of the object named in ``Definitions``. Some examples include::

    reg = Regularize(data={A: A_mat, y: y_vec})
    reg_result = reg.execute()

    reg = Regularize(data={H_r: H_r_mat, y: y_vec})
    reg_result = reg.execute()

    I = Identity(10)
    T = Parameter('lambda', value=0.1) * I
    reg = Regularize(data={H_x: H_x_mat, T: T, y: y_vec})
    reg_result = reg.execute()

`Tikhonov` will then select the best model to use in each case.

You can also provide your own model, which has to be a type of
:class:`~symfit.core.fit.BaseCallableModel`::

    N_x, N_y = symbols('N_x, N_y')
    T = MatrixSymbol('T', N_x, N_x)
    H_x = MatrixSymbol('H_x', N_x, N_x)
    W = MatrixSymbol('W', N_x, N_x)
    A = MatrixSymbol('A', N_y, N_x)
    y = MatrixSymbol('y', N_y, 1)
    x = MatrixSymbol('x', N_x, 1)
    r = MatrixSymbol('r', N_y, 1)
    d = MatrixSymbol('d', 1, 1)

    model_dict = {
        H_x: A.T * A,
        W: (T + H_x),
        x: Inverse(W) * A.T * y,
        r: A * x - y,
        d: r.T * r,
    }
    model = CallableModel(model_dict)

    T_mat = Parameter('lambda', value=0.1) * Identity(N_x)
    reg = Regularize(model=model,
                     data={d: np.linalg.norm(y_stdev)**2, T: T_mat,
                           A: A_mat, y: y_mat}
    )
    reg_result = reg.execute()

It is important when doing this, to stick to the names defined in `Definitions`.

Although a lot of work to write down carefully, `symfit` makes this relatively
easy. And fortunatelly, many such models are already present in `Tikhonov`.

Morozov Model
=============
