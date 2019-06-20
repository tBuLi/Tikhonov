Definitions
-----------

Matrices
========
Starting of, we have the general Tikhonov function

.. math::
    J = \frac{1}{2} (A x - y)^T W (A x - y) + \frac{1}{2} \vert \Lambda x \vert^2

where :math:`\Lambda` is the regularization matrix, and :math:`W` is the weight
matrix. We want to solve for :math:`x` such that :math:`J` is minimal:

.. math::
    \frac{\partial J}{\partial x^T} = A^T W (A x - y) + (\Lambda^T \Lambda) x = 0

which yields

.. math::
    \Lambda^T \Lambda x &= - A^T W A x + A^T W y \\
    (\Lambda^T \Lambda + A^T W A) x &= A^T W y \\
    x &= (\Lambda^T \Lambda + A^T W A)^{-1} A^T W y

Alternative derivation: defining the residue
:math:`r = Ax - y`, we instead find

.. math::
    \frac{\partial J}{\partial x^T} = A^T W r + (\Lambda^T \Lambda) x = 0

.. math::
    x = - (\Lambda^T \Lambda)^{-1} A^T W r

we can substitute this solution for :math:`x` back into
:math:`r`:

.. math::
    r (I + A (\Lambda^T \Lambda)^{-1} A^T W) = - y

The simplest choice of :math:`\Lambda = \lambda I`, in which case this equation
simplifies to

.. math::
    r (I + \frac{1}{\lambda^2} A A^T W) = - y

We introduce the following notation:

.. math::
    T &= \Lambda^T \Lambda \\
    H_y &= A A^T \\
    H_x &= A^T A
    R_y &= (I + A (\Lambda^T \Lambda)^{-1} A^T W)

This notation is chosen because :math:`H_x` is the Hessian of :math:`x`, and
:math:`H_y = A A^T` can be thought of as the Hessian of :math:`y`. :math:`R_y`
is so named because it is the regularized version of :math:`A`.

Multiple Datasets
=================
In the previous section :math:`y` was assumed to be a vector.
(Technically, a :math:`(N_y, 1)`-matrix.)
However, it is perfectly allowed to regularize multiple data sets at once by
turning it into a :math:`(N_y, N_{sets})`-matrix, where :math:`N_{sets}` is the
number of data sets. The function :math:`J` then becomes

.. math::
    J_k &= \frac{1}{2} (A x_k - y_k)^T W (A x_k - y_k) + \frac{1}{2} \vert \Lambda x_k \vert^2 \\
    J &= \sum_{k=1}^{N_{sets}} J_k

Functionals
===========
Things get truly interesting, and surprisingly simple, when we work with
functionals instead. We start from

.. math::
    J = \frac{1}{2} \sum_{i=1}^{N} ( \int_{-\infty}^{\infty} A_i(t) x(t) dt - y_i)^2 + \frac{1}{2} \int_{-\infty}^{\infty} (\Lambda(t) x(t))^2 dt

where :math:`A_i(t)` is the kernel of integral, for example :math:`e^{- s_i t}`
for a Laplace transform. As always, there is some ambiguity/freedom in the shape
of :math:`\Lambda`. Here it is written as a scalar function, but it could also
be chosen as a constant, or as a function with index :math:`i`.

Repeating the same steps as above, we find that

.. math::
    x(t) = - \frac{1}{\Lambda(t)^2} \sum_{i=1}^{N} A_i(t) r_i

which leads to

.. math::
    r_i &= \int_{-\infty}^{\infty} A_i(t) x(t) dt - y \\
    r_i &= - \sum_{j=1}^{N} r_j \int_{-\infty}^{\infty} \frac{A_j(t) A_i(t)}{\Lambda(t)^2} dt - y
    r_i &= - \sum_{j=1}^{N} r_j M_{ij} - y

where :math:`M_ij = \int_{-\infty}^{\infty} \frac{A_j(t) A_i(t)}{\Lambda(t)^2} dt`.

