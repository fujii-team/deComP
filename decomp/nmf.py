import numpy as np
from .utils.cp_compat import get_array_module
from .utils.dtype import float_type
from .utils.data import minibatch_index
from .utils import assertion


_JITTER = 1.0e-10


def solve(y, D, x=None, tol=1.0e-3, minibatch=None, maxiter=1000,
          method='multiplicative', mask=None, random_seed=None):
    """
    Non-negative matrix factrization.

    argmin_{x, D} {|y - xD|^2 - alpha |x|}
    s.t. |D_j|^2 <= 1 and D > 0 and x > 0

    with
    y: [n_samples, n_channels]
    x: [n_samples, n_features]
    D: [n_features, n_channels]

    Parameters
    ----------
    y: array-like.
        Shape: [n_samples, ch]
    D: array-like.
        Initial dictionary, shape [ch, n_component]
    alpha: a positive float
        Regularization parameter
    x0: array-like
        An initial estimate of x

    tol: a float.
        Criterion

    mask: an array-like of Boolean (or integer, float)
        The missing point should be zero. One for otherwise.

    Notes
    -----
    This is essentially implements
    Mensch ARTHURMENSCH, A., Mairal JULIENMAIRAL, J., & Thirion BETRANDTHIRION,
    B. (n.d.).
    Dictionary Learning for Massive Matrix Factorization Gael Varoquaux.
    Retrieved from http://proceedings.mlr.press/v48/mensch16.pdf
    """

    # Check all the class are numpy or cupy
    xp = get_array_module(y, D, x)

    rng = np.random.RandomState(random_seed)
    if x is None:
        x = xp.ones(y.shape[:-1] + (D.shape[0], ), dtype=y.dtype)

    if mask is None:
        mask = xp.ones(y.shape, dtype=float_type(y.dtype))

    assertion.assert_dtypes(y=y, D=D, x=x)
    assertion.assert_dtypes(y=y, D=D, x=x, mask=mask, dtypes='f')
    assertion.assert_shapes('x', x, 'D', D, axes=1)
    assertion.assert_shapes('y', y, 'D', D, axes=[-1])
    assertion.assert_shapes('y', y, 'mask', mask)
    assertion.assert_ndim('y', y, 2)
    assertion.assert_ndim('D', D, 2)
    assertion.assert_ndim('x', x, 2)

    return solve_fastpath(y, D, x, tol, minibatch, maxiter,
                          method, rng, xp, mask)


def solve_fastpath(y, D, x, tol, minibatch, maxiter,
                   method, rng, xp, mask):
    """ Fast path for NMF """
    if method == "multiplicative":
        return solve_multiplicative(y, D, x, tol, minibatch, maxiter,
                                    method, rng, xp, mask)
    else:
        raise NotImplementedError('Method {0:s} is not yet implemented'.format(
                                                                method))


def solve_multiplicative(y, D, x, tol, minibatch, maxiter,
                         method, rng, xp, mask):
    """ NMF with multiplicative update """
    for it in range(1, maxiter):
        indexes = minibatch_index(y.shape, minibatch, rng)
        x_minibatch = x[indexes]
        y_minibatch = y[indexes]
        mask_minibatch = mask[indexes]
        # update x
        ymu = xp.dot(x_minibatch, D) * mask_minibatch
        x_minibatch = x_minibatch * (xp.dot(y_minibatch, D.T) /
                                     xp.maximum(xp.dot(ymu, D.T), _JITTER))
        # update D
        ymu = xp.dot(x_minibatch, D) * mask_minibatch
        U = D * (xp.dot(x_minibatch.T, y_minibatch) /
                 xp.maximum(xp.dot(x_minibatch.T, ymu), _JITTER))
        # normalize
        Unorm = xp.sum(U * U, axis=-1, keepdims=True)
        D_new = U / xp.maximum(Unorm, 1.0)

        x[indexes] = x_minibatch
        if xp.max(xp.abs(D - D_new)) < tol:
            return it, D, x
        D = D_new

    return maxiter, D, x
