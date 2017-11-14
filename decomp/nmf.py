import numpy as np
from .utils.cp_compat import get_array_module
from .utils.dtype import float_type
from .utils.data import minibatch_index
from .utils import assertion, normalize


_JITTER = 1.0e-10


def solve(y, D, x=None, tol=1.0e-3, minibatch=None, maxiter=1000,
          likelihood='square', mask=None, random_seed=None):
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
        x = xp.ones((y.shape[0], D.shape[0]), dtype=y.dtype)

    assertion.assert_dtypes(y=y, D=D, x=x)
    assertion.assert_dtypes(y=y, D=D, x=x, mask=mask, dtypes='f')
    assertion.assert_shapes('x', x, 'D', D, axes=1)
    assertion.assert_shapes('y', y, 'D', D, axes=[-1])
    assertion.assert_shapes('y', y, 'mask', mask)
    assertion.assert_ndim('y', y, 2)
    assertion.assert_ndim('D', D, 2)
    assertion.assert_ndim('x', x, 2)

    return solve_fastpath(y, D, x, tol, minibatch, maxiter, likelihood,
                          rng, xp, mask)


def solve_fastpath(y, D, x, tol, minibatch, maxiter,
                   likelihood, rng, xp, mask):
    """ Fast path for NMF """
    D = normalize.l2_strict(D, axis=-1, xp=xp)
    if mask is None:
        if likelihood == 'square':
            return solve_square(y, D, x, tol, minibatch, maxiter, rng, xp)
        else:
            raise NotImplementedError('NMF with {0:s} likelihood is not yet '
                                      'implemented'.format(likelihood))
    else:
        raise NotImplementedError('NMF with mask is not yet implemented.')


def solve_square(y, D, x, tol, minibatch, maxiter, rng, xp):
    """ NMF with square sum likelihood """
    xx_sum = xp.zeros((D.shape[0], D.shape[0]), dtype=y.dtype)
    xy_sum = xp.zeros((D.shape[0], D.shape[1]), dtype=y.dtype)

    for it in range(1, maxiter):
        indexes = minibatch_index(y.shape[:1], minibatch, rng)
        x_minibatch = x[indexes]
        y_minibatch = y[indexes]
        # update x
        ymu = xp.dot(x_minibatch, D)
        x_minibatch = x_minibatch * (xp.dot(y_minibatch, D.T) /
                                     xp.dot(ymu, D.T))
        x[indexes] = x_minibatch

        xx_sum += xp.dot(x_minibatch.T, x_minibatch) / it
        xy_sum += xp.dot(x_minibatch.T, y_minibatch) / it

        # update D
        U = D * xy_sum / xp.dot(xx_sum, D)
        D_new = normalize.l2(U, axis=-1, xp=xp)

        if xp.max(xp.abs(D - D_new)) < tol:
            return it, D, x
        D = D_new

    return maxiter, D, x
