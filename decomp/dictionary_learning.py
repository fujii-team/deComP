import numpy as np
from .utils.cp_compat import get_array_module
from . import lasso


_JITTER = 1.0e-10


def minibatch_index(shape, minibatch, rng):
    """ Construct a minibatch index. """
    return tuple([rng.randint(0, s, minibatch) for s in shape[:-1]])


def solve(y, D, alpha, x=None, tol=1.0e-3,
          minibatch=1, maxiter=1000,
          lasso_method='ista', lasso_iter=10,
          mask=None,
          random_seed=None):
    """
    Learn Dictionary with lasso regularization.

    argmin_{x, D} {|y - xD|^2 - alpha |x|}
    s.t. |D_j|^2 <= 1

    with
    y: [..., n_channels]
    x: [..., n_features]
    D: [n_features, n_channels]

    Parameters
    ----------
    y: array-like.
        Shape: [..., ch]
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
    """
    # Check all the class are numpy or cupy
    if x is None:
        xp = get_array_module(y, D)
    else:
        xp = get_array_module(y, D, x)

    rng = np.random.RandomState(random_seed)
    if x is None:
        x = xp.zeros((*y.shape[:-1], D.shape[0]), dtype=y.dtype)
    else:
        x = x

    A = xp.zeros((D.shape[0], D.shape[0]), dtype=y.dtype)
    B = xp.zeros((D.shape[0], D.shape[1]), dtype=y.dtype)

    for it in range(1, maxiter):
        try:
            indexes = minibatch_index(y.shape, minibatch, rng)
            x_minibatch = x[indexes]
            y_minibatch = y[indexes]

            # lasso
            it2, x_minibatch = lasso.solve(y_minibatch, D, alpha,
                                           x0=x_minibatch, tol=tol,
                                           method=lasso_method,
                                           maxiter=lasso_iter)
            x[indexes] = x_minibatch

            # Dictionary update
            if minibatch > 1:
                theta = (it * minibatch if it < minibatch
                         else minibatch**2 + it - minibatch)
                beta = (theta + 1.0 - minibatch) / (theta + 1.0)
            else:
                beta = 1.0

            if y.dtype.kind == 'c':
                A = beta * A + xp.dot(xp.conj(x_minibatch.T), x_minibatch)
                B = beta * B + xp.dot(xp.conj(x_minibatch.T), y_minibatch)
            else:
                A = beta * A + xp.dot(x_minibatch.T, x_minibatch)
                B = beta * B + xp.dot(x_minibatch.T, y_minibatch)

            flag = []
            for k in range(D.shape[0]):
                uk = (B[k] - xp.dot(A[k], D)) / (A[k, k] + _JITTER) + D[k]

                if y.dtype.kind == 'c':
                    Unorm = xp.sum(xp.real(xp.conj(uk) * uk))
                else:
                    Unorm = xp.sum(uk * uk)

                d_new = uk / xp.maximum(Unorm, 1.0)

                flag.append(xp.max(xp.abs(D[k] - d_new)) < tol)
                D[k] = d_new

            if all(flag):
                return it, D, x

        except KeyboardInterrupt:
            return it, D, x
    return maxiter, D, x