import numpy as np
from .utils.cp_compat import get_array_module
from .utils.dtype import float_type
from .utils import assertion
from . import lasso


_JITTER = 1.0e-10


def minibatch_index(shape, minibatch, rng):
    """ Construct a minibatch index. """
    return tuple([rng.randint(0, s, minibatch) for s in shape[:-1]])


def solve(y, D, alpha, x=None, tol=1.0e-3,
          minibatch=1, maxiter=1000, method='parallel_cd',
          lasso_method='ista', lasso_iter=10, lasso_tol=1.0e-5,
          mask=None, random_seed=None):
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
    x: array-like
        An initial estimate of x

    tol: a float.
        Criterion

    method: string
        One of ['parallel_cd', 'block_cd']

    mask: an array-like of Boolean (or integer, float)
        The missing point should be zero. One for otherwise.

    Notes
    -----
    'block_cd':
        Essentially equivalent to
        Mensch ARTHURMENSCH, A., Mairal JULIENMAIRAL, J., & Thirion BETRANDTHIRION,
        B. (n.d.).
        Dictionary Learning for Massive Matrix Factorization Gael Varoquaux.
        Retrieved from http://proceedings.mlr.press/v48/mensch16.pdf

    'parallel_cd':
        Parallelized version of 'block_cd'.

    """
    # Check all the class are numpy or cupy
    xp = get_array_module(y, D, x)

    rng = np.random.RandomState(random_seed)
    if x is None:
        x = xp.zeros(y.shape[:-1] + (D.shape[0], ), dtype=y.dtype)

    assertion.assert_dtypes(y=y, D=D, x=x)
    assertion.assert_dtypes(mask=mask, dtypes='f')
    assertion.assert_shapes('x', x, 'D', D, axes=1)
    assertion.assert_shapes('y', y, 'D', D, axes=[-1])
    assertion.assert_shapes('y', y, 'mask', mask)

    return solve_fastpath(y, D, alpha, x, tol, minibatch, maxiter, method,
                          lasso_method, lasso_iter, lasso_tol, rng, xp,
                          mask=mask)


def solve_fastpath(y, D, alpha, x, tol, minibatch, maxiter, method,
                   lasso_method, lasso_iter, lasso_tol, rng, xp, mask=None):
    """
    Fast path for dictionary learning without any default value setting nor
    shape/dtype assertions.
    """
    if method in ['parallel_cd', 'block_cd']:
        return solve_cd(
            y, D, alpha, x, tol, minibatch, maxiter, method,
            lasso_method, lasso_iter, lasso_tol, rng, xp, mask=mask)


def solve_cd(y, D, alpha, x, tol, minibatch, maxiter, method,
             lasso_method, lasso_iter, lasso_tol, rng, xp, mask=None):
    """
    Mensch ARTHURMENSCH, A., Mairal JULIENMAIRAL, J., & Thirion BETRANDTHIRION,
    B. (n.d.).
    Dictionary Learning for Massive Matrix Factorization Gael Varoquaux.
    Retrieved from http://proceedings.mlr.press/v48/mensch16.pdf
    """
    A = xp.zeros((D.shape[0], D.shape[0]), dtype=y.dtype)
    B = xp.zeros((D.shape[0], D.shape[1]), dtype=y.dtype)

    if mask is not None:
        E = xp.zeros(D.shape[1], dtype=float_type(y.dtype))

    def update_block(D_old):
        # Original update logic
        D = D_old.copy()
        for k in range(D.shape[0]):
            uk = (B[k] - xp.dot(A[k], D)) / (A[k, k] + _JITTER) + D[k]
            # normalize
            if y.dtype.kind == 'c':
                Unorm = xp.sum(xp.real(xp.conj(uk) * uk))
            else:
                Unorm = xp.sum(uk * uk)
            D[k] = uk / xp.maximum(Unorm, 1.0)
        return D

    def update_prallel(D):
        # Parallelized update logic
        Adiag = xp.expand_dims(xp.diagonal(A), -1)
        U = (B - xp.dot(A, D)) / (Adiag + _JITTER) + D
        # normalize
        if y.dtype.kind == 'c':
            Unorm = xp.sum(xp.real(xp.conj(U) * U), axis=-1, keepdims=True)
        else:
            Unorm = xp.sum(U * U, axis=-1, keepdims=True)
        return U / xp.maximum(Unorm, 1.0)

    for it in range(1, maxiter):
        try:
            indexes = minibatch_index(y.shape, minibatch, rng)
            x_minibatch = x[indexes]
            y_minibatch = y[indexes]
            mask_minibatch = None if mask is None else mask[indexes]

            # lasso
            it2, x_minibatch = lasso.solve_fastpath(
                        y_minibatch, D, alpha, x=x_minibatch, tol=lasso_tol,
                        maxiter=lasso_iter, method=lasso_method,
                        mask=mask_minibatch, xp=xp)

            x[indexes] = x_minibatch

            # Dictionary update
            xT = x_minibatch.T
            if y.dtype.kind == 'c':
                xT = xp.conj(xT)

            it_inv = 1.0 / it
            A = (1.0 - it_inv) * A + it_inv * xp.dot(xT, x_minibatch)
            if mask is None:
                B = (1.0 - it_inv) * B + it_inv * xp.dot(xT, y_minibatch)
            else:
                mask_sum = xp.sum(mask_minibatch, axis=0)
                E = E + mask_sum
                B = B + 1.0 / E * (xp.dot(xT, y_minibatch * mask_minibatch)
                                   - mask_sum * B)

            if method == 'block_cd':
                D_new = update_block(D)
            elif method == 'parallel_cd':
                D_new = update_prallel(D)

            if xp.max(xp.abs(D - D_new)) < tol:
                return it, D, x
            D = D_new

        except KeyboardInterrupt:
            return it, D, x
    return maxiter, D, x
