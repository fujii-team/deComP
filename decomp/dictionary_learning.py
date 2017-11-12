import numpy as np
from .utils.cp_compat import get_array_module
from .utils.data import minibatch_index
from .utils import assertion, normalize
from . import lasso


_JITTER = 1.0e-15


def solve(y, D, alpha, x=None, tol=1.0e-3,
          minibatch=None, maxiter=1000, method='block_cd',
          lasso_method='cd', lasso_iter=10, lasso_tol=1.0e-5,
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
        Mairal, J., Bach FRANCISBACH, F., Ponce JEANPONCE, J., & Sapiro, G. (n.d.)
        Online Dictionary Learning for Sparse Coding.

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
    if method == 'block_cd':
        return solve_cd(
            y, D, alpha, x, tol, minibatch, maxiter,
            lasso_method, lasso_iter, lasso_tol, rng, xp, mask=mask)
    elif method == 'block_cd_fullbatch':
        return solve_block_cd_fullbatch(
            y, D, alpha, x, tol, maxiter,
            lasso_method, lasso_iter, lasso_tol, rng, xp, mask=mask)
    else:
        raise NotImplementedError('Method %s is not yet '
                                  ' implemented'.format(method))


def solve_block_cd_fullbatch(y, D, alpha, x, tol, maxiter,
                             lasso_method, lasso_iter, lasso_tol, rng, xp,
                             mask=None):
    """ This algorithm is prepared for the reference purpose. """
    # Normalize first
    D = normalize.l2_strict(D, axis=-1, xp=xp)
    # iteration loop
    for it in range(1, maxiter):
        try:
            # lasso
            it2, x = lasso.solve_fastpath(
                        y, D, alpha, x=x, tol=lasso_tol,
                        maxiter=lasso_iter, method=lasso_method,
                        mask=mask, xp=xp)
            # Dictionary update
            xT = x.T
            if y.dtype.kind == 'c':
                xT = xp.conj(xT)
            A = xp.dot(xT, x)
            B = xp.dot(xT, y)
            # dictionary update method
            D_new = D.copy()
            for k in range(D_new.shape[0]):
                uk = (B[k] - xp.dot(A[k], D_new)) / A[k, k] + D_new[k]
                D_new[k] = normalize.l2(uk, xp, axis=-1)

            if xp.sum(xp.abs(D - D_new)) < tol:
                return it, D_new, x
            D = D_new

        except KeyboardInterrupt:
            return it, D, x
    return maxiter, D, x


def solve_cd(y, D, alpha, x, tol, minibatch, maxiter,
             lasso_method, lasso_iter, lasso_tol, rng, xp, mask=None):
    """
    Mairal, J., Bach FRANCISBACH, F., Ponce JEANPONCE, J., & Sapiro, G. (n.d.)
    Online Dictionary Learning for Sparse Coding.
    """
    A = xp.zeros((D.shape[0], D.shape[0]), dtype=y.dtype)
    B = xp.zeros((D.shape[0], D.shape[1]), dtype=y.dtype)

    # Normalize first
    D = normalize.l2_strict(D, axis=-1, xp=xp)

    # iteration loop
    for it in range(1, maxiter):
        try:
            indexes = minibatch_index(y.shape[:-1], minibatch, rng)
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

            # equation (11)
            theta_plus1 = it * minibatch + 1.0
            beta = (theta_plus1 - minibatch) / theta_plus1

            A = beta * A + xp.dot(xT, x_minibatch)
            B = beta * B + xp.dot(xT, y_minibatch)

            D_new = D.copy()
            for k in range(D_new.shape[0]):
                uk = (B[k] - xp.dot(A[k], D_new)) / (A[k, k] + _JITTER)\
                     + D_new[k]
                # normalize
                D_new[k] = normalize.l2(uk, xp, axis=-1)

            if xp.max(xp.abs(D - D_new)) < tol:
                return it, D_new, x
            D = D_new

        except KeyboardInterrupt:
            return it, D, x
    return maxiter, D, x
