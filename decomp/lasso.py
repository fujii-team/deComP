import numpy as np
from .utils.cp_compat import get_array_module
from .utils import assertion
from .math_utils import eigen


"""
Many algorithms are taken from
http://niaohe.ise.illinois.edu/IE598/lasso_demo/index.html
"""


AVAILABLE_METHODS = ['ista', 'cd', 'fista', 'acc_ista', 'cd_normalize',
                     'parallel_cd']#, 'parallel_cd']


def soft_threshold(x, y, xp):
    """
    complex-value compatible soft-threasholding function.

    x: a float or complex array
    y: positive float (array like)

    Returns
    -------
    if x is float
        x - y if x > y
        x + y if x < -y
        0 otherwise

    if x is complex (amplitude: r, angle: phi)
        (r - y) * exp(1j * phi) if r > y
        0 otherwise
    """
    if hasattr(x, 'dtype') and x.dtype.kind == 'c':
        sign = x / (xp.abs(x) + 1.0e-12 * y)
    else:
        sign = xp.sign(x)
    x = xp.abs(x) - y
    return xp.maximum(x, 0.0) * sign


def solve(y, A, alpha, x=None, tol=1.0e-3, method='ista', maxiter=1000,
          mask=None, **kwargs):
    """
    Solve Lasso problem

    argmin_x {1 / (2 * n) * |y - xA|^2 - alpha |x|}

    with
    y: [..., n_channels]
    x: [..., n_features]
    A: [n_features, n_channels]
    n: [...]

    Parameters
    ----------
    y: array-like (float or complex)
        Target data
    A: array-like (float or complex)
        A design matrix
    alpha: a positive float
        Regularization parameter
    x: array-like
        An initial estimate of x (optional)
    tol: a float
        Criterion to stop iteration
    method: string
        'ista' | 'fista' | 'cd'

        For ista and fista, see
            Beck, A., & Teboulle, M. (n.d.).
            A Fast Iterative Shrinkage-Thresholding Algorithm for Linear
            Inverse Problems *, 2(1), 183-202.
            http://doi.org/10.1137/080716542
        for the details.

        cd: coordinate descent.
            This method is very slow because it is difficult to parallelize
            this algorithm. It is just for the reference purpose.
    """
    # Check all the class are numpy or cupy
    xp = get_array_module(y, A, x, mask)

    if x is None:
        x = xp.zeros(y.shape[:-1] + (A.shape[0], ), dtype=y.dtype)

    assertion.assert_dtypes(y=y, A=A, x=x)
    assertion.assert_dtypes(mask=mask, dtypes='f')
    assertion.assert_shapes('x', x, 'A', A, axes=1)
    assertion.assert_shapes('y', y, 'x', x,
                            axes=np.arange(x.ndim - 1).tolist())
    assertion.assert_shapes('y', y, 'A', A, axes=[-1])
    assertion.assert_shapes('y', y, 'mask', mask)

    if method not in AVAILABLE_METHODS:
        raise ValueError('Available methods are {0:s}. Given {1:s}'.format(
                            str(AVAILABLE_METHODS), method))

    return solve_fastpath(y, A, alpha, x, tol, maxiter, method, xp, mask=mask,
                          **kwargs)


def solve_fastpath(y, A, alpha, x, tol, maxiter, method, xp, mask=None,
                   **kwargs):
    """ fast path for lasso, without default value setting and shape/dtype
    assertions.
    """
    if mask is None:
        if method == 'ista':
            return solve_ista(y, A, alpha, x, tol=tol, maxiter=maxiter,
                              xp=xp)
        elif method == 'acc_ista':
            return solve_acc_ista(y, A, alpha, x, tol=tol, maxiter=maxiter,
                                  xp=xp)
        elif method == 'fista':
            return solve_fista(y, A, alpha, x, tol=tol, maxiter=maxiter,
                               xp=xp)
        elif method == 'cd':
            return solve_cd(y, A, alpha, x, tol=tol, maxiter=maxiter, xp=xp)
        elif method == 'cd_normalize':
            return solve_cd_normalize(y, A, alpha, x, tol=tol, maxiter=maxiter,
                                      xp=xp)
        elif method == 'parallel_cd':
            return solve_parallel_cd(y, A, alpha, x, tol=tol, maxiter=maxiter,
                                     xp=xp)
        elif method == 'parallel_cd':
            return solve_parallel_cd(y, A, alpha, x, tol=tol, maxiter=maxiter,
                                     xp=xp)
        else:
            raise NotImplementedError('Method ' + method + ' is not yet '
                                      'implemented.')
    else:
        if method == 'ista':
            return solve_ista_mask(y, A, alpha, x, tol=tol, maxiter=maxiter,
                                   mask=mask, xp=xp)
        elif method == 'acc_ista':
            return solve_acc_ista_mask(y, A, alpha, x, tol=tol,
                                       maxiter=maxiter, mask=mask, xp=xp)
        elif method == 'fista':
            return solve_fista_mask(y, A, alpha, x, tol=tol, maxiter=maxiter,
                                    mask=mask, xp=xp)
        elif method == 'cd':
            return solve_cd_mask(y, A, alpha, x, tol=tol, maxiter=maxiter,
                                 mask=mask, xp=xp)
        else:
            raise NotImplementedError('Method ' + method + ' is not yet '
                                      'implemented with mask.')


def solve_ista(y, A, alpha, x0, tol, maxiter, xp):
    """ Fast path to solve lasso by ista method """
    At = A.T if A.dtype.kind != 'c' else xp.conj(A.T)
    AAt = xp.dot(A, At)
    L = eigen.spectral_radius_Gershgorin(AAt, xp)

    yAt = xp.tensordot(y, At, axes=1)
    alpha = alpha * A.shape[-1]

    for i in range(maxiter):
        x0_new = _update(yAt, AAt, x0, L, alpha, xp=xp)
        if xp.max(xp.abs(x0_new - x0)) < tol:
            return i, x0_new
        else:
            x0 = x0_new

    return maxiter - 1, x0


def solve_acc_ista(y, A, alpha, x0, tol, maxiter, xp):
    """ Nesterovs' Accelerated Proximal Gradient """
    At = A.T if A.dtype.kind != 'c' else xp.conj(A.T)
    AAt = xp.dot(A, At)
    L = eigen.spectral_radius_Gershgorin(AAt, xp)

    yAt = xp.tensordot(y, At, axes=1)
    alpha = alpha * A.shape[-1]

    v = x0
    x0_new = x0
    for i in range(maxiter):
        x0 = x0_new
        x0_new = _update(yAt, AAt, v, L, alpha, xp=xp)
        v = x0_new + i / (i + 3) * (x0_new - x0)

        if xp.max(xp.abs(x0_new - x0)) < tol:
            return i, x0_new
    return maxiter - 1, x0


def solve_fista(y, A, alpha, x0, tol, maxiter, xp):
    """ Fast path to solve lasso by fista method """
    At = A.T if A.dtype.kind != 'c' else xp.conj(A.T)
    AAt = xp.dot(A, At)
    L = eigen.spectral_radius_Gershgorin(AAt, xp)

    yAt = xp.tensordot(y, At, axes=1)
    alpha = alpha * A.shape[-1]

    w0 = x0
    beta = 1.0
    for i in range(maxiter):
        x0_new = _update(yAt, AAt, w0, L, alpha, xp=xp)
        if xp.max(xp.abs(x0_new - x0)) < tol:
            return i, x0_new
        else:
            beta_new = 0.5 * (1.0 + xp.sqrt(1.0 + 4.0 * beta * beta))
            w0 = x0_new + (beta - 1.0) / beta_new * (x0_new - x0)
            x0 = x0_new
            beta = beta_new
    return maxiter - 1, x0


def _update(yAt, AAt, x0, L, alpha, xp):
    """
    1 iteration by ISTA method.
    This is also used as fista, where w0 is passed instead of x0
    """
    dx = yAt - xp.tensordot(x0, AAt, axes=1)
    return soft_threshold(x0 + 1.0 / (L * alpha) * dx, 1.0 / L, xp)


def solve_ista_mask(y, A, alpha, x0, tol, maxiter, mask, xp):
    """ Fast path to solve lasso by ista method with missing value """
    At = A.T if A.dtype.kind != 'c' else xp.conj(A.T)
    AAt = xp.dot(A, At)
    L = eigen.spectral_radius_Gershgorin(AAt, xp)
    # here, alpha is sample dependent, because of the mask
    alpha = alpha * xp.sum(mask, axis=-1, keepdims=True)

    yAt = xp.tensordot(y * mask, At, axes=1)

    for i in range(maxiter):
        x0_new = _update_w_mask(yAt, A, At, x0, L, alpha, mask=mask, xp=xp)
        if xp.max(xp.abs(x0_new - x0)) < tol:
            return i, x0_new
        else:
            x0 = x0_new
    return maxiter - 1, x0


def solve_acc_ista_mask(y, A, alpha, x0, tol, maxiter, mask, xp):
    """ Fast path to solve lasso by ista method with missing value """
    At = A.T if A.dtype.kind != 'c' else xp.conj(A.T)
    AAt = xp.dot(A, At)
    L = eigen.spectral_radius_Gershgorin(AAt, xp)
    # here, alpha is sample dependent, because of the mask
    alpha = alpha * xp.sum(mask, axis=-1, keepdims=True)

    yAt = xp.tensordot(y * mask, At, axes=1)

    v = x0
    x0_new = x0
    for i in range(maxiter):
        x0 = x0_new
        x0_new = _update_w_mask(yAt, A, At, v, L, alpha, mask=mask, xp=xp)
        v = x0_new + i / (i + 3) * (x0_new - x0)
        if xp.max(xp.abs(x0_new - x0)) < tol:
            return i, x0_new
        else:
            x0 = x0_new
    return maxiter - 1, x0


def solve_fista_mask(y, A, alpha, x0, tol, maxiter, mask, xp):
    """ Fast path to solve lasso by fista method """
    At = A.T if A.dtype.kind != 'c' else xp.conj(A.T)
    AAt = xp.dot(A, At)
    L = eigen.spectral_radius_Gershgorin(AAt, xp)
    # here, alpha is sample dependent, because of the mask
    alpha = alpha * xp.sum(mask, axis=-1, keepdims=True)

    yAt = xp.tensordot(y * mask, At, axes=1)

    w0 = x0
    beta = 1.0
    for i in range(maxiter):
        x0_new = _update_w_mask(yAt, A, At, w0, L, alpha, mask=mask, xp=xp)
        if xp.max(xp.abs(x0_new - x0)) < tol:
            return i, x0_new
        else:
            beta_new = 0.5 * (1.0 + np.sqrt(1.0 + 4.0 * beta * beta))
            w0 = x0_new + (beta - 1.0) / beta_new * (x0_new - x0)
            x0 = x0_new
            beta = beta_new
    return maxiter - 1, x0


def _update_w_mask(yAt, A, At, x0, L, alpha, mask, xp):
    """
    1 iteration by ISTA method with missing value
    This is also used as fista, where w0 is passed instead of x0
    """
    dx = yAt - xp.tensordot(xp.tensordot(x0, A, axes=1) * mask, At, axes=1)
    return soft_threshold(x0 + 1.0 / (L * alpha) * dx, 1.0 / L, xp)


class MatrixNormalizer(object):
    """ Normalize A so that [AAt]_{i, i} == 1.
    alpha value should be also modified.
    """
    def __init__(self, A, alpha, xp):
        self.AAt_diag_sqrt = xp.sqrt(xp.sum(xp.square(A), axis=-1))
        self.A = A / xp.expand_dims(self.AAt_diag_sqrt, axis=-1)
        self.alpha = alpha / self.AAt_diag_sqrt

    def restore(self, x):
        return x / self.AAt_diag_sqrt

    def get_tol(self, tol):
        return tol * self.AAt_diag_sqrt


class MatrixNormalizerComplex(MatrixNormalizer):
    def __init__(self, A, alpha, xp):
        self.AAt_diag_sqrt = xp.sqrt(xp.sum(xp.real(xp.conj(A) * A), axis=-1))
        self.A = A / xp.expand_dims(self.AAt_diag_sqrt, axis=-1)
        self.alpha = alpha / self.AAt_diag_sqrt


def solve_cd(y, A, alpha, x, tol, maxiter, xp):
    """ Fast path to solve lasso by coordinate descent with mask """
    At = A.T if A.dtype.kind != 'c' else xp.conj(A.T)
    AAt = xp.dot(A, At)
    alpha = alpha * A.shape[-1]

    for i in range(maxiter):
        flags = []
        for k in range(x.shape[-1]):
            xA = xp.tensordot(x, A, axes=1)\
                - xp.tensordot(x[..., k:k+1], A[k:k+1], axes=1)
            x_new = xp.tensordot(y - xA, At[:, k], axes=1)
            x_new = soft_threshold(x_new, alpha, xp) / AAt[k, k]
            flags.append(xp.max(xp.abs(x[..., k] - x_new)) < tol)
            x[..., k] = x_new

        if all(flags):
            return i, x
    return maxiter - 1, x


def solve_cd_normalize(y, A, alpha, x, tol, maxiter, xp):
    """ Fast path to solve lasso by coordinate descent with mask """
    if A.dtype.kind != 'c':
        normalizer = MatrixNormalizer(A, alpha, xp)
        A, alpha = normalizer.A, normalizer.alpha
        At = A.T
    else:
        normalizer = MatrixNormalizerComplex(A, alpha, xp)
        A, alpha = normalizer.A, normalizer.alpha
        At = xp.conj(A.T)

    alpha = alpha * A.shape[-1]
    tol = normalizer.get_tol(tol)
    for i in range(maxiter):
        flags = []
        for k in range(x.shape[-1]):
            xA = xp.tensordot(x, A, axes=1)\
                - xp.tensordot(x[..., k:k+1], A[k:k+1], axes=1)
            x_new = xp.tensordot(y - xA, At[:, k], axes=1)
            x_new = soft_threshold(x_new, alpha[k], xp)
            flags.append(xp.max(xp.abs(x[..., k] - x_new)) < tol[k])
            x[..., k] = x_new

        if all(flags):
            return i, normalizer.restore(x)
    return maxiter - 1, normalizer.restore(x)


def solve_parallel_cd(y, A, alpha, x0, tol, maxiter, xp):
    """
    # TODO
    Bradley, J. K., Kyrola, A., Bickson, D., & Guestrin, C. (n.d.).
    Parallel Coordinate Descent for L 1 -Regularized Loss Minimization.
    """
    if A.dtype.kind != 'c':
        normalizer = MatrixNormalizer(A, alpha, xp)
        A, alpha = normalizer.A, normalizer.alpha
        At = A.T
    else:
        normalizer = MatrixNormalizerComplex(A, alpha, xp)
        A, alpha = normalizer.A, normalizer.alpha
        At = xp.conj(A.T)

    rng = xp.random.RandomState(0)
    AAt = xp.dot(A, At)
    rho = eigen.spectral_radius_Gershgorin(AAt, xp)
    p = xp.rint(A.shape[0] / rho)  # number of parallel update
    if p <= 1:
        raise ValueError
        return solve_cd(y, A, alpha, x0, tol, maxiter, xp)

    yAt = xp.tensordot(y, At, axes=1)
    alpha = alpha * A.shape[-1]

    for i in range(maxiter):
        x0_new = _update(yAt, AAt, x0, 1.0, alpha, xp=xp)
        dx = x0_new - x0
        if xp.max(xp.abs(dx)) < tol:
            return i, x0_new
        else:
            index = rng.choise(A.shape[0], p)
            x0[..., index] += dx[..., index]

    return maxiter - 1, x0

def solve_cd_mask(y, A, alpha, x, tol, maxiter, mask, xp):
    """ Fast path to solve lasso by coordinate descent """
    At = A.T if A.dtype.kind != 'c' else xp.conj(A.T)
    AAt = xp.dot(A, At)
    # here, alpha is sample dependent, because of the mask
    alpha = alpha * xp.sum(mask, axis=-1)
    y = y * mask

    for i in range(maxiter):
        flags = []
        for k in range(x.shape[-1]):
            xA = xp.tensordot(x, A, axes=1) * mask\
                - xp.tensordot(x[..., k:k+1], A[k:k+1], axes=1)
            x_new = xp.tensordot(y - xA, At[:, k], axes=1)
            x_new = soft_threshold(x_new, alpha, xp) / AAt[k, k]
            flags.append(xp.max(xp.abs(x[..., k] - x_new)) < tol)
            x[..., k] = x_new

        if all(flags):
            return i, x
    return maxiter - 1, x
