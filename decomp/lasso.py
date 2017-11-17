import numpy as np
from .utils.cp_compat import get_array_module
from .utils import assertion, dtype
from .math_utils import eigen


"""
Many algorithms are taken from
http://niaohe.ise.illinois.edu/IE598/lasso_demo/index.html
"""


AVAILABLE_METHODS = ['ista', 'acc_ista', 'fista', 'parallel_cd', 'cd']
_JITTER = 1.0e-15


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
    assertion.assert_nonnegative(mask)
    assertion.assert_ndim('A', A, ndim=2)
    assertion.assert_shapes('x', x, 'A', A, axes=1)
    assertion.assert_shapes('y', y, 'x', x,
                            axes=np.arange(x.ndim - 1).tolist())
    assertion.assert_shapes('y', y, 'A', A, axes=[-1])
    if mask is not None and mask.ndim == 1:
        assertion.assert_shapes('y', y, 'mask', mask, axes=[-1])
    else:
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

    In this method, some correction takes place,

    alpha scaling:
        We changed the model from
            argmin_x {1 / (2 * n) * |y - xA|^2 - alpha |x|}
        to
            argmin_x {1 / 2 * |y - xA|^2 - alpha |x|}
        by scaling alpha by n.
        (Make sure with mask case, n is the number of valid entries)

    A scaling
        We also scale A, so that [AAt]_i,i is 1.
    """
    if mask is not None and mask.ndim == 1:
        y = y * mask
        A = A * mask
    # A scaling
    if A.dtype.kind != 'c':
        AAt_diag_sqrt = xp.sqrt(xp.sum(xp.square(A), axis=-1))
    else:
        AAt_diag_sqrt = xp.sqrt(xp.sum(xp.real(xp.conj(A) * A), axis=-1))
    A = A / xp.expand_dims(AAt_diag_sqrt, axis=-1)
    alpha = alpha / AAt_diag_sqrt
    tol = tol * AAt_diag_sqrt

    if mask is None or mask.ndim == 1:
        # alpha scaling
        if mask is not None:  # mask.ndim == 1
            alpha = alpha * xp.sum(mask, axis=-1)
        else:
            alpha = alpha * A.shape[-1]
        if method == 'ista':
            it, x = _solve_ista(y, A, alpha, x, tol=tol, maxiter=maxiter,
                                xp=xp)
        elif method == 'acc_ista':
            it, x = _solve_acc_ista(y, A, alpha, x, tol=tol, maxiter=maxiter,
                                    xp=xp)
        elif method == 'fista':
            it, x = _solve_fista(y, A, alpha, x, tol=tol, maxiter=maxiter,
                                 xp=xp)
        elif method == 'cd':
            it, x = _solve_cd(y, A, alpha, x, tol=tol, maxiter=maxiter, xp=xp)
        elif method == 'parallel_cd':
            it, x = _solve_parallel_cd(y, A, alpha, x, tol=tol,
                                       maxiter=maxiter, xp=xp)
        else:
            raise NotImplementedError('Method ' + method + ' is not yet '
                                      'implemented.')
    else:
        # alpha scaling
        alpha = alpha * xp.sum(mask, axis=-1, keepdims=True)
        if method == 'ista':
            it, x = _solve_ista_mask(y, A, alpha, x, tol=tol, maxiter=maxiter,
                                     mask=mask, xp=xp)
        elif method == 'acc_ista':
            it, x = _solve_acc_ista_mask(y, A, alpha, x, tol=tol,
                                         maxiter=maxiter, mask=mask, xp=xp)
        elif method == 'fista':
            it, x = _solve_fista_mask(y, A, alpha, x, tol=tol, maxiter=maxiter,
                                      mask=mask, xp=xp)
        elif method == 'cd':
            it, x = _solve_cd_mask(y, A, alpha, x, tol=tol, maxiter=maxiter,
                                   mask=mask, xp=xp)
        elif method == 'parallel_cd':
            it, x = _solve_parallel_cd_mask(y, A, alpha, x, tol=tol,
                                            maxiter=maxiter, mask=mask, xp=xp)
        else:
            raise NotImplementedError('Method ' + method + ' is not yet '
                                      'implemented with mask.')
    # not forget to restore x value.
    return it, x / AAt_diag_sqrt


def soft_threshold_float(x, y, xp):
    """
    soft-threasholding function for real values.

    x: a float or complex array
    y: positive float (array like)

    Returns
    -------
    if x is float
        x - y if x > y
        x + y if x < -y
        0 otherwise
    """
    sign = xp.sign(x)
    return xp.maximum(xp.abs(x) - y, 0.0) * sign


def soft_threshold_complex(x, y, xp):
    """
    soft-threasholding function for complex values.

    x: a float or complex array
    y: positive float (array like)

    Returns
    -------
    if x is complex (amplitude: r, angle: phi)
        (r - y) * exp(1j * phi) if r > y
        0 otherwise
    """
    abs_x = xp.abs(x)
    sign = x / (abs_x + _JITTER)
    return xp.maximum(abs_x - y, 0.0) * sign


def _update_float(yAt, AAt, x0, Lalpha_inv, L_inv, xp):
    dx = yAt - xp.tensordot(x0, AAt, axes=1)
    return soft_threshold_float(x0 + Lalpha_inv * dx, L_inv, xp)


def _update_complex(yAt, AAt, x0, Lalpha_inv, L_inv, xp):
    dx = yAt - xp.tensordot(x0, AAt, axes=1)
    return soft_threshold_complex(x0 + Lalpha_inv * dx, L_inv, xp)


def _update_float_mask(yAt, A, At, x0, Lalpha_inv, L_inv, mask, xp):
    dx = yAt - xp.tensordot(xp.tensordot(x0, A, axes=1) * mask, At, axes=1)
    return soft_threshold_float(x0 + Lalpha_inv * dx, L_inv, xp)


def _update_complex_mask(yAt, A, At, x0, Lalpha_inv, L_inv, mask, xp):
    dx = yAt - xp.tensordot(xp.tensordot(x0, A, axes=1) * mask, At, axes=1)
    return soft_threshold_complex(x0 + Lalpha_inv * dx, L_inv, xp)


def _solve_ista(y, A, alpha, x0, tol, maxiter, xp):
    """ Fast path to solve lasso by ista method """
    if A.dtype.kind != 'c':
        At = A.T
        updater = _update_float
    else:
        At = xp.conj(A.T)
        updater = _update_complex
    AAt = xp.dot(A, At)
    Lalpha_inv = 1.0 / eigen.spectral_radius_Gershgorin(AAt, xp)
    L_inv = Lalpha_inv * alpha

    yAt = xp.tensordot(y, At, axes=1)

    for i in range(maxiter):
        x0_new = updater(yAt, AAt, x0, Lalpha_inv, L_inv, xp=xp)
        if i % 10 == 0 and xp.max(xp.abs(x0_new - x0) - tol) < 0.0:
            return i, x0_new
        x0 = x0_new

    return maxiter - 1, x0


def mean_except_last(x, xp):
    for _ in range(x.ndim - 1):
        x = xp.mean(x, 0)
    return x


def _solve_ista_mask(y, A, alpha, x0, tol, maxiter, mask, xp):
    """ Fast path to solve lasso by ista method with missing value """
    if A.dtype.kind != 'c':
        At = A.T
        updater = _update_float_mask
    else:
        At = xp.conj(A.T)
        updater = _update_complex_mask
    AAt = xp.dot(A * mean_except_last(mask, xp), At)
    Lalpha_inv = 1.0 / eigen.spectral_radius_Gershgorin(AAt, xp)
    L_inv = Lalpha_inv * alpha

    yAt = xp.tensordot(y * mask, At, axes=1)

    for i in range(maxiter):
        x0_new = updater(yAt, A, At, x0, Lalpha_inv, L_inv, mask=mask, xp=xp)
        if i % 10 == 0 and xp.max(xp.abs(x0_new - x0) - tol) < 0.0:
            return i, x0_new
        x0 = x0_new
    return maxiter - 1, x0


def _solve_acc_ista(y, A, alpha, x0, tol, maxiter, xp):
    """ Nesterovs' Accelerated Proximal Gradient """
    if A.dtype.kind != 'c':
        At = A.T
        updater = _update_float
    else:
        At = xp.conj(A.T)
        updater = _update_complex
    AAt = xp.dot(A, At)
    Lalpha_inv = 1.0 / eigen.spectral_radius_Gershgorin(AAt, xp)
    L_inv = Lalpha_inv * alpha

    yAt = xp.tensordot(y, At, axes=1)

    v = x0
    x0_new = x0
    for i in range(maxiter):
        x0 = x0_new
        x0_new = updater(yAt, AAt, v, Lalpha_inv, L_inv, xp=xp)
        v = x0_new + i / (i + 3) * (x0_new - x0)

        if i % 10 == 0 and xp.max(xp.abs(x0_new - x0) - tol) < 0.0:
            return i, x0_new
    return maxiter - 1, x0


def _solve_acc_ista_mask(y, A, alpha, x0, tol, maxiter, mask, xp):
    """ Fast path to solve lasso by ista method with missing value """
    if A.dtype.kind != 'c':
        At = A.T
        updater = _update_float_mask
    else:
        At = xp.conj(A.T)
        updater = _update_complex_mask
    AAt = xp.dot(A * mean_except_last(mask, xp), At)
    Lalpha_inv = 1.0 / eigen.spectral_radius_Gershgorin(AAt, xp)
    L_inv = Lalpha_inv * alpha

    yAt = xp.tensordot(y * mask, At, axes=1)

    v = x0
    x0_new = x0
    for i in range(maxiter):
        x0 = x0_new
        x0_new = updater(yAt, A, At, v, Lalpha_inv, L_inv, mask=mask, xp=xp)
        v = x0_new + i / (i + 3) * (x0_new - x0)
        if i % 10 == 0 and xp.max(xp.abs(x0_new - x0) - tol) < 0.0:
            return i, x0_new
    return maxiter - 1, x0


def _solve_fista(y, A, alpha, x0, tol, maxiter, xp):
    """ Fast path to solve lasso by fista method """
    if A.dtype.kind != 'c':
        At = A.T
        updater = _update_float
    else:
        At = xp.conj(A.T)
        updater = _update_complex
    AAt = xp.dot(A, At)
    Lalpha_inv = 1.0 / eigen.spectral_radius_Gershgorin(AAt, xp)
    L_inv = Lalpha_inv * alpha

    yAt = xp.tensordot(y, At, axes=1)

    w0 = x0
    beta = 1.0
    for i in range(maxiter):
        x0_new = updater(yAt, AAt, w0, Lalpha_inv, L_inv, xp=xp)
        if i % 10 == 0 and xp.max(xp.abs(x0_new - x0) - tol) < 0.0:
            return i, x0_new
        beta_new = 0.5 * (1.0 + xp.sqrt(1.0 + 4.0 * beta * beta))
        w0 = x0_new + (beta - 1.0) / beta_new * (x0_new - x0)
        x0 = x0_new
        beta = beta_new
    return maxiter - 1, x0


def _solve_fista_mask(y, A, alpha, x0, tol, maxiter, mask, xp):
    """ Fast path to solve lasso by fista method """
    if A.dtype.kind != 'c':
        At = A.T
        updater = _update_float_mask
    else:
        At = xp.conj(A.T)
        updater = _update_complex_mask
    AAt = xp.dot(A * mean_except_last(mask, xp), At)
    Lalpha_inv = 1.0 / eigen.spectral_radius_Gershgorin(AAt, xp)
    L_inv = Lalpha_inv * alpha

    yAt = xp.tensordot(y * mask, At, axes=1)

    w0 = x0
    beta = 1.0
    for i in range(maxiter):
        x0_new = updater(yAt, A, At, w0, Lalpha_inv, L_inv, mask=mask, xp=xp)
        if i % 10 == 0 and xp.max(xp.abs(x0_new - x0) - tol) < 0.0:
            return i, x0_new
        beta_new = 0.5 * (1.0 + np.sqrt(1.0 + 4.0 * beta * beta))
        w0 = x0_new + (beta - 1.0) / beta_new * (x0_new - x0)
        x0 = x0_new
        beta = beta_new
    return maxiter - 1, x0_new


def _solve_parallel_cd(y, A, alpha, x0, tol, maxiter, xp):
    """
    Bradley, J. K., Kyrola, A., Bickson, D., & Guestrin, C. (n.d.).
    Parallel Coordinate Descent for L 1 -Regularized Loss Minimization.
    """
    if A.dtype.kind != 'c':
        At = A.T
        updater = _update_float
    else:
        At = xp.conj(A.T)
        updater = _update_complex

    rng = xp.random.RandomState(0)
    AAt = xp.dot(A, At)
    rho = eigen.spectral_radius_Gershgorin(AAt, xp)
    Lalpha_inv = 1.0
    L_inv = alpha
    p = int(A.shape[0] / rho)  # number of parallel update
    if p <= 1:
        return _solve_cd(y, A, alpha, x0, tol, maxiter, xp)

    yAt = xp.tensordot(y, At, axes=1)
    random_mask = xp.zeros(A.shape[0], dtype=dtype.float_type(y.dtype))
    random_mask[:p] = 1.0

    for i in range(maxiter):
        x0_new = updater(yAt, AAt, x0, Lalpha_inv, L_inv, xp=xp)
        dx = x0_new - x0
        if i % 10 == 0 and xp.max(xp.abs(dx) - tol) < 0.0:
            return i, x0_new
        rng.shuffle(random_mask)
        x0 += dx * random_mask

    return maxiter - 1, x0


def _solve_parallel_cd_mask(y, A, alpha, x0, tol, maxiter, mask, xp):
    """
    Bradley, J. K., Kyrola, A., Bickson, D., & Guestrin, C. (n.d.).
    Parallel Coordinate Descent for L 1 -Regularized Loss Minimization.
    """
    if A.dtype.kind != 'c':
        At = A.T
        updater = _update_float_mask
    else:
        At = xp.conj(A.T)
        updater = _update_complex_mask

    rng = xp.random.RandomState(0)
    AAt = xp.dot(A, At)
    rho = eigen.spectral_radius_Gershgorin(AAt, xp)
    Lalpha_inv = 1.0
    L_inv = alpha
    p = int(A.shape[0] / rho)  # number of parallel update
    if p <= 1:
        return _solve_cd_mask(y, A, alpha, x0, tol, maxiter, mask, xp)

    yAt = xp.tensordot(y * mask, At, axes=1)
    random_mask = xp.zeros(A.shape[0], dtype=dtype.float_type(y.dtype))
    random_mask[:p] = 1.0

    for i in range(maxiter):
        x0_new = updater(yAt, A, At, x0, Lalpha_inv, L_inv, mask=mask, xp=xp)
        dx = x0_new - x0
        if i % 10 == 0 and xp.max(xp.abs(dx) - tol) < 0.0:
            return i, x0_new
        rng.shuffle(random_mask)
        x0 += dx * random_mask

    return maxiter - 1, x0


def _solve_cd(y, A, alpha, x, tol, maxiter, xp):
    """ Fast path to solve lasso by coordinate descent """
    # Note that AAt is already normalized, i.e. AAt[i, i] == 1 for all i
    if A.dtype.kind != 'c':
        At = A.T
        soft_threshold = soft_threshold_float
    else:
        At = xp.conj(A.T)
        soft_threshold = soft_threshold_complex

    for i in range(maxiter):
        flags = []
        for k in range(x.shape[-1]):
            xA = xp.tensordot(x, A, axes=1)\
                - xp.tensordot(x[..., k:k+1], A[k:k+1], axes=1)
            x_new = xp.tensordot(y - xA, At[:, k], axes=1)
            x_new = soft_threshold(x_new, alpha[k], xp)
            if i % 10 == 0:
                flags.append(xp.max(xp.abs(x[..., k] - x_new) - tol[k]) < 0.0)
            x[..., k] = x_new

        if i % 10 == 0 and all(flags):
            return i, x
    return maxiter - 1, x


def _solve_cd_mask(y, A, alpha, x, tol, maxiter, mask, xp):
    """ Fast path to solve lasso by coordinate descent """
    # Note that AAt is already normalized, i.e. AAt[i, i] == 1 for all i
    if A.dtype.kind != 'c':
        At = A.T
        soft_threshold = soft_threshold_float
    else:
        At = xp.conj(A.T)
        soft_threshold = soft_threshold_complex
    y = y * mask

    for i in range(maxiter):
        flags = []
        for k in range(x.shape[-1]):
            # TODO In theory, the below is not accurate, but actually works...
            xA = xp.tensordot(x, A, axes=1) * mask\
                - xp.tensordot(x[..., k:k+1], A[k:k+1], axes=1)
            x_new = xp.tensordot(y - xA, At[:, k], axes=1)
            x_new = soft_threshold(x_new, alpha[..., k], xp)
            if i % 10 == 0:
                flags.append(xp.max(xp.abs(x[..., k] - x_new) - tol[k]) < 0.0)
            x[..., k] = x_new

        if i % 10 == 0 and all(flags):
            return i, x
    return maxiter - 1, x

    """  This should be true...
    AAt = xp.tensordot(A * xp.expand_dims(mask, -2), At, axes=1)  # [..., m, m]
    AAt_diag = xp.diagonal(AAt, axis1=-2, axis2=-1)  # [..., m]

    for i in range(maxiter):
        flags = []
        for k in range(x.shape[-1]):
            xA = xp.tensordot(x, A, axes=1)\
                - xp.tensordot(x[..., k:k+1], A[k:k+1], axes=1)
            x_new = xp.tensordot(y - xA * mask, At[:, k],
                                 axes=1) / AAt_diag[..., k]
            x_new = soft_threshold(x_new, alpha[..., k], xp)
    """
