from ..utils.data import minibatch_index
from ..utils import assertion, normalize
from .grads import get_gradients


_JITTER = 1.0e-15


def solve(y, D, x, tol, maxiter, likelihood, mask, xp,
          grad_x=None, grad_d=None):
    """
    updator_x, updator_d: callable.
        Custom updators.
    """
    gradients_x, gradients_d = get_gradients(likelihood, mask)
    grad_x = gradients_x[likelihood] if grad_x is None else grad_x
    grad_d = gradients_d[likelihood] if grad_d is None else grad_d

    # main iteration loop
    for it in range(1, maxiter):
        # update x
        grad_x_pos, grad_x_neg = grad_x(y, x, D, mask, xp)
        x = x * xp.maximum(grad_x_pos, 0.0) / xp.maximum(grad_x_neg, _JITTER)
        # update D
        grad_d_pos, grad_d_neg = grad_d(y, x, D, mask, xp)
        U = D * xp.maximum(grad_d_pos, 0.0) / xp.maximum(grad_d_neg, _JITTER)
        D_new = normalize.l2_strict(U, axis=-1, xp=xp)
        if xp.max(xp.abs(D - D_new)) < tol:
            return it, D_new, x
        D = D_new

    return maxiter, D, x
