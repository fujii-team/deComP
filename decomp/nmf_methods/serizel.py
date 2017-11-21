from ..utils.data import SequentialMinibatchData
from ..utils.cp_compat import get_array_module
from ..utils import assertion, normalize
from .grads import get_gradients


_JITTER = 1.0e-15


def solve(y, D, x, tol, minibatch, maxiter, method,
          likelihood, mask, rng, xp,
          grad_x=None, grad_d=None, forget_rate=0.5):
    """
    Implementations of
    Serizel, R., Essid, S., & Richard, G. (2016).
    MINI-BATCH STOCHASTIC APPROACHES FOR ACCELERATED MULTIPLICATIVE UPDATES IN NONNEGATIVE MATRIX FACTORISATION WITH BETA-DIVERGENCE, 13-16.
    """
    gradients_x, gradients_d = get_gradients(likelihood, mask)
    grad_x = gradients_x[likelihood] if grad_x is None else grad_x
    grad_d = gradients_d[likelihood] if grad_d is None else grad_d

    if method == 'asg-mu':
        return solve_asg_mu(y, D, x, tol, minibatch, maxiter,
                            mask, rng, xp, grad_x, grad_d)
    elif method == 'gsg-mu':
        return solve_asg_mu(y, D, x, tol, minibatch, maxiter,
                            mask, rng, xp, grad_x, grad_d)
    elif method == 'asag-mu':
        return solve_asag_mu(y, D, x, tol, minibatch, maxiter,
                             mask, rng, xp, grad_x, grad_d,
                             forget_rate)
    elif method == 'gsag-mu':
        return solve_gsag_mu(y, D, x, tol, minibatch, maxiter,
                             mask, rng, xp, grad_x, grad_d,
                             forget_rate)
    raise NotImplementedError('NMF with {} algorithm is not yet '
                              'implemented.'.format(method))


def shuffle(y, x, mask, rng):
    xp = get_array_module(y.array)
    index = xp.arange(len(y.array))
    rng.shuffle(index)
    y.shuffle(index)
    x.shuffle(index)
    mask.shuffle(index)


def solve_asg_mu(y, D, x, tol, minibatch, maxiter,
                 mask, rng, xp, grad_x, grad_d):
    """ Algorithm 5 in the paper """
    for it in range(1, maxiter):
        shuffle(y, x, mask, rng)

        for y_minibatch, x_minibatch, mask_minibatch in zip(y, x, mask):
            grad_x_pos, grad_x_neg = grad_x(
                            y_minibatch, x_minibatch, D, mask_minibatch, xp)
            x_minibatch[:] = x_minibatch * \
                xp.maximum(grad_x_pos, 0.0) / xp.maximum(grad_x_neg, _JITTER)

            # update D
            grad_D_pos, grad_D_neg = grad_d(y_minibatch, x_minibatch, D,
                                            mask_minibatch, xp)
            D_new = D * xp.maximum(grad_D_pos, 0.0) / xp.maximum(grad_D_neg,
                                                                 _JITTER)
            D_new = normalize.l2_strict(D_new, axis=-1, xp=xp)
            if xp.max(xp.abs(D - D_new)) < tol:
                return it, D, x.array
            D = D_new

    return maxiter, D, x.array


def solve_gsg_mu(y, D, x, tol, minibatch, maxiter,
                 mask, rng, xp, grad_x, grad_d):
    """ Algorithm 6 in the paper """
    for it in range(1, maxiter):
        shuffle(y, x, mask, rng)

        for y_minibatch, x_minibatch, mask_minibatch in zip(y, x, mask):
            grad_x_pos, grad_x_neg = grad_x(
                            y_minibatch, x_minibatch, D, mask_minibatch, xp)
            x_minibatch[:] = x_minibatch * \
                xp.maximum(grad_x_pos, 0.0) / xp.maximum(grad_x_neg, _JITTER)

        # update D
        grad_D_pos, grad_D_neg = grad_d(y_minibatch, x_minibatch, D,
                                        mask_minibatch, xp)
        D_new = D * xp.maximum(grad_D_pos, 0.0) / xp.maximum(grad_D_neg,
                                                             _JITTER)
        D_new = normalize.l2_strict(D_new, axis=-1, xp=xp)
        if xp.max(xp.abs(D - D_new)) < tol:
            return it, D, x.array
        D = D_new

    return maxiter, D, x.array


def solve_asag_mu(y, D, x, tol, minibatch, maxiter,
                  mask, rng, xp, grad_x, grad_d, forget_rate):
    """ Algorithm 7 in the paper """
    def accumurate_grad(grad_sum, grad):
        return (1.0 - forget_rate) * grad_sum + forget_rate * grad

    for it in range(1, maxiter):
        shuffle(y, x, mask, rng)

        grad_D_pos_sum = xp.zeros_like(D)
        grad_D_neg_sum = xp.zeros_like(D)

        for y_minibatch, x_minibatch, mask_minibatch in zip(y, x, mask):
            grad_x_pos, grad_x_neg = grad_x(
                            y_minibatch, x_minibatch, D, mask_minibatch, xp)
            x_minibatch[:] = x_minibatch * \
                xp.maximum(grad_x_pos, 0.0) / xp.maximum(grad_x_neg, _JITTER)

            # update D
            grad_D_pos, grad_D_neg = grad_d(y_minibatch, x_minibatch, D,
                                            mask_minibatch, xp)
            grad_D_pos_sum = accumurate_grad(grad_D_pos_sum, grad_D_pos)
            grad_D_neg_sum = accumurate_grad(grad_D_neg_sum, grad_D_neg)

            D_new = D * xp.maximum(grad_D_pos_sum, 0.0) \
                        / xp.maximum(grad_D_neg_sum, _JITTER)
            D_new = normalize.l2_strict(D_new, axis=-1, xp=xp)
            if xp.max(xp.abs(D - D_new)) < tol:
                return it, D, x.array
            D = D_new

    return maxiter, D, x.array


def solve_gsag_mu(y, D, x, tol, minibatch, maxiter,
                  mask, rng, xp, grad_x, grad_d, forget_rate):
    """ Algorithm 7 in the paper """
    def accumurate_grad(grad_sum, grad):
        return (1.0 - forget_rate) * grad_sum + forget_rate * grad

    for it in range(1, maxiter):
        shuffle(y, x, mask, rng)

        grad_D_pos_sum = xp.zeros_like(D)
        grad_D_neg_sum = xp.zeros_like(D)

        for y_minibatch, x_minibatch, mask_minibatch in zip(y, x, mask):
            grad_x_pos, grad_x_neg = grad_x(
                            y_minibatch, x_minibatch, D, mask_minibatch, xp)
            x_minibatch[:] = x_minibatch * \
                xp.maximum(grad_x_pos, 0.0) / xp.maximum(grad_x_neg, _JITTER)

            # update D
            grad_D_pos, grad_D_neg = grad_d(y_minibatch, x_minibatch, D,
                                            mask_minibatch, xp)
            grad_D_pos_sum = accumurate_grad(grad_D_pos_sum, grad_D_pos)
            grad_D_neg_sum = accumurate_grad(grad_D_neg_sum, grad_D_neg)

        D_new = D * xp.maximum(grad_D_pos_sum, 0.0)\
                        / xp.maximum(grad_D_neg_sum, _JITTER)
        D_new = normalize.l2_strict(D_new, axis=-1, xp=xp)
        if xp.max(xp.abs(D - D_new)) < tol:
            return it, D, x.array
        D = D_new

    return maxiter, D, x.array
