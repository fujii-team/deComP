from ..utils.data import minibatch_index
from ..utils import assertion, normalize


_JITTER = 1.0e-15


def solve(y, D, x, tol, maxiter, likelihood, mask, xp,
          updator_x=None, updator_d=None):
    """
    updator_x, updator_d: callable.
        Custom updators.
    """
    if mask is None:
        # poisson liklihood is essentially the same to KL distance
        updators_x = {'l2': updator_x_l2,
                      'kl': updator_x_kl,
                      'poisson': updator_x_kl}

        updators_d = {'l2': updator_d_l2,
                      'kl': updator_d_kl,
                      'poisson': updator_d_kl}
    else:
        updators_x = {'l2': updator_x_l2_mask,
                      'kl': updator_x_kl_mask,
                      'poisson': updator_x_kl_mask}
        updators_d = {'l2': updator_d_l2_mask,
                      'kl': updator_d_kl_mask,
                      'poisson': updator_d_kl_mask}

    updator_x = updators_x[likelihood] if updator_x is None else updator_x
    updator_d = updators_d[likelihood] if updator_d is None else updator_d

    # main iteration loop
    for it in range(1, maxiter):
        # update x
        x = updator_x(y, x, D, mask, xp)
        # update D
        U = updator_d(y, x, D, mask, xp)
        D_new = normalize.l2_strict(U, axis=-1, xp=xp)
        if xp.max(xp.abs(D - D_new)) < tol:
            return it, D_new, x
        D = D_new

    return maxiter, D, x


# --- l2 loss ---
def updator_x_l2(y, x, d, mask, xp):
    """ Multiplicative update rule for square loss.
    Returns Positive (numerator) and negative (denominator).
    mask is not used.
    """
    f = xp.dot(x, d)
    return x * xp.maximum(xp.dot(y, d.T), 0.0) / xp.maximum(
                                                    xp.dot(f, d.T), _JITTER)


def updator_d_l2(y, x, d, mask, xp):
    """ update d with l2 loss """
    f = xp.dot(x, d)
    return d * xp.maximum(xp.dot(x.T, y), 0.0) / xp.maximum(
                                                    xp.dot(x.T, f), _JITTER)


def updator_x_l2_mask(y, x, d, mask, xp):
    """ Multiplicative update rule for square loss.
    Returns Positive (numerator) and negative (denominator).
    mask is not used.
    """
    f = xp.dot(x, d) * mask
    y = y * mask
    return x * xp.maximum(xp.dot(y, d.T), 0.0) / xp.maximum(
                                                    xp.dot(f, d.T), _JITTER)


def updator_d_l2_mask(y, x, d, mask, xp):
    """ update d with l2 loss """
    f = xp.dot(x, d) * mask
    y = y * mask
    return d * xp.maximum(xp.dot(x.T, y), 0.0) / xp.maximum(
                                                    xp.dot(x.T, f), _JITTER)


# --- KL loss ---
def updator_x_kl(y, x, d, mask, xp):
    """ Multiplicative update rule for KL loss.    """
    f = xp.dot(x, d) + _JITTER
    return x * xp.maximum(xp.dot(y / f, d.T), 0.) / xp.maximum(
                                xp.sum(d.T, axis=0, keepdims=True), _JITTER)


def updator_d_kl(y, x, d, mask, xp):
    """ update d with KL loss """
    f = xp.dot(x, d) + _JITTER
    return d * xp.maximum(xp.dot(x.T, y / f), 0.) / xp.maximum(
                                xp.sum(x.T, axis=1, keepdims=True), _JITTER)


def updator_x_kl_mask(y, x, d, mask, xp):
    """ Multiplicative update rule for KL loss with mask. """
    f = xp.dot(x, d) + _JITTER
    y = y * mask
    return x * xp.maximum(xp.dot(y / f, d.T), 0.) / xp.maximum(
                                                xp.dot(mask, d.T), _JITTER)


def updator_d_kl_mask(y, x, d, mask, xp):
    """ update d with l2 loss """
    f = xp.dot(x, d) + _JITTER
    y = y * mask
    return d * xp.maximum(xp.dot(x.T, y / f), 0.) / xp.maximum(
                                                xp.dot(x.T, mask), _JITTER)
