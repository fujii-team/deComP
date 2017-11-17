from ..utils.data import MinibatchEpochIndex
from ..utils import assertion, normalize


_JITTER = 1.0e-15


def solve(y, D, x, tol, minibatch, maxiter, method,
          likelihood, mask, rng, xp,
          grad_x=None, grad_d=None, forget_rate=0.5):
    """
    Implementations of
    Serizel, R., Essid, S., & Richard, G. (2016).
    MINI-BATCH STOCHASTIC APPROACHES FOR ACCELERATED MULTIPLICATIVE UPDATES IN NONNEGATIVE MATRIX FACTORISATION WITH BETA-DIVERGENCE, 13–16.
    """
    # mini-batch methods
    if mask is None:
        gradients_x = {'l2': _grad_x_l2,
                       'kl': _grad_x_kl,
                       'poisson': _grad_x_kl}
        gradients_d = {'l2': _grad_d_l2,
                       'kl': _grad_d_kl,
                       'poisson': _grad_d_kl}
    else:
        gradients_x = {'l2': _grad_x_l2_mask,
                       'kl': _grad_x_kl_mask,
                       'poisson': _grad_x_kl_mask}
        gradients_d = {'l2': _grad_d_l2_mask,
                       'kl': _grad_d_kl_mask,
                       'poisson': _grad_d_kl_mask}

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


def solve_asg_mu(y, D, x, tol, minibatch, maxiter,
                 mask, rng, xp, grad_x, grad_d):
    """ Algorithm 5 in the paper """
    minibatch_index = MinibatchEpochIndex(len(y), minibatch, rng, xp)

    for it in range(1, maxiter):
        minibatch_index.shuffle()

        for index in minibatch_index:
            x_minibatch = x[index]
            y_minibatch = y[index]
            mask_minibatch = mask[index] if mask is not None else None

            grad_x_pos, grad_x_neg = grad_x(
                            y_minibatch, x_minibatch, D, mask_minibatch, xp)
            x_minibatch = x_minibatch * \
                xp.maximum(grad_x_pos, 0.0) / xp.maximum(grad_x_neg, _JITTER)
            x[index] = x_minibatch

            # update D
            grad_D_pos, grad_D_neg = grad_d(y_minibatch, x_minibatch, D,
                                            mask_minibatch, xp)
            U = D * xp.maximum(grad_D_pos, 0.0) / xp.maximum(grad_D_neg,
                                                             _JITTER)
            D_new = normalize.l2_strict(U, axis=-1, xp=xp)
            if xp.max(xp.abs(D - D_new)) < tol:
                return it, D, x
            D = D_new

    return maxiter, D, x


def solve_gsg_mu(y, D, x, tol, minibatch, maxiter,
                 mask, rng, xp, grad_x, grad_d):
    """ Algorithm 6 in the paper """
    minibatch_index = MinibatchEpochIndex(len(y), minibatch, rng, xp)

    for it in range(1, maxiter):
        minibatch_index.shuffle()

        for index in minibatch_index:
            x_minibatch = x[index]
            y_minibatch = y[index]
            mask_minibatch = mask[index] if mask is not None else None

            grad_x_pos, grad_x_neg = grad_x(
                            y_minibatch, x_minibatch, D, mask_minibatch, xp)
            x_minibatch = x_minibatch * \
                xp.maximum(grad_x_pos, 0.0) / xp.maximum(grad_x_neg, _JITTER)
            x[index] = x_minibatch

        # update D
        grad_D_pos, grad_D_neg = grad_d(y_minibatch, x_minibatch, D,
                                        mask_minibatch, xp)
        U = D * xp.maximum(grad_D_pos, 0.0) / xp.maximum(grad_D_neg,
                                                         _JITTER)
        D_new = normalize.l2_strict(U, axis=-1, xp=xp)
        if xp.max(xp.abs(D - D_new)) < tol:
            return it, D, x
        D = D_new

    return maxiter, D, x


def solve_asag_mu(y, D, x, tol, minibatch, maxiter,
                  mask, rng, xp, grad_x, grad_d, forget_rate):
    """ Algorithm 7 in the paper """
    minibatch_index = MinibatchEpochIndex(len(y), minibatch, rng, xp)

    def accumurate_grad(grad_sum, grad):
        return (1.0 - forget_rate) * grad_sum + forget_rate * grad

    for it in range(1, maxiter):
        minibatch_index.shuffle()

        grad_D_pos_sum = xp.zeros_like(D)
        grad_D_neg_sum = xp.zeros_like(D)
        for index in minibatch_index:
            x_minibatch = x[index]
            y_minibatch = y[index]
            mask_minibatch = mask[index] if mask is not None else None

            grad_x_pos, grad_x_neg = grad_x(
                            y_minibatch, x_minibatch, D, mask_minibatch, xp)
            x_minibatch = x_minibatch * \
                xp.maximum(grad_x_pos, 0.0) / xp.maximum(grad_x_neg, _JITTER)
            x[index] = x_minibatch

            # update D
            grad_D_pos, grad_D_neg = grad_d(y_minibatch, x_minibatch, D,
                                            mask_minibatch, xp)
            grad_D_pos_sum = accumurate_grad(grad_D_pos_sum, grad_D_pos)
            grad_D_neg_sum = accumurate_grad(grad_D_neg_sum, grad_D_neg)

            U = D * xp.maximum(grad_D_pos_sum, 0.0) / xp.maximum(grad_D_neg_sum,
                                                                 _JITTER)
            D_new = normalize.l2_strict(U, axis=-1, xp=xp)
            if xp.max(xp.abs(D - D_new)) < tol:
                return it, D, x
            D = D_new

    return maxiter, D, x


def solve_gsag_mu(y, D, x, tol, minibatch, maxiter, 
                  mask, rng, xp, grad_x, grad_d, forget_rate):
    """ Algorithm 7 in the paper """
    minibatch_index = MinibatchEpochIndex(len(y), minibatch, rng, xp)

    def accumurate_grad(grad_sum, grad):
        return (1.0 - forget_rate) * grad_sum + forget_rate * grad

    for it in range(1, maxiter):
        minibatch_index.shuffle()

        grad_D_pos_sum = xp.zeros_like(D)
        grad_D_neg_sum = xp.zeros_like(D)
        for index in minibatch_index:
            x_minibatch = x[index]
            y_minibatch = y[index]
            mask_minibatch = mask[index] if mask is not None else None

            grad_x_pos, grad_x_neg = grad_x(
                            y_minibatch, x_minibatch, D, mask_minibatch, xp)
            x_minibatch = x_minibatch * \
                xp.maximum(grad_x_pos, 0.0) / xp.maximum(grad_x_neg, _JITTER)
            x[index] = x_minibatch

            # update D
            grad_D_pos, grad_D_neg = grad_d(y_minibatch, x_minibatch, D,
                                            mask_minibatch, xp)
            grad_D_pos_sum = accumurate_grad(grad_D_pos_sum, grad_D_pos)
            grad_D_neg_sum = accumurate_grad(grad_D_neg_sum, grad_D_neg)

        U = D * xp.maximum(grad_D_pos_sum, 0.0) / xp.maximum(grad_D_neg_sum,
                                                             _JITTER)
        D_new = normalize.l2_strict(U, axis=-1, xp=xp)
        if xp.max(xp.abs(D - D_new)) < tol:
            return it, D, x
        D = D_new

    return maxiter, D, x


# --- l2 loss ---
def _grad_x_l2(y, x, d, mask, xp):
    """ Multiplicative update rule for square loss.
    Returns Positive (numerator) and negative (denominator).
    mask is not used.
    """
    f = xp.dot(x, d)
    return xp.dot(y, d.T), xp.dot(f, d.T)


def _grad_d_l2(y, x, d, mask, xp):
    """ update d with l2 loss """
    f = xp.dot(x, d)
    return xp.dot(x.T, y), xp.dot(x.T, f)


def _grad_x_l2_mask(y, x, d, mask, xp):
    """ Multiplicative update rule for square loss.
    Returns Positive (numerator) and negative (denominator).
    mask is not used.
    """
    f = xp.dot(x, d) * mask
    y = y * mask
    return xp.dot(y, d.T), xp.dot(f, d.T)


def _grad_d_l2_mask(y, x, d, mask, xp):
    """ update d with l2 loss """
    f = xp.dot(x, d) * mask
    y = y * mask
    return xp.dot(x.T, y), xp.dot(x.T, f)


# --- KL loss ---
def _grad_x_kl(y, x, d, mask, xp):
    """ Multiplicative update rule for KL loss.    """
    f = xp.dot(x, d) + _JITTER
    return xp.dot(y / f, d.T), xp.sum(d.T, axis=0, keepdims=True)


def _grad_d_kl(y, x, d, mask, xp):
    """ update d with KL loss """
    f = xp.dot(x, d) + _JITTER
    return xp.dot(x.T, y / f), xp.sum(x.T, axis=1, keepdims=True)


def _grad_x_kl_mask(y, x, d, mask, xp):
    """ Multiplicative update rule for KL loss with mask. """
    f = xp.dot(x, d) + _JITTER
    y = y * mask
    return xp.dot(y / f, d.T), xp.dot(mask, d.T)


def _grad_d_kl_mask(y, x, d, mask, xp):
    """ update d with l2 loss """
    f = xp.dot(x, d) + _JITTER
    y = y * mask
    return xp.dot(x.T, y / f), xp.dot(x.T, mask)
