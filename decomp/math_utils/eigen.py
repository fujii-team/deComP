""" Collection of eigen value estimator """


def spectral_radius_svd(X, xp):
    """ Largest magnitude of eigen values by SVD """
    return xp.linalg.svd(X)[1][0]


def spectral_radius_Gershgorin(X, xp):
    """
    An upper bound for the largest eigen velue by
    Gershgorin circle theorem.
    https://en.wikipedia.org/wiki/Gershgorin_circle_theorem

    Here, we assume X is symmetric.
    """
    return xp.max(xp.sum(xp.abs(X), axis=0))
