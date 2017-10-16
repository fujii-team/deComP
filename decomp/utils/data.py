"""
Compilation of utility functions for minibatching, memory transferer
"""


def minibatch_index(shape, minibatch, rng):
    """ Construct a minibatch index. """
    if minibatch is None and len(shape) == 2:
        return tuple([slice(None, None, None) for s in shape[:-1]])
    return tuple([rng.randint(0, s, minibatch) for s in shape[:-1]])
