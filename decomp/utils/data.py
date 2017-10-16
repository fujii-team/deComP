"""
Compilation of utility functions for minibatching, memory transferer
"""


def minibatch_index(shape, minibatch, rng):
    """ Construct a minibatch index. """
    return tuple([rng.randint(0, s, minibatch) for s in shape[:-1]])
