"""
Compilation of utility functions for minibatching, memory transferer
"""


def minibatch_index(shape, minibatch, rng):
    """ Construct a minibatch index from random indexing. """
    if minibatch is None and len(shape) == 1:
        return tuple([slice(None, None, None) for s in shape])
    return tuple([rng.randint(0, s, minibatch) for s in shape])


class MinibatchEpochIndex(object):
    """ A simple class to provide a minibatch index.
    usage:
        minibatch_index = MinibatchEpochIndex(size, minibatch, rng, xp)
        # run 1 epoch
        for index in minibatch_index:
            x_minibatch = x[index]
            # do something with the minibatch
    """
    def __init__(self, size, minibatch, rng, xp):
        """
        size: int
            shape of the indexes.
        minibatch: int
            number of minibatch
        rng: np.random.RandomState or cp.random.RandomState
        """
        self._i = 0
        self._minibatch = minibatch
        self._indexes = xp.arange(size)
        self.rng = rng
        self.shuffle()

    def shuffle(self):
        self.rng.shuffle(self._indexes)

    def __len__(self):
        return int(len(self._indexes) / self._minibatch)

    def __iter__(self):
        return self

    def __next__(self):
        return self.next()

    def next(self):
        if self._i + self._minibatch > len(self._indexes):
            self._i = 0
            raise StopIteration()
        value = self._indexes[self._i: self._i + self._minibatch]
        self._i += self._minibatch
        return value
