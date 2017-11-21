import numpy as np
import threading
from .cp_compat import numpy_or_cupy, get_array_module
from . import assertion

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


class MinibatchBase(object):
    def __init__(self, array, minibatch):
        """
        Load minibatch by slice.
        array should be already shuffled.

        array: array-like
            Array to be minibatched.
            The first dimension is considered as batch dimension.
        minibatch: int
            minibatch size
        """
        self.i = 0
        self.minibatch = minibatch
        self._array = array
        self.size = len(array)

    @property
    def shape(self):
        return self._array.shape

    @property
    def dtype(self):
        return self._array.dtype

    @property
    def array(self):
        return self._array

    @array.setter
    def array(self, arr):
        self._array = arr

    @property
    def n_loop(self):
        """ number of loops for one epoch """
        return int(self.size / self.minibatch)

    def reset(self):
        self.i = 0

    def __iter__(self):
        self.reset()
        return self

    def __next__(self):
        return self.next()

    def next(self):
        if self.i + self.minibatch > self.size:
            raise StopIteration()
        value = self._array[self.i: self.i + self.minibatch]
        self.i += self.minibatch
        return value


class MinibatchData(MinibatchBase):
    """ Data for minibatching. """
    def __init__(self, array, minibatch, shuffle_index=None):
        """
        array: array-like
            Array to be minibatched.
            The first dimension is considered as batch dimension.
        minibatch: int
            minibatch size
        shuffle_index: array-like
            Index array if shuffle is necessary.
        """
        xp = get_array_module(array)
        assertion.assert_shapes('array', array, 'shuffle_index', shuffle_index,
                                axes=[0])
        self.i = 0
        self.minibatch = minibatch
        self.size = len(array)
        if shuffle_index is not None:
            self._array = array[shuffle_index]
            self.restore_index = xp.arange(self.size)[shuffle_index]
        else:
            self._array = array
            self.restore_index = xp.arange(self.size)

    @property
    def array(self):
        index = self.restore_index.argsort()
        return self._array[index]

    def shuffle(self, shuffle_index):
        assertion.assert_shapes('array', self._array,
                                'shuffle_index', shuffle_index, axes=[0])
        self._array = self._array[shuffle_index]
        self.restore_index = self.restore_index[shuffle_index]


class SequentialMinibatchData(object):
    def __init__(self, array, minibatch, semibatch=100, shuffle_index=None,
                 needs_update=True):
        """
        Data with sequentiall memory transfer between cpu <-> gpu

        array: array-like
            Array to be minibatched.
            The first dimension is considered as batch dimension.
        minibatch: int
            minibatch size
        semibatch: int
            How many minibatches to be transferred to GPU memory.
            GPU memory should be larger than
            minibatch * semibatch * n_parallel
        shuffle_index: array-like
            Index array.
        needs_update: boolean
            True if array should be updated. (i.e. parameter)
        """
        # TODO This still assumes array is in_memory.
        # This should be improved by preserving shuffle_index
        assertion.assert_shapes('array', array, 'shuffle_index', shuffle_index,
                                axes=[0])
        self.size = len(array)
        if shuffle_index is not None:
            self._array = array[shuffle_index]
            self.restore_index = np.arange(self.size)[shuffle_index]
        else:
            self._array = array
            self.restore_index = np.arange(self.size)

        self.minibatch = minibatch
        self.semibatch = semibatch
        self.needs_update = needs_update

        self.round = 0
        self.n_round = int(len(array) / (self.minibatch * semibatch))

        indexes = slice(0, minibatch * semibatch)
        self.minibatch_data = MinibatchBase(
            numpy_or_cupy.array(self._array[indexes]), minibatch)

    @property
    def shape(self):
        return self._array.shape

    @property
    def dtype(self):
        return self._array.dtype

    @property
    def array(self):
        index = self.restore_index.argsort()
        return self._array[index]

    @property
    def n_loop(self):
        """ number of loops for one epoch """
        return self.n_round * self.minibatch_data.n_loop

    def shuffle(self, shuffle_index):
        assertion.assert_shapes('array', self._array,
                                'shuffle_index', shuffle_index, axes=[0])
        self._array = self._array[shuffle_index]
        self.restore_index = self.restore_index[shuffle_index]

    def reset(self):
        self.minibatch_data.reset()
        self.round = 0
        self.send_to_gpu(self.round)

    def __iter__(self):
        self.reset()
        return self

    def __next__(self):
        return self.next()

    def next(self):
        try:
            return self.minibatch_data.next()
        except StopIteration:
            # start writing data
            if self.needs_update:
                self.copy_from_gpu(self.round)

            self.round += 1
            if self.round >= self.n_round:
                raise StopIteration()

            # start reading data
            self.send_to_gpu(self.round)

            return self.minibatch_data.next()

    def copy_from_gpu(self, round_):
        mini_semibatch = self.minibatch * self.semibatch
        indexes = slice(round_ * mini_semibatch,
                        (round_ + 1) * mini_semibatch)
        if numpy_or_cupy is np:
            self._array[indexes] = self.minibatch_data.array
        else:  # cupy case
            self._array[indexes] = self.minibatch_data.array.get()

    def send_to_gpu(self, round_):
        mini_semibatch = self.minibatch * self.semibatch
        indexes = slice(round_ * mini_semibatch,
                        (round_ + 1) * mini_semibatch)
        self.minibatch_data.reset()
        self.minibatch_data.array = numpy_or_cupy.array(self._array[indexes])


class ParallelMinibatchData(SequentialMinibatchData):
    """
    Data with sequentiall memory transfer between cpu <-> gpu
    """
    def __init__(self, array, minibatch, semibatch=100,
                 n_parallel=4, shuffle_index=None, needs_update=True):
        """
        array: array-like
            Array to be minibatched.
            The first dimension is considered as batch dimension.
        minibatch: int
            minibatch size
        semibatch: int
            How many minibatches to be transferred to GPU memory.
            GPU memory should be larger than
            minibatch * semibatch * n_parallel
        n_parallel: int
            How many parallel job to be used for memory transfer.
        shuffle_index: array-like
            Index array.
        needs_update: boolean
            True if array should be updated. (i.e. parameter)
        """
        assertion.assert_shapes('array', array, 'shuffle_index', shuffle_index,
                                axes=[0])
        self.size = len(array)
        if shuffle_index is not None:
            self._array = array[shuffle_index]
            self.restore_index = np.arange(self.size)[shuffle_index]
        else:
            self._array = array
            self.restore_index = np.arange(self.size)

        self.minibatch = minibatch
        self.semibatch = semibatch
        self.needs_update = needs_update

        self.round = 0
        self.n_round = int(len(array) / (self.minibatch * semibatch))

        self.minibatch_data = []
        self.threads = []
        n_parallel = np.minimum(n_parallel, self.n_round)
        for i in range(n_parallel):
            indexes = slice(i * self.minibatch * semibatch,
                            (i + 1) * self.minibatch * semibatch)
            self.minibatch_data.append(
                MinibatchBase(numpy_or_cupy.array(self._array[indexes]),
                              minibatch))
            self.threads.append(None)
        self.n_parallel = n_parallel
        self.send_first_data()

    @property
    def n_loop(self):
        """ number of loops for one epoch """
        return self.n_round * self.minibatch_data[0].n_loop

    def shuffle(self, shuffle_index):
        super(ParallelMinibatchData, self).shuffle(shuffle_index)
        self.send_first_data()

    def send_first_data(self):
        for i in range(self.n_parallel):
            self.send_to_gpu(i)

    def reset(self):
        self.round = 0
        for i in range(self.n_parallel):
            self.minibatch_data[i].reset()
        self.send_first_data()

    def next(self):
        minibatch_index = self.round % self.n_parallel
        try:
            return self.minibatch_data[minibatch_index].next()
        except StopIteration:
            # start writing and sending data
            self.threads[minibatch_index] = threading.Thread(
                    target=self.copy_and_send, args=(self.round, ))
            self.threads[minibatch_index].start()

            self.round += 1
            if self.round >= self.n_round:
                self.round = 0
                # Wait until all the copy and send finish
                for th in self.threads:
                    if th is not None:
                        th.join()
                raise StopIteration()

            # wait until sending is finished
            minibatch_index = self.round % self.n_parallel
            if self.threads[minibatch_index] is not None:
                self.threads[minibatch_index].join()
            return self.minibatch_data[minibatch_index].next()

    def copy_and_send(self, round_):
        if self.needs_update:
            self.copy_from_gpu(round_)
        if round_ + self.n_parallel < self.n_round:
            self.send_to_gpu(round_ + self.n_parallel)

    def copy_from_gpu(self, round_):
        minibatch_index = round_ % self.n_parallel
        mini_semibatch = self.minibatch * self.semibatch
        indexes = slice(round_ * mini_semibatch,
                        (round_ + 1) * mini_semibatch)
        if numpy_or_cupy is np:
            self._array[indexes] = self.minibatch_data[minibatch_index].array
        else:  # cupy case
            self._array[indexes] = \
                            self.minibatch_data[minibatch_index].array.get()

    def send_to_gpu(self, round_):
        minibatch_index = round_ % self.n_parallel
        mini_semibatch = self.minibatch * self.semibatch
        indexes = slice(round_ * mini_semibatch,
                        (round_ + 1) * mini_semibatch)
        self.minibatch_data[minibatch_index].reset()
        self.minibatch_data[minibatch_index].array = numpy_or_cupy.array(
                                                        self._array[indexes])


class NoneIterator(object):
    """ Iterator just gives None """
    def __iter__(self):
        return self

    def __next__(self):
        return self.next()

    def next(self):
        """ Dummy method """
        return None

    def shuffle(self, index):
        """ Dummy method """
        pass
