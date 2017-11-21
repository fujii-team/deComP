import unittest
import numpy as np
import time
from decomp.utils.cp_compat import numpy_or_cupy as xp
from decomp.utils.data import (MinibatchBase, MinibatchData,
                               SequentialMinibatchData, ParallelMinibatchData)

from .testings import allclose


class Test_MinibatchBase(unittest.TestCase):
    def setUp(self):
        self.array = xp.random.randn(100, 20)
        self.minibatch = 11
        self.data = MinibatchBase(self.array.copy(), self.minibatch)
        self.data2 = MinibatchBase(self.array.copy(), self.minibatch)

    def test_read(self):
        for i, (arr, arr2) in enumerate(zip(self.data, self.data2)):
            # wait to mimic heavy calculation
            time.sleep(0.01)
            assert allclose(
                self.array[i * self.minibatch: (i + 1) * self.minibatch], arr)
            assert allclose(
                self.array[i * self.minibatch: (i + 1) * self.minibatch], arr2)
        assert allclose(self.data.array, self.array)

        # make sure the loop can be repeated.
        for i, arr in enumerate(self.data):
            # wait to mimic heavy calculation
            time.sleep(0.01)
            assert allclose(
                self.array[i * self.minibatch: (i + 1) * self.minibatch], arr)
        assert allclose(self.data.array, self.array)

    def one_loop(self):
        count = 0
        for i, arr in enumerate(self.data):
            count += 1
            assert allclose(
                self.array[i * self.minibatch: (i + 1) * self.minibatch], arr)
        assert count == self.data.n_loop

    def two_loop(self):
        count = 0
        for i, (arr, arr2) in enumerate(zip(self.data, self.data2)):
            assert allclose(
                self.array[i * self.minibatch: (i + 1) * self.minibatch], arr)
            assert allclose(
                self.array[i * self.minibatch: (i + 1) * self.minibatch], arr2)
            count += 1
        assert count == self.data.n_loop

    def test_loop(self):
        for _ in range(10):
            self.one_loop()
        for _ in range(10):
            self.two_loop()

    def test_write(self):
        # make sure writing to array is possible
        for i, arr in enumerate(self.data):
            # wait to mimic heavy calculation
            time.sleep(0.01)
            arr[:] = xp.ones_like(arr) * i
        assert not allclose(self.data.array, self.array)

        for i, arr in enumerate(self.data):
            assert allclose(arr, xp.ones_like(arr) * i)


class Test_MinibatchData(Test_MinibatchBase):
    def setUp(self):
        self.array = xp.random.randn(100, 20)
        self.minibatch = 11
        self.data = MinibatchData(self.array.copy(), self.minibatch)
        self.data2 = MinibatchData(self.array.copy(), self.minibatch)


class Test_MinibatchData_with_shuffle(unittest.TestCase):
    def setUp(self):
        self.rng = xp.random.RandomState(0)
        self.array = xp.random.randn(100, 20)
        self.minibatch = 11
        self.shuffle_index = xp.arange(100)
        self.rng.shuffle(self.shuffle_index)
        self.data = MinibatchData(self.array.copy(), self.minibatch,
                                  self.shuffle_index)
        self.data2 = MinibatchData(self.array.copy(), self.minibatch,
                                   self.shuffle_index)

    def test_read(self):
        for i, (arr, arr2) in enumerate(zip(self.data, self.data2)):
            # wait to mimic heavy calculation
            time.sleep(0.01)
            array = self.array[self.shuffle_index]
            assert allclose(
                array[i * self.minibatch: (i + 1) * self.minibatch], arr)
        assert allclose(self.data.array, self.array)

    def one_loop(self):
        count = 0
        for i, arr in enumerate(self.data):
            count += 1
            array = self.array[self.shuffle_index]
            assert allclose(
                    array[i * self.minibatch: (i + 1) * self.minibatch], arr)
        assert count == self.data.n_loop

    def two_loop(self):
        count = 0
        for i, (arr, arr2) in enumerate(zip(self.data, self.data2)):
            count += 1
            array = self.array[self.shuffle_index]
            assert allclose(
                    array[i * self.minibatch: (i + 1) * self.minibatch], arr)
        assert count == self.data.n_loop

    def test_loop(self):
        for _ in range(10):
            self.one_loop()
        for _ in range(10):
            self.two_loop()

    def test_write(self):
        # make sure writing to array is possible
        for i, arr in enumerate(self.data):
            # wait to mimic heavy calculation
            time.sleep(0.01)
            arr[...] = xp.ones_like(arr) * i
        assert not allclose(self.data.array, self.array)

        for i, arr in enumerate(self.data):
            assert allclose(arr, xp.ones_like(arr) * i)

    def test_shuffle(self):
        assert not allclose(self.data._array, self.array)
        assert allclose(self.data.array, self.array)

        shuffle_index = xp.arange(len(self.shuffle_index))
        self.rng.shuffle(shuffle_index)
        self.data.shuffle(shuffle_index)
        assert not allclose(self.data._array, self.array)
        assert allclose(self.data.array, self.array)


class Test_SequentialMinibatchData(Test_MinibatchBase):
    def setUp(self):
        self.array = np.random.randn(1000, 20)
        self.minibatch = 12
        self.data = SequentialMinibatchData(self.array.copy(), self.minibatch,
                                            semibatch=8)
        self.data2 = SequentialMinibatchData(self.array.copy(), self.minibatch,
                                             semibatch=8)


class Test_SequentialMinibatchData_w_shuffle(Test_MinibatchData_with_shuffle):
    def setUp(self):
        self.rng = np.random.RandomState(0)
        self.array = np.random.randn(1000, 20)
        self.minibatch = 12
        self.shuffle_index = np.arange(1000)
        self.rng.shuffle(self.shuffle_index)
        self.data = SequentialMinibatchData(self.array.copy(), self.minibatch,
                                            semibatch=8,
                                            shuffle_index=self.shuffle_index)
        self.data2 = SequentialMinibatchData(self.array.copy(), self.minibatch,
                                             semibatch=8,
                                             shuffle_index=self.shuffle_index)

    def test_shuffle(self):
        assert not allclose(self.data._array, self.array)
        assert allclose(self.data.array, self.array)

        shuffle_index = np.arange(len(self.shuffle_index))
        self.rng.shuffle(shuffle_index)
        self.data.shuffle(shuffle_index)
        assert not allclose(self.data._array, self.array)
        assert allclose(self.data.array, self.array)


class Test_ParallelMinibatchData(Test_MinibatchBase):
    def setUp(self):
        self.array = np.random.randn(1000, 20)
        self.minibatch = 12
        self.data = ParallelMinibatchData(self.array.copy(), self.minibatch,
                                          semibatch=8)
        self.data2 = ParallelMinibatchData(self.array.copy(), self.minibatch,
                                           semibatch=8)


class Test_ParallelMinibatchData_2para(Test_MinibatchBase):
    def setUp(self):
        self.array = np.random.randn(1000, 20)
        self.minibatch = 12
        self.data = ParallelMinibatchData(self.array.copy(), self.minibatch,
                                          semibatch=8, n_parallel=2)
        self.data2 = ParallelMinibatchData(self.array.copy(), self.minibatch,
                                           semibatch=8, n_parallel=2)


class Test_ParallelMinibatchData_w_shuffle(
                                    Test_SequentialMinibatchData_w_shuffle):
    def setUp(self):
        self.rng = np.random.RandomState(0)
        self.array = np.random.randn(1000, 20)
        self.minibatch = 12
        self.shuffle_index = np.arange(1000)
        self.rng.shuffle(self.shuffle_index)
        self.data = ParallelMinibatchData(self.array.copy(), self.minibatch,
                                          semibatch=8,
                                          shuffle_index=self.shuffle_index)
        self.data2 = ParallelMinibatchData(self.array.copy(), self.minibatch,
                                           semibatch=8,
                                           shuffle_index=self.shuffle_index)


if __name__ == '__main__':
    unittest.main()
