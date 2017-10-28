import numpy as np
import pytest
from decomp.utils.cp_compat import numpy_or_cupy as xp
from decomp import lasso
from decomp.utils.exceptions import ShapeMismatchError, DtypeMismatchError


class LassoData(object):
    def __init__(self):
        self.rng = xp.random.RandomState(0)
        # The design matrix is highly correlated
        self.A = self.randn(90, 100) + self.randn(100) * 0.3
        x_true = self.randn(99000) * xp.rint(self.rng.uniform(size=99000))
        x_true = x_true.reshape(1100, 90)
        self.y = xp.dot(x_true, self.A) + self.randn(1100, 100) * 0.1

        v = self.rng.uniform(0.45, 1.0, size=110000).reshape(1100, 100)
        self.mask = xp.rint(v)
        self.alphas = np.exp(np.linspace(np.log(0.1), np.log(10.0), 5))
        self.maxiter = 3000

        self.A = self.A.astype(np.float32)
        self.y = self.y.astype(np.float32)
        self.alphas = self.alphas.astype(np.float32)
        self.mask = self.mask.astype(np.float32)

    def randn(self, *shape):
        return self.rng.randn(*shape)

    def test(self, method):
        it_list = []
        for alpha in self.alphas:
            it, x = lasso.solve(self.y, self.A, alpha=alpha, tol=1.0e-4,
                                method=method, maxiter=self.maxiter)
            it_list.append(it)
        return it_list


def test_ista(benchmark):
    lasso_data = LassoData()
    it_list = benchmark(lasso_data.test, 'ista')
    assert all(it < lasso_data.maxiter - 1 for it in it_list)

def test_acc_ista(benchmark):
    lasso_data = LassoData()
    it_list = benchmark(lasso_data.test, 'acc_ista')
    assert all(it < lasso_data.maxiter - 1 for it in it_list)

def test_fista(benchmark):
    lasso_data = LassoData()
    it_list = benchmark(lasso_data.test, 'fista')
    assert all(it < lasso_data.maxiter - 1 for it in it_list)

def test_cd(benchmark):
    lasso_data = LassoData()
    it_list = benchmark(lasso_data.test, 'cd')
    assert all(it < lasso_data.maxiter - 1 for it in it_list)

def test_parallel_cd(benchmark):
    lasso_data = LassoData()
    it_list = benchmark(lasso_data.test, 'parallel_cd')
    assert all(it < lasso_data.maxiter - 1 for it in it_list)
