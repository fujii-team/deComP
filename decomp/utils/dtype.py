import numpy as np


def float_type(dtype):
    """ Convert to equivalent float type """
    if dtype.kind == 'f':
        return dtype
    if dtype is np.complex64:
        return np.float32
    if dtype is np.complex128:
        return np.float64

    try:
        import cupy as cp
        if dtype is cp.complex64:
            return cp.float32
        if dtype is cp.complex128:
            return cp.float64
    except ImportError:
        pass
