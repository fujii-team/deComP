import warnings

try:
    import cupy
    numpy_or_cupy = cupy

    def get_array_module(*arrays):
        xp = cupy.get_array_module(arrays[0])
        if any(x is not None and cupy.get_array_module(x)
               is not xp for x in arrays):
            raise TypeError("All the data types should be the same.")

        warnings.warn('Using cupy as backend.', stacklevel=1)
        return xp

except ImportError:
    import numpy
    numpy_or_cupy = numpy

    def get_array_module(*arrays):
        warnings.warn('Using numpy as backend.', stacklevel=1)
        return numpy
