from chainer import cuda


try:
    import cupy
    numpy_or_cupy = cupy
    has_cupy = True

    def get_array_module(*arrays):
        xp = cupy.get_array_module(arrays[0])
        if any(x is not None and cupy.get_array_module(x)
               is not xp for x in arrays):
            raise TypeError("All the data types should be the same.")

        return xp

    def linalg_inv(x):
        """ Batch version of np.linalg.inv
        """
        if x.ndim == 2:
            return cupy.linalg.inv(x)
        elif x.ndim == 3:
            inv = cupy.ndarray(x.shape, x.dtype)
            for i in len(inv):
                inv[i] = cupy.linalg.inv(x[i])
            return inv
        elif x.ndim == 4:
            inv = cupy.ndarray(x.shape, x.dtype)
            for i in range(inv.shape[0]):
                for j in range(inv.shape[1]):
                    inv[i, j] = cupy.linalg.inv(x[i, j])
            return inv
        elif x.ndim == 5:
            inv = cupy.ndarray(x.shape, x.dtype)
            for i in range(inv.shape[0]):
                for j in range(inv.shape[1]):
                    for k in range(inv.shape[2]):
                        inv[i, j, k] = cupy.linalg.inv(x[i, j, k])
            return inv
        else:
            raise ValueError('linalg_inv only support x.ndim == 5. Given',
                             x.ndim)

    def to_gpu(array, device=None):
        device = device if device is not None else cuda.get_device_from_id(0)
        return cuda.to_gpu(array, device=device)

    def to_cpu(array):
        return cuda.to_cpu(array)


except ImportError:
    import numpy
    numpy_or_cupy = numpy
    has_cupy = False

    def get_array_module(*arrays):
        return numpy

    linalg_inv = numpy.linalg.inv

    def to_gpu(array, device=None):
        return numpy.array(array)

    def to_cpu(array):
        return numpy.array(array)
