import numpy as np
from .exceptions import ShapeMismatchError, DtypeMismatchError

def assert_shapes(name_x='x', name_y='y', axes=None, **arrays):
    """
    Make sure the shapes of x and y are consistent.
    If axes==None, assert x.shape == y.shape
    If axes is integer such as 1,
        assert x.shape[-axes:] == y.shape[:axes]
    If axes is a list of integers, it makes sure
        x.shape[axes] == y.shape[axes]

    Parameters
    ----------
    **arrays: Mapping from name to arrays.
    """
    assert len(arrays.keys()) == 2

    (x_name, x), (y_name, y) = arrays.items()
    if x is None or y is None:
        return

    if axes is None:
        if x.shape != y.shape:
            raise ShapeMismatchError(
                'Shapes of x and y should be identical. '
                'Given {0:s}: {1:s} and {2:s}: {3:s}'.format(
                                x_name, str(x.shape), y_name, str(y.shape)))
        return

    if isinstance(axes, int):
        if x.shape[-axes:] != y.shape[:axes]:
            raise ShapeMismatchError(
                '{0:s}.shape[-{2:d}:] == {1:s}.shape[:{2:d}] '
                'should be satisfied. Given: {0:s}.shape = {3:s} '
                'and {1:s}.shape = {4:s}'.format(x_name, y_name, axes,
                                                   str(x.shape), str(y.shape)))
        return

    if isinstance(axes, (list, tuple)):
        if any(np.array(x.shape)[axes] != np.array(y.shape)[axes]):
            raise ShapeMismatchError(
                '{0:s}.shape[{1:s}] == {2:s}.shape[{1:s}] '
                'should be satisfied. Given: {0:s}.shape = {3:s} and '
                '{2:s}.shape = {4:s}'.format(x_name, str(axes), y_name,
                                             str(x.shape), str(y.shape)))
        return
    raise TypeError('Argument axes is invalid, given ' + str(axes))


def assert_dtypes(dtypes='fc', **arrays):
    """
    Make sure the dtypes of two arrays are valid.
    The valid dtypes are given by dtypes arguments.

    Parameters
    ----------
    dtypes: string
        Allowed dtype kinds. 'f' for float, 'c' for complex.
    arrays:
        Arrays dtype of which to be compared.
    """
    x1 = None
    for k, x in arrays.items():
        if x is not None:
            k1 = k
            x1 = x
            break
    if x1 is None:  # no arrays are provided other than None
        return

    for k, x in arrays.items():
        if x is not None and x.dtype != x1.dtype:
            raise DtypeMismatchError(
                'Data type should be all identical, {0:s}: {1:s} '
                'and {2:s}: {3:s}'.format(k1, str(x1.dtype), k, str(x.dtype)))

        if x is not None and x.dtype.kind not in dtypes:
            raise DtypeMismatchError(
                'Data type should be one of ' + str(dtypes) +
                ', but given {0:s} for {1:s}'.format(str(x.dtype), k))
