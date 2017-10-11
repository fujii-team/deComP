# deComP

Python Library for Large Scale Matrix Decomposition with CuPy compatibility.

## What is deComP

deComP is a python library for large scale matrix decomposition,
designed to realize a maximum parallelism.

Our algorithms utilize numpy's parallelization capacity as much as possible.
Furthermore, deComP is also compatible to
[`CuPy`](https://github.com/cupy/cupy),
which gives `numpy`-like interface for gpu computing.


## Implemented models

Currently, we implemented

+ Lasso Regression
+ Dictionary Learning

All the models support complex values as well as real values.


## Requirements

The official requirements are only `numpy`.
However, in order to work on GPU, we recommend to install `cupy`.
