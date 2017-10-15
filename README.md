# deComP
[![Travis Status for fujii-team/deComP](https://travis-ci.org/fujii-team/deComP.svg?branch=master)](https://travis-ci.org/fujii-team/deComP)


Python Library for Large Scale Matrix Decomposition with CuPy compatibility.

## What is deComP

deComP is a compilation of matrix decomposition algorithms,
especially for large scale matrices.

We compiled (and updated) several algorithms that utilizes numpy's
parallelization capacity as much as possible.

Furthermore, deComP is also compatible to
[`CuPy`](https://github.com/cupy/cupy),
which gives `numpy`-like interface for gpu computing.


## Implemented models

Currently, we implemented

+ Lasso Regression
+ Dictionary Learning

All the models support complex values as well as real values.
It also supports missing values.


## Requirements

The official requirements are only `numpy`.
However, in order to work on GPU, we recommend to install `cupy`.
