"""Explicit SVD regression backends for reproducible individual-model studies."""
from .svd import SVDLeastSquares, RidgeOnly, SequentialSVD

__all__ = ['SVDLeastSquares','RidgeOnly','SequentialSVD']
