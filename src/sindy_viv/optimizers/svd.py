"""Unscaled OLS, genuine Ridge, and sequential support refits without unbiasing."""
import warnings
import numpy as np
from scipy import linalg
from sklearn.linear_model import Ridge
from pysindy.optimizers import BaseOptimizer


class _UnscaledOptimizer(BaseOptimizer):
    def __init__(self, max_iter=100, normalize_columns=False, unbias=False):
        if normalize_columns or unbias:
            raise ValueError('This convention requires normalize_columns=False and unbias=False.')
        super().__init__(max_iter=max_iter, normalize_columns=False, unbias=False)

    def fit(self, x_, y, sample_weight=None, **kwargs):
        # BaseOptimizer otherwise performs an unnecessary initial NumPy solve.
        target=np.asarray(y)
        self.initial_guess=np.zeros((1 if target.ndim==1 else target.shape[1],np.shape(x_)[1]))
        try:return super().fit(x_,y,sample_weight=sample_weight,**kwargs)
        finally:self.initial_guess=None

    def _ols(self, x, y):
        self.rank_cutoff_=max(x.shape)*np.finfo(float).eps
        coef,_,rank,singular=linalg.lstsq(x,y,cond=self.rank_cutoff_,lapack_driver='gelsd')
        self.rank_,self.singular_=int(rank),singular
        return coef


class SVDLeastSquares(_UnscaledOptimizer):
    """True no-penalty GELSD OLS, relative cutoff=max(N,P)*float64 eps."""
    def __init__(self, normalize_columns=False, unbias=False):
        super().__init__(normalize_columns=normalize_columns,unbias=unbias)

    def _reduce(self, x, y):
        self.coef_=self._ols(x,y).T
        self.history_=[self.coef_.copy()]


class RidgeOnly(_UnscaledOptimizer):
    """Actual sklearn Ridge(solver='svd'), with no threshold or final OLS refit."""
    def __init__(self, alpha=1e-3, normalize_columns=False, unbias=False):
        if not np.isfinite(alpha) or alpha<=0:raise ValueError('Ridge-only alpha must be positive.')
        self.alpha=alpha
        super().__init__(normalize_columns=normalize_columns,unbias=unbias)

    def _reduce(self, x, y):
        self.estimator_=Ridge(alpha=self.alpha,solver='svd',fit_intercept=False).fit(x,y)
        self.coef_=self.estimator_.coef_.reshape(y.shape[1],x.shape[1])
        self.history_=[self.coef_.copy()]


class SequentialSVD(_UnscaledOptimizer):
    """Shrink supports, refitting with GELSD OLS for alpha=0 or SVD Ridge otherwise.

    Thresholds act on original polynomial coefficients. Positive Ridge uses
    ||Y-XW||_F**2 + alpha*||W||_F**2, including the polynomial bias coefficient.
    Supports do not regrow. No normal equations, zero-alpha Ridge, or unbiasing.
    """
    def __init__(self, alpha=0., threshold=.1, max_iter=100, normalize_columns=False, unbias=False):
        if not np.isfinite(alpha) or alpha<0 or not np.isfinite(threshold) or threshold<0:
            raise ValueError('Use finite nonnegative alpha and threshold.')
        self.alpha,self.threshold=alpha,threshold
        super().__init__(max_iter=max_iter,normalize_columns=normalize_columns,unbias=unbias)

    def _solve(self, x, y):
        if self.alpha==0:return self._ols(x,y).T
        return Ridge(alpha=self.alpha,solver='svd',fit_intercept=False).fit(x,y).coef_

    def _reduce(self, x, y):
        coef=self._solve(x,y).reshape(y.shape[1],x.shape[1])
        support=np.ones_like(coef,dtype=bool);self.history_=[coef.copy()]
        self.converged_=False
        for iteration in range(self.max_iter):
            active=support & (np.abs(coef)>=self.threshold)
            if np.array_equal(active,support):
                self.converged_=True;break
            coef.fill(0.)
            for target,mask in enumerate(active):
                if np.any(mask):coef[target,mask]=self._solve(x[:,mask],y[:,target])
            support=active;self.history_.append(coef.copy())
        self.iters=iteration+1
        if not self.converged_:
            coef[np.abs(coef)<self.threshold]=0.
            warnings.warn('SequentialSVD reached max_iter; remaining small coefficients zeroed.',RuntimeWarning)
        self.coef_=coef
