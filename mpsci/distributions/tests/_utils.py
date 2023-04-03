
from itertools import product
from mpmath import mp


def check_mle(nll, x, p_hat, delta=None):
    """
    `nll` must be a callable with signature `nll(x, *p)`.
    """
    if delta is None:
        delta = mp.power(mp.eps, 0.25)
    else:
        delta = mp.mpf(delta)
    nll0 = nll(x, *p_hat)
    n = len(p_hat)
    dirs = set(product(*([[-1, 0, 1]]*n))) - set([(0,)*n])
    for d in dirs:
        p = [param + s*delta for param, s in zip(p_hat, d)]
        nll1 = nll(x, *p)
        assert nll0 < nll1, f'{nll0 = }  {nll1 =}'


def call_and_check_mle(mle, nll, x, delta=None):
    """
    `mle` must be a callable that accepts a single parameter, x, and returns
    a tuple of parameter values.

    `nll` must be a callable that accepts the parameters `x` and the result
    of `mle(x)` as if passed in as `*mle(x)`.
    """
    p_hat = mle(x)
    check_mle(nll, x, p_hat, delta=delta)
