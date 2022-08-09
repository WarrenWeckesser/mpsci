
from itertools import product
import mpmath


def check_mle(dist, x, delta=None, **kwds):
    # This is a crude test of dist.mle().
    if delta is None:
        delta = mpmath.power(mpmath.mp.eps, 0.25)
    else:
        delta = mpmath.mpf(delta)
    p_hat = dist.mle(x, **kwds)
    nll0 = dist.nll(x, *p_hat, **kwds)
    n = len(p_hat)
    dirs = set(product(*([[-1, 0, 1]]*n))) - set([(0,)*n])
    for d in dirs:
        p = [param + s*delta for param, s in zip(p_hat, d)]
        nll1 = dist.nll(x, *p, **kwds)
        assert nll0 < nll1
