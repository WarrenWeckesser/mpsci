
from itertools import product


def check_mle(dist, x, delta=1e-9, **kwds):
    # This is a crude test of dist.mle().
    p_hat = dist.mle(x, **kwds)
    nll = dist.nll(x, *p_hat, **kwds)
    n = len(p_hat)
    dirs = set(product(*([[-1, 0, 1]]*n))) - set([(0,)*n])
    for d in dirs:
        p = [param + s*delta for param, s in zip(p_hat, d)]
        assert nll < dist.nll(x, *p, **kwds)
