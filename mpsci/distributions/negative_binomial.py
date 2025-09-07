"""
Negative binomial distribution
------------------------------

There are several different ways to parameterize the negative binomial
distribution.  Here, the quantiles are the number of "successes" that
occur when draws from a binomial distribution are made repeatedly until
the number of "failures" drawn is `r`.  `p` is the probability of drawing
a "success".
"""

import itertools
from mpmath import mp
from ..fun import logbinomial, xlogy, xlog1py
from ..stats import mean as _mean
from ._common import Initial, _validate_counts, _validate_int


__all__ = ['support', 'pmf', 'logpmf', 'sf', 'cdf', 'mean', 'var',
           'nll', 'mle']


def _validate_params(r, p, allow_noninteger_r=True):
    if r <= 0:
        raise ValueError('r must be greater than 0')
    r = mp.mpf(r)
    if not allow_noninteger_r and int(r) != r:
        raise ValueError('r must be an integer')
    if not (0 <= p <= 1):
        raise ValueError('p must be in the interval [0, 1]')
    p = mp.mpf(p)
    return r, p


def support(r, p):
    """
    Support of the negative binomial distribution.

    The support is the integers 0, 1, 2, 3, ..., so the support is returned
    as an instance of `itertools.count(start=0)`.

    Examples
    --------
    >>> from mpsci.distributions import negative_binomial
    >>> sup = negative_binomial.support()
    >>> next(sup)
    0
    >>> next(sup)
    1

    """
    return itertools.count(start=0)


def _validate_x(x):
    y = []
    for xi in x:
        if xi < 0:
            raise ValueError('All values in x must be nonnegative')
        ximp = mp.mpmathify(xi)
        if ximp != int(ximp):
            raise ValueError('All values in x must be integers')
        y.append(ximp)
    return y


def logpmf(k, r, p):
    """
    Log of the probability mass function of the negative binomial distribution.

    Parameters
    ----------
    r : int
        Number of failures until the experiment is stopped.
    p : float
        Probability of success.
    """
    with mp.extradps(5):
        r, p = _validate_params(r, p)
        k = _validate_int(k)
        if k < 0:
            return mp.ninf
        return logbinomial(k + r - 1, k) + xlog1py(r, -p) + xlogy(k, p)


def pmf(k, r, p):
    """
    Probability mass function of the negative binomial distribution.

    Parameters
    ----------
    r : int
        Number of failures until the experiment is stopped.
    p : float
        Probability of success.
    """
    return mp.exp(logpmf(k, r, p))


def sf(k, r, p):
    """
    Survival function of the negative binomial distribution.

    Parameters
    ----------
    r : int
        Number of failures until the experiment is stopped.
    p : float
        Probability of success.
    """
    with mp.extradps(5):
        r, p = _validate_params(r, p)
        k = _validate_int(k)
        if k < 0:
            return mp.one
        return mp.betainc(k + 1, r, 0, p, regularized=True)


def cdf(k, r, p):
    """
    Cumulative distribution function of the negative binomial distribution.

    Parameters
    ----------
    r : int
        Number of failures until the experiment is stopped.
    p : float
        Probability of success.
    """
    with mp.extradps(5):
        r, p = _validate_params(r, p)
        k = _validate_int(k)
        if k < 0:
            return mp.zero
        return mp.betainc(k + 1, r, p, 1, regularized=True)


def mean(r, p):
    """
    Mean of the negative binomial distribution.

    Parameters
    ----------
    r : int
        Number of failures until the experiment is stopped.
    p : float
        Probability of success.
    """
    with mp.extradps(5):
        r, p = _validate_params(r, p)
        return p*r / (1 - p)


def var(r, p):
    """
    Variance of the negative binomial distribution.

    Parameters
    ----------
    r : int
        Number of failures until the experiment is stopped.
    p : float
        Probability of success.
    """
    with mp.extradps(5):
        r, p = _validate_params(r, p)
        return p*r / (1 - p)**2


def nll(x, r, p, *, counts=None):
    """
    Negative log-likelihood of sample for the negative binomial distribution.

    Parameters
    ----------
    x : sequence of nonnegative integers
        The sample for which the negative log-likelihood is to be calculated.
    r : int
        Number of failures until the experiment is stopped.
    p : float
        Probability of success.
    counts : sequence of integers, optional
        If given, this sequence must be the same length as `x`.
        It gives the number of occurrences in the sample of the corresponding
        value in `x`.
    """
    with mp.extradps(5):
        r, p = _validate_params(r, p)
        x = _validate_x(x)
        counts = _validate_counts(x, counts, expand_none=True)
        return -mp.fsum([count*logpmf(xi, r, p)
                         for xi, count in zip(x, counts)])


def mle(x, *, counts=None, r=None, p=None, allow_noninteger_r=True):
    """
    Maximum likelihood estimation for the negative binomial distribution.

    x  must be a sequence of nonnegative integers.
    """
    with mp.extradps(5):
        x = _validate_x(x)
        counts = _validate_counts(x, counts, expand_none=False)
        r_fixed = not (r is None or isinstance(r, Initial))
        p_fixed = not (p is None or isinstance(p, Initial))
        if r_fixed:
            if p_fixed:
                # Other than validation, there is nothing to do.
                r, p = _validate_params(r, p, allow_noninteger_r)
                return r, p
            else:
                # XXX p=Initial(...) is ignored.  Should Initial be disallowed
                # for p?
                r, _ = _validate_params(r, 0.5, allow_noninteger_r)
                m = _mean(x, weights=counts)
                phat = m/(r + m)
                return r, phat
        # r is not fixed.
        if not p_fixed:
            m = _mean(x, weights=counts)

            def mle_r_eqn(r):
                p1 = m/(r + m)
                return (_mean([mp.digamma(xi + r) for xi in x], weights=counts)
                        - mp.digamma(r) + mp.log1p(-p1))

            r0 = 1 if r is None else r.initial
            rhat = mp.findroot(mle_r_eqn, r0)
            phat = m/(rhat + m)
            if allow_noninteger_r or rhat == int(rhat):
                return rhat, phat
            # Integer r is required.
            rhat0 = mp.floor(rhat)
            rhat1 = mp.ceil(rhat)
            phat0 = m/(rhat0 + m)
            phat1 = m/(rhat1 + m)
            nll0 = nll(x, r=rhat0, p=phat0, counts=counts)
            nll1 = nll(x, r=rhat1, p=phat1, counts=counts)
            if nll0 <= nll1:
                return rhat0, phat0
            else:
                return rhat1, phat1
        else:
            # r is free, p is fixed.
            _, p = _validate_params(1, p)
            m = _mean(x, weights=counts)

            def mle_r_eqn(r):
                return (_mean([mp.digamma(xi + r) for xi in x], weights=counts)
                        - mp.digamma(r) + mp.log1p(-p))

            r0 = 1 if r is None else r.initial
            rhat = mp.findroot(mle_r_eqn, r0)
            if allow_noninteger_r or rhat == int(rhat):
                return rhat, p
            # Integer r is required.
            rhat0 = mp.floor(rhat)
            rhat1 = mp.ceil(rhat)
            nll0 = nll(x, r=rhat0, p=p, counts=counts)
            nll1 = nll(x, r=rhat1, p=p, counts=counts)
            if nll0 <= nll1:
                return rhat0, p
            else:
                return rhat1, p
