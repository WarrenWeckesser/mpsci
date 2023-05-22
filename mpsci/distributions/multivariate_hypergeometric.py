"""
Multivariate hypergeometric distribution
----------------------------------------

The multivariate hypergeometric distribution is a generalization of the
hypergeometric distribution.
"""

from mpmath import mp
from ..fun import logbinomial

__all__ = ['support', 'pmf_dict', 'logpmf_dict', 'mean', 'cov', 'entropy']

_multivariate = True


def _validate_params(colors, nsample):
    if any(color < 0 for color in colors):
        raise ValueError("all values in colors must be nonnegative.")
    if nsample > sum(colors):
        raise ValueError("nsample must not exceed sum(colors).")


def support(colors, nsample):
    """
    Generator of the support of the multivariate hypergeometric distribution.

    Examples
    --------
    >>> list(multivariate_hypergeometric.support([4, 5, 6], 6))
    [[0, 0, 6],
     [0, 1, 5],
     [0, 2, 4],
     [0, 3, 3],
     [0, 4, 2],
     [0, 5, 1],
     [1, 0, 5],
     [1, 1, 4],
     [1, 2, 3],
     [1, 3, 2],
     [1, 4, 1],
     [1, 5, 0],
     [2, 0, 4],
     [2, 1, 3],
     [2, 2, 2],
     [2, 3, 1],
     [2, 4, 0],
     [3, 0, 3],
     [3, 1, 2],
     [3, 2, 1],
     [3, 3, 0],
     [4, 0, 2],
     [4, 1, 1],
     [4, 2, 0]]
    """
    _validate_params(colors, nsample)
    return _support_gen(colors, nsample)


def _support_gen(colors, nsample):
    if len(colors) == 1:
        yield [nsample]
    else:
        c0 = colors[0]
        c1 = sum(colors[1:])
        first = max(nsample - c1, 0)
        last = min(c0, nsample)
        for k in range(first, last + 1):
            yield from [[k] + t for t in _support_gen(colors[1:], nsample - k)]


def pmf_dict(colors, nsample):
    """
    Returns a dictionary of the PMF of the multivariate hypergeometric distr.

    The keys are the points in the support, and the values are the
    corresponding probabilities.

    Example
    -------
    >>> from mpmath import mp
    >>> mp.dps = 25
    >>> from mpsci.distributions import multivariate_hypergeometric

    >>> pmf = multivariate_hypergeometric.pmf_dict([4, 5, 6], 6)
    >>> len(pmf)
    24
    >>> pmf[2, 2, 2]
    mpf('0.17982017982017982017982018')
    """
    _validate_params(colors, nsample)
    total = sum(colors)
    denom = mp.binomial(total, nsample)
    pmf = {}
    for coords in support(colors, nsample):
        numer = 1
        for color, k in zip(colors, coords):
            numer *= mp.binomial(color, k)
        prob = numer/denom
        pmf[tuple(coords)] = prob
    return pmf


def logpmf_dict(colors, nsample):
    """
    Log of the PMF of the multivariate hypergeometric distribution.

    The values are returned as a dictionary.  The keys are the points in
    the support, and the values are the corresponding logarithms of the
    probabilities.

    Example
    -------
    >>> from mpmath import mp
    >>> mp.dps = 25
    >>> from mpsci.distributions import multivariate_hypergeometric

    >>> logpmf = multivariate_hypergeometric.logpmf_dict([4, 5, 6], 6)
    >>> len(pmf)
    24
    >>> logpmf[2, 2, 2]
    mpf('-1.71579792842501020899506948')
    """
    _validate_params(colors, nsample)
    total = sum(colors)
    logdenom = logbinomial(total, nsample)
    logpmf = {}
    for coords in support(colors, nsample):
        lognumer = 0
        for color, k in zip(colors, coords):
            lognumer += logbinomial(color, k)
        logprob = lognumer - logdenom
        logpmf[tuple(coords)] = logprob
    return logpmf


def mean(colors, nsample):
    """
    Mean of the multivariate hypergeometric distribution.

    Examples
    --------
    First configure the default mpmath precision.

    >>> from mpmath import mp
    >>> mp.dps = 25

    Some examples of the `mean` function.

    >>> from mpsci.distributions import multivariate_hypergeometric
    >>> multivariate_hypergeometric.mean([5, 10, 15], 18)
    [mpf('3.0'), mpf('6.0'), mpf('9.0')]

    >>> multivariate_hypergeometric.mean([1, 2, 4, 8, 16], 20)
    [mpf('0.6451612903225806451612903226'),
     mpf('1.290322580645161290322580645'),
     mpf('2.58064516129032258064516129'),
     mpf('5.161290322580645161290322581'),
     mpf('10.32258064516129032258064516')]
    """
    _validate_params(colors, nsample)
    with mp.extradps(5):
        s = mp.fsum(colors)
        if nsample == s:
            # This includes the edge case where colors = [0, ..., 0]
            # and nsample = 0.
            return [mp.one * k for k in colors]
        return [nsample * (k / s) for k in colors]


def cov(colors, nsample):
    """
    Covariance matrix of the multivariate hypergeometric distribution.

    Let n be the length of `colors`. The n x n covariance matrix is
    represented as a list of lists, where the length of the outer list
    and the lengths of the inner lists are all n.

    Examples
    --------
    First configure the default mpmath precision.  Also import `pprint`
    for nicer printed output.

    >>> from pprint import pprint
    >>> from mpmath import mp
    >>> mp.dps = 25

    Some examples of the `cov` function.

    >>> from mpsci.distributions import multivariate_hypergeometric
    >>> c = multivariate_hypergeometric.cov([5, 10, 15], 18)
    >>> pprint(c)
    [[mpf('1.034482758620689655172413793'),
      mpf('-0.4137931034482758620689655172'),
      mpf('-0.6206896551724137931034482759')],
     [mpf('-0.4137931034482758620689655172'),
      mpf('1.655172413793103448275862069'),
      mpf('-1.241379310344827586206896552')],
     [mpf('-0.6206896551724137931034482759'),
      mpf('-1.241379310344827586206896552'),
      mpf('1.862068965517241379310344828')]]
    >>> c = multivariate_hypergeometric.cov([1, 2, 4, 8], 6)
    >>> pprint(c)
    [[mpf('0.24'),
      mpf('-0.03428571428571428571428571429'),
      mpf('-0.06857142857142857142857142857'),
      mpf('-0.1371428571428571428571428571')],
     [mpf('-0.03428571428571428571428571429'),
      mpf('0.4457142857142857142857142857'),
      mpf('-0.1371428571428571428571428571'),
      mpf('-0.2742857142857142857142857143')],
     [mpf('-0.06857142857142857142857142857'),
      mpf('-0.1371428571428571428571428571'),
      mpf('0.7542857142857142857142857143'),
      mpf('-0.5485714285714285714285714286')],
     [mpf('-0.1371428571428571428571428571'),
      mpf('-0.2742857142857142857142857143'),
      mpf('-0.5485714285714285714285714286'),
      mpf('0.96')]]
    """
    _validate_params(colors, nsample)
    n = len(colors)
    with mp.extradps(5):
        s = mp.fsum(colors)
        if nsample == s:
            return [[mp.zero]*n for _ in range(n)]
        u = [k / s for k in colors]
        f = nsample * (s - nsample) / (s - 1)
        c = [[None]*n for _ in range(n)]
        for i in range(n):
            for j in range(n):
                if i == j:
                    c[i][j] = f * u[i] * (1 - u[i])
                else:
                    c[i][j] = -f * u[i] * u[j]
        return c


def entropy(colors, nsample):
    """
    Entropy of the multivariate hypergeometric distribution.

    The entropy is computed using the natural logarithm.

    Examples
    --------
    First configure the default mpmath precision.

    >>> from mpmath import mp
    >>> mp.dps = 25

    Some examples of the `entropy` function.

    >>> from mpsci.distributions import multivariate_hypergeometric
    >>> multivariate_hypergeometric.entropy([5, 10, 15], 18)
    mpf('3.046643631879764168841250566')

    >>> multivariate_hypergeometric.entropy([1, 2, 4, 8], 6)
    mpf('2.877874138861812367967354693')
    """
    _validate_params(colors, nsample)
    with mp.extradps(5):
        return -mp.fsum(p * mp.log(p)
                        for p in pmf_dict(colors, nsample).values())
