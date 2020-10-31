"""
Multivariate hypergeometric distribution
----------------------------------------
"""

import mpmath


__all__ = ['support', 'pmf_dict', 'mean', 'cov', 'entropy']

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
    >>> pmf = multivariate_hypergeometric.pmf_dict([4, 5, 6], 6)
    >>> len(pmf)
    24
    >>> float(pmf[2, 2, 2])
    0.1798201798201798
    """
    _validate_params(colors, nsample)
    total = sum(colors)
    denom = mpmath.binomial(total, nsample)
    pmf = {}
    for coords in support(colors, nsample):
        numer = 1
        for color, k in zip(colors, coords):
            numer *= mpmath.binomial(color, k)
        prob = numer/denom
        pmf[tuple(coords)] = prob
    return pmf


def mean(colors, nsample):
    """
    Mean of the multivariate hypergeometric distribution.
    """
    _validate_params(colors, nsample)
    with mpmath.extradps(5):
        s = mpmath.fsum(colors)
        if nsample == s:
            # This includes the edge case where colors = [0, ..., 0]
            # and nsample = 0.
            return [mpmath.mp.one * k for k in colors]
        return [nsample * (k / s) for k in colors]


def cov(colors, nsample):
    """
    Covariance matrix of the multivariate hypergeometric distribution.

    Let n be the length of `colors`. The n x n covariance matrix is
    represented as a list of lists, where the length of the outer list
    and the lengths of the inner lists are all n.
    """
    _validate_params(colors, nsample)
    n = len(colors)
    with mpmath.extradps(5):
        s = mpmath.fsum(colors)
        if nsample == s:
            return [[mpmath.mp.zero]*n for _ in range(n)]
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
    """
    _validate_params(colors, nsample)
    with mpmath.extradps(5):
        return -mpmath.fsum(p * mpmath.log(p)
                            for p in pmf_dict(colors, nsample).values())
