"""
Multivariate hypergeometric distribution
----------------------------------------
"""

import mpmath


__all__ = ['mvhg_support', 'pmf_dict']


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
    if any(color < 0 for color in colors):
        raise ValueError("all values in colors must be nonnegative.")
    if nsample > sum(colors):
        raise ValueError("nsample must not exceed sum(colors).")
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
