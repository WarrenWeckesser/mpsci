from mpmath import mp
from ..fun import xlogy


__all__ = ['chisquare', 'gtest']


def chisquare(observed, expected, sum_rel_tol=None):
    """
    Pearson's chi-square test.

    Test whether the observed frequency distribution differs from the
    expected frequency distribution.

    Returns the chi-square statistic and the p-value.
    """
    if sum_rel_tol is None:
        sum_rel_tol = 10*mp.eps
    observed = [mp.mpf(t) for t in observed]
    expected = [mp.mpf(t) for t in expected]
    if not mp.almosteq(sum(observed), sum(expected),
                       rel_eps=sum_rel_tol, abs_eps=0):
        raise ValueError('sum(observed) differs from sum(expected)')
    chi2 = sum((obs - exp)**2/exp for obs, exp in zip(observed, expected))
    df = len(observed) - 1
    p = mp.gammainc(df/2, chi2/2) / mp.gamma(df/2)
    return chi2, p


def gtest(observed, expected, sum_rel_tol=None):
    r"""
    G-test (likelihood-ratio test, maximum likelihood test).

    Test whether the observed frequency distribution differs from the
    expected frequency distribution.

    The G statistic is::

        G = 2 * sum(observed*log(observed/expected))

    or

    .. math::  G = 2\sum_{i=0}^{n-1} O_{i} \log(O_{i}/E_{i})

    Returns the G statistic and the p-value.
    """
    if sum_rel_tol is None:
        sum_rel_tol = 10*mp.eps
    observed = [mp.mpf(t) for t in observed]
    expected = [mp.mpf(t) for t in expected]
    if not mp.almosteq(sum(observed), sum(expected),
                       rel_eps=sum_rel_tol, abs_eps=0):
        raise ValueError('sum(observed) differs from sum(expected)')
    stat = sum(2*xlogy(obs, obs/exp) for obs, exp in zip(observed, expected))
    df = len(observed) - 1
    p = mp.gammainc(df/2, stat/2) / mp.gamma(df/2)
    return stat, p
