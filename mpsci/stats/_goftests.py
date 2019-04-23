import mpmath
from ..fun import xlogy


__all__ = ['chisquare', 'gtest']


def chisquare(observed, expected):
    """
    Pearson's chi-square test.

    Test whether the observed frequency distribution differs from the
    expected frequency distribution.
    """
    chi2 = sum((obs - exp)**2/exp for obs, exp in zip(observed, expected))
    df = len(observed) - 1
    p = mpmath.gammainc(df/2, chi2/2) / mpmath.gamma(df/2)
    return chi2, p


def gtest(observed, expected):
    """
    G-test (likelihood-ratio test, maximum likelihood test).

    Test whether the observed frequency distribution differs from the
    expected frequency distribution.

    The G statistic is::

        G = 2 * sum(observed*log(observed/expected))

    """
    stat = sum(2*xlogy(obs, obs/exp) for obs, exp in zip(observed, expected))
    df = len(observed) - 1
    p = mpmath.gammainc(df/2, stat/2) / mpmath.gamma(df/2)
    return stat, p
