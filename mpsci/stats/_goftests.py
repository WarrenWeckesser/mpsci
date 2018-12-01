import mpmath
from ..fun import xlogy


__all__ = ['chisquare', 'gtest']


def chisquare(observed, expected):
    chi2 = sum((obs - exp)**2/exp for obs, exp in zip(observed, expected))
    df = len(observed) - 1
    p = mpmath.gammainc(df/2, chi2/2) / mpmath.gamma(df/2)
    return chi2, p


def gtest(observed, expected):
    stat = sum(2*xlogy(obs, obs/exp) for obs, exp in zip(observed, expected))
    df = len(observed) - 1
    p = mpmath.gammainc(df/2, stat/2) / mpmath.gamma(df/2)
    return stat, p
