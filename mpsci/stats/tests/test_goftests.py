import pytest
from mpmath import mp
from mpsci.stats import chisquare, gtest


def test_chisquare():
    # This is a simple example.  The expected values can be computed by hand
    # or with most any other statistical software (e.g. scipy)
    obs = [10, 20, 21, 17, 22]
    exp = [10, 20, 20, 20, 20]
    stat, p = chisquare(obs, exp)
    assert mp.almosteq(stat, 0.7)
    assert mp.almosteq(p, 0.95132892112, rel_eps=5e-11, abs_eps=0)


def test_chisquare_mismatch_totals():
    obs = [10, 20, 21, 17, 22]
    exp = [10, 20, 20, 20, 10]
    with pytest.raises(ValueError,
                       match=r'sum\(observed\) differs from sum\(expected\)'):
        chisquare(obs, exp)


def test_gtest():
    obs = [10, 20, 21, 17, 22]
    exp = [10, 20, 20, 20, 20]
    stat, p = gtest(obs, exp)
    # Reference values are from scipy.stats.power_divergence with lambda_=0.
    assert mp.almosteq(stat, 0.717191203582, rel_eps=5e-12, abs_eps=0)
    assert mp.almosteq(p, 0.94919209612956, rel_eps=5e-12, abs_eps=0)


def test_gtest_mismatch_totals():
    obs = [10, 20, 21, 17, 22]
    exp = [10, 20, 20, 20, 10]
    with pytest.raises(ValueError,
                       match=r'sum\(observed\) differs from sum\(expected\)'):
        gtest(obs, exp)
