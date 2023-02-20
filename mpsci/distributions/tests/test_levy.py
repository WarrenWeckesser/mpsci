
from mpmath import mp
from mpsci.distributions import levy


def test_pdf():
    with mp.workdps(50):
        x = 5
        p = levy.pdf(x)
        # Expected value computed with Wolfram Alpha:
        #    PDF[LevyDistribution[0, 1], 5]
        valstr = '0.032286845174307237010266347059999367773609621784713947'
        expected = mp.mpf(valstr)
        assert mp.almosteq(p, expected)


def test_cdf():
    with mp.workdps(50):
        x = 5
        p = levy.cdf(x)
        # Expected value computed with Wolfram Alpha:
        #    CDF[LevyDistribution[0, 1], 5]
        valstr = '0.6547208460185770294032359293626406196053124463243527949'
        expected = mp.mpf(valstr)
        assert mp.almosteq(p, expected)


def test_sf():
    with mp.workdps(50):
        x = 5
        p = levy.sf(x)
        # Expected value computed with Wolfram Alpha:
        #    SurvivalFunction[LevyDistribution[0, 1], 5]
        valstr = '0.34527915398142297059676407063735938039468755367564720509'
        expected = mp.mpf(valstr)
        assert mp.almosteq(p, expected)


def test_cdf_invcdf():
    with mp.workdps(50):
        x = mp.mpf(5)
        p = levy.cdf(x)
        y = levy.invcdf(p)
        assert mp.almosteq(y, x)


def test_sf_invsf():
    with mp.workdps(50):
        x = mp.mpf(5)
        p = levy.sf(x)
        y = levy.invsf(p)
        assert mp.almosteq(y, x)
