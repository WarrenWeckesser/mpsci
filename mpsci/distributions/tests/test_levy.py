from mpmath import mp
from mpsci.distributions import levy


@mp.workdps(50)
def test_pdf():
    x = 5
    p = levy.pdf(x)
    # Expected value computed with Wolfram Alpha:
    #    PDF[LevyDistribution[0, 1], 5]
    valstr = '0.032286845174307237010266347059999367773609621784713947'
    expected = mp.mpf(valstr)
    assert mp.almosteq(p, expected)


@mp.workdps(50)
def test_cdf():
    x = 5
    p = levy.cdf(x)
    # Expected value computed with Wolfram Alpha:
    #    CDF[LevyDistribution[0, 1], 5]
    valstr = '0.6547208460185770294032359293626406196053124463243527949'
    expected = mp.mpf(valstr)
    assert mp.almosteq(p, expected)


@mp.workdps(50)
def test_sf():
    x = 5
    p = levy.sf(x)
    # Expected value computed with Wolfram Alpha:
    #    SurvivalFunction[LevyDistribution[0, 1], 5]
    valstr = '0.34527915398142297059676407063735938039468755367564720509'
    expected = mp.mpf(valstr)
    assert mp.almosteq(p, expected)


@mp.workdps(50)
def test_cdf_invcdf():
    x = mp.mpf(5)
    p = levy.cdf(x)
    y = levy.invcdf(p)
    assert mp.almosteq(y, x)


@mp.workdps(50)
def test_sf_invsf():
    x = mp.mpf(5)
    p = levy.sf(x)
    y = levy.invsf(p)
    assert mp.almosteq(y, x)
