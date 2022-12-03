
import pytest
from mpmath import mp
from mpsci.distributions import normal


mp.dps = 50


# Expected values were computed with Wolfram Alpha.


@pytest.mark.parametrize(
    'x, cdf',
    [(1.5, '0.74750746245307708693593817561058267476764391762263'),
     (-30, '4.0447964135863665408996212512977819124135460426002e-23')]
)
def test_cdf(x, cdf):
    mu = -0.5
    sigma = 3.0
    cdf = mp.mpf(cdf)

    p = normal.cdf(x, mu, sigma)
    assert mp.almosteq(p, cdf)

    x2 = normal.invcdf(cdf, mu, sigma)
    assert mp.almosteq(x2, x)


@pytest.mark.parametrize(
    'x, sf',
    [(1.5, '0.252492537546922913064061824389417325232356082377372'),
     (25,  '9.47953482220331835415105046784755149282645008676382e-18')]
)
def test_sf(x, sf):
    mu = -0.5
    sigma = 3.0
    sf = mp.mpf(sf)

    p = normal.sf(x, mu, sigma)
    assert mp.almosteq(p, sf)

    x2 = normal.invsf(sf, mu, sigma)
    assert mp.almosteq(x2, x)
