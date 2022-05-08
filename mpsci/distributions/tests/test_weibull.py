

import pytest
import mpmath
from mpsci.distributions import weibull_max, weibull_min


@pytest.mark.parametrize('dist, xsign', [(weibull_min, 1), (weibull_max, -1)])
def test_pdf(dist, xsign):
    with mpmath.workdps(50):
        k = 1.25
        loc = 1
        scale = 3
        x = 2.5
        p = dist.pdf(xsign*x, k, xsign*loc, scale)
        # Expected value was computed with Wolfram Alpha:
        #   PDF[WeibullDistribution[5/4, 3, 1], 5/2]
        valstr = '0.23010863853495101956594599926808749710908978279269136511'
        expected = mpmath.mpf(valstr)
        assert mpmath.almosteq(p, expected)


@pytest.mark.parametrize('dist', [weibull_min, weibull_max])
def test_cdf_sf(dist):
    with mpmath.workdps(50):
        k = 1.25
        loc = 1
        scale = 3
        x = 2.5
        if dist == weibull_min:
            cdf = dist.cdf(x, k, loc, scale)
            sf = dist.sf(x, k, loc, scale)
        else:
            cdf = dist.sf(-x, k, -loc, scale)
            sf = dist.cdf(-x, k, -loc, scale)
        # Expected value computed with Wolfram Alpha:
        #   CDF[WeibullDistribution[5/4, 3, 1], 5/2]
        valstr = '0.34324760759355263295507068694174586069396497366940988665'
        expected = mpmath.mpf(valstr)
        assert mpmath.almosteq(cdf, expected)
        assert mpmath.almosteq(sf, 1 - expected)


@pytest.mark.parametrize('dist, sign', [(weibull_min, 1), (weibull_max, -1)])
def test_skewness(dist, sign):
    with mpmath.workdps(50):
        k = 1.25
        loc = 1
        scale = 3
        skew = dist.skewness(k, loc, scale)
        # Expected value computed with Wolfram Alpha:
        #   Skewness[WeibullDistribution[5/4, 3, 1]]
        valstr = '1.429545236590974853525527387620583784997166374292021040338'
        expected = mpmath.mpf(valstr)
        assert mpmath.almosteq(skew, sign*expected)


@pytest.mark.parametrize('dist', [weibull_min, weibull_max])
def test_kurtosis(dist):
    with mpmath.workdps(50):
        k = 1.25
        loc = 1
        scale = 3
        kurt = dist.kurtosis(k, loc, scale)
        # Expected value computed with Wolfram Alpha:
        #   ExcessKurtosis[WeibullDistribution[5/4, 3, 1]]
        valstr = '2.8021519350984650074697694858304410798423229238041266467027'
        expected = mpmath.mpf(valstr)
        assert mpmath.almosteq(kurt, expected)
