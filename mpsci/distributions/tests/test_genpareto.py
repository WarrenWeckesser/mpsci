
import pytest
from mpmath import mp
from mpsci.distributions import genpareto


@mp.workdps(50)
def test_pdf():
    xi = 3
    x = 5
    p = genpareto.pdf(x, xi)
    # The formula for the expected value as given by Mathematica:
    #    PDF[ParetoPickandsDistribution[3], 5]
    expected = mp.power(2, -mp.one/3)/32
    assert mp.almosteq(p, expected)


@mp.workdps(50)
def test_cdf_sf():
    # The formula for the expected values was given by Mathematica:
    #     CDF[ParetoPickandsDistribution[3], 5]
    xi = 3
    x = 5
    sf = genpareto.sf(x, xi)
    expected_sf = mp.power(2, -mp.one/3)/2
    assert mp.almosteq(sf, expected_sf)

    cdf = genpareto.cdf(x, xi)
    assert mp.almosteq(cdf, 1 - expected_sf)


@pytest.mark.parametrize('func, invfunc',
                         [(genpareto.cdf, genpareto.invcdf),
                          (genpareto.sf, genpareto.invsf)])
@pytest.mark.parametrize('x, xi, mu, sigma',
                         [(8, 1.5, 2, 3.0),
                          (3, 0.5, 2.5, 10.0),
                          (3, 0, 2, 3.0),
                          (3, -2, 1.5, 5.0)])
@mp.workdps(50)
def test_cdf_invcdf_sf_invsf_roundtrip(func, invfunc, x, xi, mu, sigma):
    p = func(x, xi, mu, sigma)
    x1 = invfunc(p, xi, mu, sigma)
    assert mp.almosteq(x1, x)


@mp.workdps(50)
def test_nll():
    x = [0.5, 1.5, 2.0, 8.0]
    xi = 3.0
    mu = -1.0
    sigma = 4.0
    nll = genpareto.nll(x, xi, mu, sigma)
    # XXX Not really a great test. This is the same implementation as
    # in genpareto.nll().
    s = -mp.fsum([genpareto.logpdf(t, xi, mu, sigma) for t in x])
    assert mp.almosteq(nll, s)
