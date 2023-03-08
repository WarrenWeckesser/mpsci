
import pytest
from mpmath import mp
from mpsci.distributions import invgauss


@mp.workdps(50)
def test_pdf():
    # Wolfram Alpha:
    #   PDF[InverseGaussianDistribution[3, 1], 1] => 1/(e^{2/9}*sqrt(2*pi)
    p = invgauss.pdf(1, 3, loc=0, scale=1)
    assert mp.almosteq(p, mp.exp(mp.mpf(-2)/9)/mp.sqrt(2*mp.pi))

    #   PDF[InverseGaussianDistribution[3, 2], 5] => 1/(5*e^{4/45}*sqrt(5*pi))
    p = invgauss.pdf(5, 3/2, loc=0, scale=2)
    assert mp.almosteq(p, mp.one/5/mp.exp(mp.mpf(4)/45)/mp.sqrt(5*mp.pi))


@mp.workdps(50)
def test_cdf():
    # Wolfram Alpha:
    #      CDF[InverseGaussianDistribution[3, 2], 5]
    p = invgauss.cdf(5, 3/2, loc=0, scale=2)
    val = '0.837276248854267498706462771732316695810386079642075926'
    assert mp.almosteq(p, mp.mpf(val))


@mp.workdps(50)
def test_cdf_invcdf_roundtrip():
    x0 = 5
    p = invgauss.cdf(x0, 3/2, loc=1, scale=2)
    x1 = invgauss.invcdf(p, 3/2, loc=1, scale=2)
    assert mp.almosteq(x1, x0)


@mp.workdps(50)
def test_sf():
    # Wolfram Alpha:
    #      1 - CDF[InverseGaussianDistribution[3, 2], 5]
    p = invgauss.sf(5, 3/2, loc=0, scale=2)
    val = '0.162723751145732501293537228267683304189613920357924074'
    assert mp.almosteq(p, mp.mpf(val))


@mp.workdps(50)
def test_sf_invsf_roundtrip():
    x0 = 5
    p = invgauss.sf(x0, 3/2, loc=1, scale=2)
    x1 = invgauss.invsf(p, 3/2, loc=1, scale=2)
    assert mp.almosteq(x1, x0)


@pytest.mark.parametrize('m', ['1e-6', 3])
def test_entropy_against_integral(m):

    def integrand(x):
        # This is a closure that captures m.
        return invgauss.pdf(x, m) * invgauss.logpdf(x, m)

    with mp.workdps(50):
        m = mp.mpf(m)
        entr = invgauss.entropy(m)
        with mp.extradps(25):
            mode = invgauss.mode(m)
            val = -mp.quad(integrand, [0, mode, mp.inf])
        assert mp.almosteq(entr, val)
