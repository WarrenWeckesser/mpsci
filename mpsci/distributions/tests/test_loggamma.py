import pytest
from mpmath import mp
from mpsci.distributions import loggamma, Initial
from ._utils import check_mle


def test_basic_cdf_sf():
    with mp.workdps(50):
        k = 2
        theta = 7
        x = 1
        p = loggamma.cdf(x, k, theta)
        # From Wolfram Alpha:
        #    CDF[ExpGammaDistribution[2, 7, 0], 1]
        val = '0.3205287717282638332374539773545351545714864926297322725929829'
        expected = mp.mpf(val)
        assert mp.almosteq(p, expected)

        p = loggamma.sf(x, k, theta)
        assert mp.almosteq(p, 1 - expected)


@pytest.mark.parametrize('funcpair', [(loggamma.cdf, loggamma.invcdf),
                                      (loggamma.sf, loggamma.invsf)])
@pytest.mark.parametrize('params',
                         [(1, 1), (0.125, 1.5), (1.25, 0.75)])
@pytest.mark.parametrize('x0', [-2, -1, 0, 1, 2])
def test_dist_roundtrip(funcpair, params, x0):
    with mp.workdps(50):
        func, invfunc = funcpair
        k, theta = params
        p = func(x0, k, theta)
        x1 = invfunc(p, k, theta)
        assert mp.almosteq(x1, x0)


def test_mean():
    with mp.workdps(50):
        k = mp.mpf(2)
        theta = mp.mpf(7)
        mean = loggamma.mean(k, theta)
        # Expected value computed with Wolfram Alpha:
        #     Mean[ExpGammaDistribution[2, 7, 0]]
        valstr = '2.9594903456892699757544153694231829827048846484205348083596'
        expected = mp.mpf(valstr)
        assert mp.almosteq(mean, expected)


def test_var():
    with mp.workdps(50):
        k = mp.mpf(2)
        theta = mp.mpf(7)
        var = loggamma.var(k, theta)
        # Expected value computed with Wolfram Alpha:
        #     Variance[ExpGammaDistribution[2, 7, 0]]
        valstr = '31.601769275563095387148343165655234271728545159133123449042'
        expected = mp.mpf(valstr)
        assert mp.almosteq(var, expected)


def test_skewness():
    with mp.workdps(50):
        k = mp.mpf(2)
        theta = mp.mpf(7)
        skew = loggamma.skewness(k, theta)
        # Expected value computed with Wolfram Alpha:
        #     Skewness[ExpGammaDistribution[2, 7, 0]]
        valstr = '-0.780244491437787061530103769675602605070996940757461555988'
        expected = mp.mpf(valstr)
        assert mp.almosteq(skew, expected)


def test_kurtosis():
    with mp.workdps(50):
        k = mp.mpf(2)
        theta = mp.mpf(7)
        kurt = loggamma.kurtosis(k, theta)
        # Expected value computed with Wolfram Alpha:
        #     ExcessKurtosis[ExpGammaDistribution[2, 7, 0]]
        valstr = '1.1875257511965620472368652875718220772212082979314426565409'
        expected = mp.mpf(valstr)
        assert mp.almosteq(kurt, expected)


@pytest.mark.parametrize(
    'x, theta0',
    [([0.5, 1, 1.5, 3], 1),
     ([0.01, 0.008, 0.006, 0.009, 0.009, 0.005, 0.011, 0.010], 0.005)],
)
@mp.workdps(50)
def test_mle(x, theta0):
    p_hat = loggamma.mle(x, theta=Initial(theta0))
    check_mle(loggamma.nll, x, p_hat)


@pytest.mark.parametrize(
    'x, fixed_theta',
    [([0.5, 1, 1.5, 3], 3),
     ([0.01, 0.008, 0.006, 0.009, 0.009, 0.005, 0.011, 0.010], 0.25)],
)
@mp.workdps(50)
def test_mle_theta_fixed(x, fixed_theta):
    p_hat = loggamma.mle(x, theta=fixed_theta)
    check_mle(lambda x, k: loggamma.nll(x, k, fixed_theta), x, p_hat[:1])


@pytest.mark.parametrize(
    'x, fixed_k',
    [([0.5, 1, 1.5, 3], 1),
     ([1, 2, 3, 4, 5, 8, 13], 11.0)],
)
@mp.workdps(50)
def test_mle_k_fixed(x, fixed_k):
    p_hat = loggamma.mle(x, k=fixed_k)
    check_mle(lambda x, theta: loggamma.nll(x, fixed_k, theta), x, p_hat[1:])