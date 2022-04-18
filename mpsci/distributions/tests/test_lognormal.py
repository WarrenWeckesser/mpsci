
import mpmath
from mpsci.distributions import lognormal


def test_mean():
    mu = 2.0
    sigma = 3.0
    # Wolfram Alpha:
    #     Mean[LogNormalDistribution[2, 3]]
    # returns
    #     665.141633044361840693961494242634383221132254094828803184906532...
    s = '665.141633044361840693961494242634383221132254094828803184906532'
    with mpmath.workdps(len(s) - 1):
        expected = mpmath.mpf(s)
        mean = lognormal.mean(mu, sigma)
        assert mpmath.almosteq(mean, expected)


def test_var():
    mu = -1
    sigma = 3/4
    # Wolfram Alpha:
    #     Var[LogNormalDistribution[-1, 3/4]]
    # returns
    #     0.17934120058305027077170627780841441260829384280347521202891893...
    s = '0.17934120058305027077170627780841441260829384280347521202891893'
    with mpmath.workdps(len(s) - 1):
        expected = mpmath.mpf(s)
        var = lognormal.var(mu, sigma)
        assert mpmath.almosteq(var, expected)


def test_skewness():
    mu = -1
    sigma = 3/4
    # Wolfram Alpha:
    #     Skewness[LogNormalDistribution[-1, 3/4]]
    # returns
    #     3.26291272820700198848298205623905499569637496768603483385045187...
    s = '3.2629127282070019884829820562390549956963749676860348338504519'
    with mpmath.workdps(len(s) - 1):
        expected = mpmath.mpf(s)
        skew = lognormal.skewness(mu, sigma)
        assert mpmath.almosteq(skew, expected)


def test_kurtosis():
    mu = -1
    sigma = 3/4
    # Wolfram Alpha:
    #     ExcessKurtosis[LogNormalDistribution[-1, 3/4]]
    # returns
    #     23.5402842333949536453917913054445343888977841929771374742004...'
    s = '23.5402842333949536453917913054445343888977841929771374742004'
    with mpmath.workdps(len(s) - 1):
        expected = mpmath.mpf(s)
        kurt = lognormal.kurtosis(mu, sigma)
        assert mpmath.almosteq(kurt, expected)
