import pytest
from mpmath import mp
from mpsci.distributions import lognormal
from ._utils import call_and_check_mle


def test_pdf():
    x = 5.0
    mu = 2.0
    sigma = 3.0
    # Wolfram Alpha:
    #    PDF[LogNormalDistribution[2, 3], 5]
    s = '0.026371718579138012712645997572295063382723046111242729868'
    with mp.workdps(len(s) - 3):
        expected = mp.mpf(s)
        pdf = lognormal.pdf(x, mu, sigma)
        assert mp.almosteq(pdf, expected)


def test_logpdf():
    x = 5.0
    mu = 2.0
    sigma = 3.0
    # Wolfram Alpha:
    #    log[PDF[LogNormalDistribution[2, 3], 5]]
    s = '-3.63546310898709577704173062167032566943949209403601586605421466'
    with mp.workdps(len(s) - 2):
        expected = mp.mpf(s)
        logpdf = lognormal.logpdf(x, mu, sigma)
        assert mp.almosteq(logpdf, expected)


def test_cdf():
    x = 5.0
    mu = 2.0
    sigma = 3.0
    # Wolfram Alpha:
    #    CDF[LogNormalDistribution[2, 3], 5]
    s = '0.4482090963664131928987017320145556980701415906740292202885'
    with mp.workdps(len(s) - 2):
        expected = mp.mpf(s)
        cdf = lognormal.cdf(x, mu, sigma)
        assert mp.almosteq(cdf, expected)


def test_invcdf():
    mu = 2.0
    sigma = 3.0
    # Wolfram Alpha:
    #    InverseCDF[LogNormalDistribution[2, 3], 1/100]
    s = '0.0068806399035674338044297169530782501683046556866941825057284'
    with mp.workdps(len(s) - 4):
        p = mp.mpf('0.01')
        expected = mp.mpf(s)
        invcdf = lognormal.invcdf(p, mu, sigma)
        assert mp.almosteq(invcdf, expected)


def test_sf():
    x = 5.0
    mu = 2.0
    sigma = 3.0
    # Wolfram Alpha:
    #    SurvivalFunction[LogNormalDistribution[2, 3], 5]
    s = '0.55179090363358680710129826798544430192985840932597077971147'
    with mp.workdps(len(s) - 2):
        expected = mp.mpf(s)
        sf = lognormal.sf(x, mu, sigma)
        assert mp.almosteq(sf, expected)


def test_invsf():
    mu = 2.0
    sigma = 3.0
    # Wolfram Alpha:
    #    InverseCDF[LogNormalDistribution[2, 3], 99/100]
    s = '7935.0395890993380729703921306125939057678238091794644776865236'
    with mp.workdps(len(s) - 1):
        p = mp.mpf('0.01')
        expected = mp.mpf(s)
        invsf = lognormal.invsf(p, mu, sigma)
        assert mp.almosteq(invsf, expected)


def test_mean():
    mu = 2.0
    sigma = 3.0
    # Wolfram Alpha:
    #     Mean[LogNormalDistribution[2, 3]]
    # returns
    #     665.141633044361840693961494242634383221132254094828803184906532...
    s = '665.141633044361840693961494242634383221132254094828803184906532'
    with mp.workdps(len(s) - 1):
        expected = mp.mpf(s)
        mean = lognormal.mean(mu, sigma)
        assert mp.almosteq(mean, expected)


def test_var():
    mu = -1
    sigma = 3/4
    # Wolfram Alpha:
    #     Var[LogNormalDistribution[-1, 3/4]]
    # returns
    #     0.17934120058305027077170627780841441260829384280347521202891893...
    s = '0.17934120058305027077170627780841441260829384280347521202891893'
    with mp.workdps(len(s) - 1):
        expected = mp.mpf(s)
        var = lognormal.var(mu, sigma)
        assert mp.almosteq(var, expected)


def test_skewness():
    mu = -1
    sigma = 3/4
    # Wolfram Alpha:
    #     Skewness[LogNormalDistribution[-1, 3/4]]
    # returns
    #     3.26291272820700198848298205623905499569637496768603483385045187...
    s = '3.2629127282070019884829820562390549956963749676860348338504519'
    with mp.workdps(len(s) - 1):
        expected = mp.mpf(s)
        skew = lognormal.skewness(mu, sigma)
        assert mp.almosteq(skew, expected)


def test_kurtosis():
    mu = -1
    sigma = 3/4
    # Wolfram Alpha:
    #     ExcessKurtosis[LogNormalDistribution[-1, 3/4]]
    # returns
    #     23.5402842333949536453917913054445343888977841929771374742004...'
    s = '23.5402842333949536453917913054445343888977841929771374742004'
    with mp.workdps(len(s) - 1):
        expected = mp.mpf(s)
        kurt = lognormal.kurtosis(mu, sigma)
        assert mp.almosteq(kurt, expected)


@mp.workdps(50)
def test_entropy_with_integral():
    mu = 2
    sigma = 3
    entr = lognormal.entropy(mu, sigma)
    with mp.extradps(mp.dps // 2):

        def integrand(t):
            return lognormal.pdf(t, mu, sigma) * lognormal.logpdf(t, mu, sigma)

        intgrl = mp.quad(integrand, [0, mp.inf])

    assert mp.almosteq(entr, -intgrl)


def test_noncentral_moment():
    mu = 2
    sigma = 3
    with mp.workdps(50):
        assert lognormal.noncentral_moment(0, mu, sigma) == 1
        assert mp.almosteq(lognormal.noncentral_moment(1, mu, sigma),
                           mp.exp(13/2))
        assert mp.almosteq(lognormal.noncentral_moment(2, mu, sigma),
                           mp.exp(22))
        assert mp.almosteq(lognormal.noncentral_moment(3, mu, sigma),
                           mp.exp(93/2))
        assert mp.almosteq(lognormal.noncentral_moment(4, mu, sigma),
                           mp.exp(80))


@pytest.mark.parametrize(
    'x',
    [[0.03125, 0.0625, 0.125, 0.25, 0.5, 1],
     [1, 2, 3, 5, 8, 13, 21, 34, 55, 89],
     [3, 3.25, 4.25, 8, 8.5, 9.125]]
)
@mp.workdps(50)
def test_mle(x):
    call_and_check_mle(lognormal.mle, lognormal.nll, x)
