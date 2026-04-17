import pytest
from mpmath import mp
from mpsci.distributions import genextreme
from ._expect import (
    check_entropy_with_integral,
    check_skewness_with_integral,
    check_kurtosis_with_integral,
)


def test_support_xi_eq_0():
    xi = 0
    mu = 4
    sigma = 3
    sup = genextreme.support(xi, mu, sigma)
    assert sup == (mp.ninf, mp.inf)


@mp.workdps(50)
def test_basic_pdf():
    # The expected values were computed "by hand".
    expected = mp.exp(mp.mpf('-0.5'))/16
    assert mp.almosteq(genextreme.pdf(6, 2, 3, 2), expected)
    expected = mp.exp(mp.mpf('-2'))/4
    assert mp.almosteq(genextreme.pdf(0, -2, 3, 2), expected)


@mp.workdps(50)
def test_pdf_from_wolfram_alpha():
    x = 4
    xi = 2
    mu = 0
    sigma = 1
    p = genextreme.pdf(x, xi, mu, sigma)
    # Wolfram Alpha (https://www.wolframalpha.com):
    #    PDF[MaxStableDistribution[0, 1, 2], 4]
    refstr = '0.02653819668791812038613348507131035805381896517857330'
    assert mp.almosteq(p, mp.mpf(refstr))


def test_pdf_bad_sigma():
    x = 1
    xi = -3
    mu = 0
    sigma = -0.25
    with pytest.raises(ValueError, match='sigma must be positive'):
        genextreme.pdf(x, xi, mu, sigma)


@pytest.mark.parametrize('x, xi, mu, sigma', [(4, -2.5, -1, 5), (-6, 0.5, 3, 4)])
@mp.workdps(50)
def test_pdf_logpdf_outside_support(x, xi, mu, sigma):
    p = genextreme.pdf(x, xi, mu, sigma)
    assert p == 0
    logp = genextreme.logpdf(x, xi, mu, sigma)
    assert logp == mp.ninf


@mp.workdps(50)
def test_logpdf_xi_eq_0_from_wolfram_alpha():
    x = 2.5
    xi = 0
    mu = -1
    sigma = 1.5
    logp = genextreme.logpdf(x, xi, mu, sigma)
    # Wolfram Alpha (https://www.wolframalpha.com):
    #    Log[PDF[MaxStableDistribution[-1, 3/2, 0], 5/2]]
    refstr = '-2.8357704093059027781212531080960532013860446157206319'
    assert mp.almosteq(logp, mp.mpf(refstr))


@mp.workdps(50)
def test_basic_logpdf():
    # The expected values were computed "by hand".
    expected = mp.mpf('-0.5') - mp.log(16)
    assert mp.almosteq(genextreme.logpdf(6, 2, 3, 2), expected)
    expected = mp.mpf('-2') - mp.log(4)
    assert mp.almosteq(genextreme.logpdf(0, -2, 3, 2), expected)


@mp.workdps(50)
def test_pdf_xi_pos_x_at_bound():
    xi = mp.mpf(2)
    mu = 0
    sigma = 1
    p = genextreme.pdf(-1/xi, xi, mu, sigma)
    assert p == 0


@mp.workdps(50)
def test_basic_cdf():
    # The expected value was computed "by hand".
    expected = mp.exp(mp.mpf('-0.5'))
    assert mp.almosteq(genextreme.cdf(6, 2, 3, 2), expected)


@mp.workdps(50)
def test_cdf_from_wolfram_alpha():
    # The reference value was computed with Wolfram Alpha:
    #    CDF[MaxStableDistribution[0, 1, 3/2], 1]
    ref = mp.mpf('0.5810703045626315358283025571396011491804386988291087')
    p = genextreme.cdf(1, 1.5, 0, 1)
    assert mp.almosteq(p, ref)


@pytest.mark.parametrize('xi, x', [(1, -3), (2, -0.75)])
def test_logcdf_below_support(xi, x):
    logp = genextreme.logcdf(x, xi, 0, 1)
    assert logp == mp.ninf


@pytest.mark.parametrize('xi, x', [(-1, 1.25), (-2, 0.75)])
def test_logcdf_above_support(xi, x):
    logp = genextreme.logcdf(x, xi, 0, 1)
    assert logp == 0


@mp.workdps(50)
def test_logcdf_from_wolfram_alpha():
    with mp.workdps(100):
        # From Wolfram Alpha:
        #   CDF[MaxStableDistribution[0, 1, 1/2], 10000000000000000000000000]
        cdf_ref = ('0.999999999999999999999999999999999999999999999999960000'
                   '000000000000000000015999999999999999999999996')
        logcdf_ref = mp.log(mp.mpf(cdf_ref))
    logp = genextreme.logcdf(1e25, 0.5, 0, 1)
    assert mp.almosteq(logp, logcdf_ref)


@mp.workdps(50)
def test_basic_sf():
    # The expected value was computed "by hand".
    expected = 1 - mp.exp(mp.mpf('-0.5'))
    assert mp.almosteq(genextreme.sf(6, 2, 3, 2), expected)


@mp.workdps(50)
def test_sf_from_wolfram_alpha():
    # The reference value was computed with Wolfram Alpha:
    #    SurvivalFunction[MaxStableDistribution[0, 1, 3/2],
    #                     10000000000000000000000000]
    ref = mp.mpf('3.999999999999999999999998400000000000000000000000'
                 '399999999999999999999999936e-50')
    p = genextreme.sf(1e25, 0.5, 0, 1)
    assert mp.almosteq(p, ref)


@pytest.mark.parametrize('xi, x', [(1, -3), (2, -0.75)])
def test_logsf_below_support(xi, x):
    logp = genextreme.logsf(x, xi, 0, 1)
    assert logp == 0


@pytest.mark.parametrize('xi, x', [(-1, 1.25), (-2, 0.75)])
def test_logsf_above_support(xi, x):
    logp = genextreme.logsf(x, xi, 0, 1)
    assert logp == mp.ninf


@pytest.mark.parametrize('x, xi, mu, sigma',
                         [(3.0, 0.5, -3, 0.25),
                          (-0.5, 1.5, 0.5, 4.0),
                          (-2.0, 0.0, 5.0, 3.0),
                          (0.0, -0.5, 2, 1.5),
                          (1.5, -2.0, 2, 1.5)])
@mp.workdps(50)
def test_cdf_invcdf_roundtrip(x, xi, mu, sigma):
    p = genextreme.cdf(x, xi, mu, sigma)
    x1 = genextreme.invcdf(p, xi, mu, sigma)
    assert mp.almosteq(x1, x)


@pytest.mark.parametrize('x, xi, mu, sigma',
                         [(3.0, 0.5, -3, 0.25),
                          (-0.5, 1.5, 0.5, 4.0),
                          (-2.0, 0.0, 5.0, 3.0),
                          (0.0, -0.5, 2, 1.5),
                          (1.5, -2.0, 2, 1.5)])
@mp.workdps(50)
def test_sf_invsf_roundtrip(x, xi, mu, sigma):
    p = genextreme.sf(x, xi, mu, sigma)
    x1 = genextreme.invsf(p, xi, mu, sigma)
    assert mp.almosteq(x1, x)


@mp.workdps(50)
def test_mean():
    xi = mp.mpf('0.5')
    mu = 3
    sigma = 2
    g1 = mp.gamma(1 - xi)
    assert mp.almosteq(genextreme.mean(xi, mu, sigma),
                       mu + sigma * (g1 - 1)/xi)


@mp.workdps(50)
def test_mean_xi_zero():
    xi = 0
    mu = 3
    sigma = 2
    assert mp.almosteq(genextreme.mean(xi, mu, sigma),
                       mu + sigma * mp.euler)


@mp.workdps(50)
def test_inf_mean():
    assert genextreme.mean(2, 3, 2) == mp.inf


@pytest.mark.parametrize(
    'xi, mu, sigma',
    [(-1, 0.25, 1), (-0.25, 0, 3), (0, 0, 1), (0, 2, 3), (0.75, -10, 25),
     (2, 4, 2.5), (3, -1, 4), (8, -3, 0.25)]
)
@mp.workdps(50)
def test_mode_xi_ge_neg1(xi, mu, sigma):
    # A crude test of the mode.
    m = genextreme.mode(xi, mu, sigma)
    pm = genextreme.pdf(m, xi, mu, sigma)
    delta = mp.sqrt(mp.eps)
    if m == 0:
        left = -delta
        right = delta
    else:
        left = (1 - mp.sign(m) * delta) * m
        right = (1 + mp.sign(m) * delta) * m
    assert genextreme.pdf(left, xi, mu, sigma) < pm
    assert genextreme.pdf(right, xi, mu, sigma) < pm


@pytest.mark.parametrize('xi, mu, sigma',
                         [(-1.25, 2, 3), (-2, 0, 1), (-5, -1, 0.125)])
@mp.workdps(35)
def test_mode_xi_lt_neg1(xi, mu, sigma):
    m = genextreme.mode(xi, mu, sigma)
    sup = genextreme.support(xi, mu, sigma)
    assert mp.almosteq(m, sup[1])


@mp.workdps(50)
def test_var():
    xi = mp.mpf('0.25')
    mu = 3
    sigma = 2
    g1 = mp.gamma(1 - xi)
    g2 = mp.gamma(1 - 2*xi)
    assert mp.almosteq(genextreme.var(xi, mu, sigma),
                       sigma**2 * (g2 - g1**2) / xi**2)


@mp.workdps(50)
def test_var_xi_zero():
    xi = 0
    mu = 3
    sigma = 2
    assert mp.almosteq(genextreme.var(xi, mu, sigma),
                       sigma**2 * mp.pi**2 / 6)


@mp.workdps(50)
def test_inf_var():
    assert genextreme.var(2, 3, 2) == mp.inf


@pytest.mark.parametrize('xi, mu, sigma',
                         [(0.25, 0, 1),
                          (-3, 2, 5)])
@mp.workdps(50)
def test_noncentral_moment_trivial_cases(xi, mu, sigma):
    mom0 = genextreme.noncentral_moment(0, xi, mu, sigma)
    assert mom0 == 1

    mom1 = genextreme.noncentral_moment(1, xi, mu, sigma)
    assert mp.almosteq(mom1, genextreme.mean(xi, mu, sigma))

    mom2 = genextreme.noncentral_moment(2, xi, mu, sigma)
    altmom2 = genextreme.var(xi, mu, sigma) + genextreme.mean(xi, mu, sigma)**2
    assert mp.almosteq(mom2, altmom2)


@mp.workdps(50)
def test_entropy_with_integral():
    xi = 0.75
    mu = 2.5
    sigma = 8.0
    check_entropy_with_integral(genextreme, (xi, mu, sigma))


@pytest.mark.parametrize('xi, mu, sigma', [(0.125, 2.5, 8.0),
                                           (0.25, 0.0, 1.0),
                                           (-1.25, 0.0, 1.0)])
@mp.workdps(50)
def test_skewness_with_integral(xi, mu, sigma):
    check_skewness_with_integral(genextreme, (xi, mu, sigma))


@mp.workdps(50)
def test_skewness_xi_eq_0_with_custom_integral():
    sk = genextreme.skewness(0, 0, 1)
    with mp.extradps(2*mp.dps):
        eps = mp.eps / 2
        x0 = mp.lambertw(-eps, k=-1) - mp.log(eps)
        mn = genextreme.mean(0, 0, 1)
        sd = mp.sqrt(genextreme.var(0, 0, 1))
        q = mp.quad(lambda x: ((x - mn)/sd)**3 * mp.exp(-(x + mp.exp(-x))),
                    [x0, 0, mp.inf],
                    method='tanh-sinh')
    assert mp.almosteq(sk, q)


def test_skewness_xi_gt_one_third():
    sk = genextreme.skewness(0.75, 0, 1)
    assert sk == mp.inf


@mp.workdps(50)
def test_kurtosis_with_integral():
    xi = 0.125
    mu = 2.5
    sigma = 8.0
    check_kurtosis_with_integral(genextreme, (xi, mu, sigma))


@mp.workdps(50)
def test_kurtosis_xi_eq_0_with_custom_integral():
    k = genextreme.kurtosis(0, 0, 1)
    with mp.extradps(2*mp.dps):
        eps = mp.eps / 16
        x0 = mp.lambertw(-eps, k=-1) - mp.log(eps)
        mn = genextreme.mean(0, 0, 1)
        sd = mp.sqrt(genextreme.var(0, 0, 1))
        q = mp.quad(lambda x: ((x - mn)/sd)**4 * mp.exp(-(x + mp.exp(-x))),
                    [x0, 0, mp.inf],
                    method='tanh-sinh')
    # Subtract 3 from q for the excess kurtosis.
    assert mp.almosteq(k, q - 3)


def test_kurtosis_xi_gt_one_fourth():
    k = genextreme.kurtosis(0.5, 0, 1)
    assert k == mp.inf
