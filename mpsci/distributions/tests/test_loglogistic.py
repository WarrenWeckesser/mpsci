import pytest
from mpmath import mp
from mpsci.distributions import loglogistic
from ._expect import check_noncentral_moment_with_integral


def test_bad_params():
    with pytest.raises(ValueError, match='beta must be greater than 0'):
        loglogistic.pdf(1, -2, 3)
    with pytest.raises(ValueError, match='scale must be greater than 0'):
        loglogistic.pdf(1, 2, -3)


def test_pdf_logpdf_out_of_support():
    beta = 3
    scale = 8
    x = -1
    assert loglogistic.pdf(x, beta, scale) == 0
    assert loglogistic.logpdf(x, beta, scale) == mp.ninf


@mp.workdps(50)
def test_pdf():
    x = mp.mpf(1.5)
    beta = mp.mpf(3)
    scale = mp.mpf(0.25)
    pdf = loglogistic.pdf(x, beta, scale)
    # Expected value computed with Wolfram Alpha:
    #   PDF[LogLogisticDistribution[3, 1/4], 3/2]
    expected_pdf = mp.mpf('432/47089')
    assert mp.almosteq(pdf, expected_pdf)


@mp.workdps(50)
def test_logpdf():
    x = mp.mpf(1.5)
    beta = mp.mpf(3)
    scale = mp.mpf(0.25)
    logpdf = loglogistic.logpdf(x, beta, scale)
    # Expected value computed with Wolfram Alpha:
    #   PDF[LogLogisticDistribution[3, 1/4], 3/2]
    expected_logpdf = mp.log(mp.mpf('432/47089'))
    assert mp.almosteq(logpdf, expected_logpdf)


@mp.workdps(50)
def test_cdf():
    x = mp.mpf(1.5)
    beta = mp.mpf(3)
    scale = mp.mpf(0.25)
    cdf = loglogistic.cdf(x, beta, scale)
    # Expected value computed with Wolfram Alpha:
    #   CDF[LogLogisticDistribution[3, 1/4], 3/2]
    expected_cdf = mp.mpf('216/217')
    assert mp.almosteq(cdf, expected_cdf)


@mp.workdps(50)
def test_sf():
    x = mp.mpf(1.5)
    beta = mp.mpf(3)
    scale = mp.mpf(0.25)
    sf = loglogistic.sf(x, beta, scale)
    # Expected value computed with Wolfram Alpha:
    #   CDF[LogLogisticDistribution[3, 1/4], 3/2]
    expected_sf = mp.mpf('1/217')
    assert mp.almosteq(sf, expected_sf)


def test_cdf_sf_out_of_support():
    beta = 2
    scale = 8
    x = -1
    assert loglogistic.cdf(x, beta, scale) == 0
    assert loglogistic.sf(x, beta, scale) == 1


@mp.workdps(50)
def test_invcdf():
    p = mp.mpf(0.75)
    beta = mp.mpf(3)
    scale = mp.mpf(0.25)
    x = loglogistic.invcdf(p, beta, scale)
    # Expected value computed with Wolfram Alpha:
    #   InverseCDF[LogLogisticDistribution[3, 1/4], 3/4]
    expected_x = mp.cbrt(3)/4
    assert mp.almosteq(x, expected_x)


@mp.workdps(50)
def test_invsf():
    p = mp.mpf(0.25)
    beta = mp.mpf(3)
    scale = mp.mpf(0.25)
    x = loglogistic.invsf(p, beta, scale)
    # Expected value computed with Wolfram Alpha:
    #   InverseCDF[LogLogisticDistribution[3, 1/4], 3/4]
    expected_x = mp.cbrt(3)/4
    assert mp.almosteq(x, expected_x)


@mp.workdps(50)
def test_mean():
    beta = mp.mpf(3)
    scale = mp.mpf(0.25)
    mean = loglogistic.mean(beta, scale)
    # Expected value computed with Wolfram Alpha:
    #   Mean[LogLogisticDistribution[3, 1/4]]
    expected_mean = mp.pi/(6*mp.sqrt(3))
    assert mp.almosteq(mean, expected_mean)


@mp.workdps(50)
def test_var():
    beta = mp.mpf(3)
    scale = mp.mpf(0.25)
    var = loglogistic.var(beta, scale)
    # Expected value computed with Wolfram Alpha:
    #   Variance[LogLogisticDistribution[3, 1/4]]
    expected_var = (mp.sqrt(3) - mp.pi/3)*mp.pi/36
    assert mp.almosteq(var, expected_var)


@pytest.mark.parametrize('order', [0, 1, 2, 3, 4])
@mp.workdps(50)
def test_noncentral_moment_with_integral(order):
    check_noncentral_moment_with_integral(order, loglogistic, (4.5, 1.75))
