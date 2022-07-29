
import mpmath
from mpsci.distributions import loglogistic


def test_pdf():
    with mpmath.workdps(50):
        x = mpmath.mpf(1.5)
        beta = mpmath.mpf(3)
        scale = mpmath.mpf(0.25)
        pdf = loglogistic.pdf(x, beta, scale)
        # Expected value computed with Wolfram Alpha:
        #   PDF[LogLogisticDistribution[3, 1/4], 3/2]
        expected_pdf = mpmath.mpf('432/47089')
        assert mpmath.almosteq(pdf, expected_pdf)


def test_logpdf():
    with mpmath.workdps(50):
        x = mpmath.mpf(1.5)
        beta = mpmath.mpf(3)
        scale = mpmath.mpf(0.25)
        logpdf = loglogistic.logpdf(x, beta, scale)
        # Expected value computed with Wolfram Alpha:
        #   PDF[LogLogisticDistribution[3, 1/4], 3/2]
        expected_logpdf = mpmath.log(mpmath.mpf('432/47089'))
        assert mpmath.almosteq(logpdf, expected_logpdf)


def test_cdf():
    with mpmath.workdps(50):
        x = mpmath.mpf(1.5)
        beta = mpmath.mpf(3)
        scale = mpmath.mpf(0.25)
        cdf = loglogistic.cdf(x, beta, scale)
        # Expected value computed with Wolfram Alpha:
        #   CDF[LogLogisticDistribution[3, 1/4], 3/2]
        expected_cdf = mpmath.mpf('216/217')
        assert mpmath.almosteq(cdf, expected_cdf)


def test_sf():
    with mpmath.workdps(50):
        x = mpmath.mpf(1.5)
        beta = mpmath.mpf(3)
        scale = mpmath.mpf(0.25)
        sf = loglogistic.sf(x, beta, scale)
        # Expected value computed with Wolfram Alpha:
        #   CDF[LogLogisticDistribution[3, 1/4], 3/2]
        expected_sf = mpmath.mpf('1/217')
        assert mpmath.almosteq(sf, expected_sf)


def test_invcdf():
    with mpmath.workdps(50):
        p = mpmath.mpf(0.75)
        beta = mpmath.mpf(3)
        scale = mpmath.mpf(0.25)
        x = loglogistic.invcdf(p, beta, scale)
        # Expected value computed with Wolfram Alpha:
        #   InverseCDF[LogLogisticDistribution[3, 1/4], 3/4]
        expected_x = mpmath.cbrt(3)/4
        assert mpmath.almosteq(x, expected_x)


def test_invsf():
    with mpmath.workdps(50):
        p = mpmath.mpf(0.25)
        beta = mpmath.mpf(3)
        scale = mpmath.mpf(0.25)
        x = loglogistic.invsf(p, beta, scale)
        # Expected value computed with Wolfram Alpha:
        #   InverseCDF[LogLogisticDistribution[3, 1/4], 3/4]
        expected_x = mpmath.cbrt(3)/4
        assert mpmath.almosteq(x, expected_x)


def test_mean():
    with mpmath.workdps(50):
        beta = mpmath.mpf(3)
        scale = mpmath.mpf(0.25)
        mean = loglogistic.mean(beta, scale)
        # Expected value computed with Wolfram Alpha:
        #   Mean[LogLogisticDistribution[3, 1/4]]
        expected_mean = mpmath.pi/(6*mpmath.sqrt(3))
        assert mpmath.almosteq(mean, expected_mean)


def test_var():
    with mpmath.workdps(50):
        beta = mpmath.mpf(3)
        scale = mpmath.mpf(0.25)
        var = loglogistic.var(beta, scale)
        # Expected value computed with Wolfram Alpha:
        #   Variance[LogLogisticDistribution[3, 1/4]]
        expected_var = (mpmath.sqrt(3) - mpmath.pi/3)*mpmath.pi/36
        assert mpmath.almosteq(var, expected_var)
