
from mpmath import mp
from mpsci.distributions import loglogistic


def test_pdf():
    with mp.workdps(50):
        x = mp.mpf(1.5)
        beta = mp.mpf(3)
        scale = mp.mpf(0.25)
        pdf = loglogistic.pdf(x, beta, scale)
        # Expected value computed with Wolfram Alpha:
        #   PDF[LogLogisticDistribution[3, 1/4], 3/2]
        expected_pdf = mp.mpf('432/47089')
        assert mp.almosteq(pdf, expected_pdf)


def test_logpdf():
    with mp.workdps(50):
        x = mp.mpf(1.5)
        beta = mp.mpf(3)
        scale = mp.mpf(0.25)
        logpdf = loglogistic.logpdf(x, beta, scale)
        # Expected value computed with Wolfram Alpha:
        #   PDF[LogLogisticDistribution[3, 1/4], 3/2]
        expected_logpdf = mp.log(mp.mpf('432/47089'))
        assert mp.almosteq(logpdf, expected_logpdf)


def test_cdf():
    with mp.workdps(50):
        x = mp.mpf(1.5)
        beta = mp.mpf(3)
        scale = mp.mpf(0.25)
        cdf = loglogistic.cdf(x, beta, scale)
        # Expected value computed with Wolfram Alpha:
        #   CDF[LogLogisticDistribution[3, 1/4], 3/2]
        expected_cdf = mp.mpf('216/217')
        assert mp.almosteq(cdf, expected_cdf)


def test_sf():
    with mp.workdps(50):
        x = mp.mpf(1.5)
        beta = mp.mpf(3)
        scale = mp.mpf(0.25)
        sf = loglogistic.sf(x, beta, scale)
        # Expected value computed with Wolfram Alpha:
        #   CDF[LogLogisticDistribution[3, 1/4], 3/2]
        expected_sf = mp.mpf('1/217')
        assert mp.almosteq(sf, expected_sf)


def test_invcdf():
    with mp.workdps(50):
        p = mp.mpf(0.75)
        beta = mp.mpf(3)
        scale = mp.mpf(0.25)
        x = loglogistic.invcdf(p, beta, scale)
        # Expected value computed with Wolfram Alpha:
        #   InverseCDF[LogLogisticDistribution[3, 1/4], 3/4]
        expected_x = mp.cbrt(3)/4
        assert mp.almosteq(x, expected_x)


def test_invsf():
    with mp.workdps(50):
        p = mp.mpf(0.25)
        beta = mp.mpf(3)
        scale = mp.mpf(0.25)
        x = loglogistic.invsf(p, beta, scale)
        # Expected value computed with Wolfram Alpha:
        #   InverseCDF[LogLogisticDistribution[3, 1/4], 3/4]
        expected_x = mp.cbrt(3)/4
        assert mp.almosteq(x, expected_x)


def test_mean():
    with mp.workdps(50):
        beta = mp.mpf(3)
        scale = mp.mpf(0.25)
        mean = loglogistic.mean(beta, scale)
        # Expected value computed with Wolfram Alpha:
        #   Mean[LogLogisticDistribution[3, 1/4]]
        expected_mean = mp.pi/(6*mp.sqrt(3))
        assert mp.almosteq(mean, expected_mean)


def test_var():
    with mp.workdps(50):
        beta = mp.mpf(3)
        scale = mp.mpf(0.25)
        var = loglogistic.var(beta, scale)
        # Expected value computed with Wolfram Alpha:
        #   Variance[LogLogisticDistribution[3, 1/4]]
        expected_var = (mp.sqrt(3) - mp.pi/3)*mp.pi/36
        assert mp.almosteq(var, expected_var)
