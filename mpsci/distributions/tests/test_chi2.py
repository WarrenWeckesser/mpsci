
import mpmath
from mpsci.distributions import chi2


def test_basic_pdf():
    with mpmath.workdps(50):

        # Precomputed results using Wolfram
        #   PDF[ChiSquareDistribution[k], x]

        valstr = '0.15418032980376927681122055778139364630515950769552'
        expected = mpmath.mpf(valstr)
        assert mpmath.almosteq(chi2.pdf(3, 5), expected)

        valstr = '0.073224912809632435566001478191006253712616408469863'
        expected = mpmath.mpf(valstr)
        assert mpmath.almosteq(chi2.pdf(5, 3), expected)

        valstr = '8.2025293495250421352683463106457295660525245864016e-7'
        expected = mpmath.mpf(valstr)
        assert mpmath.almosteq(chi2.pdf(50, 25/2), expected)


def test_basic_logpdf():
    with mpmath.workdps(50):

        # Precomputed results using Wolfram
        #   log(PDF[ChiSquareDistribution[k], x])

        valstr = '-1.8696323888706178960827071179443547875376521947264'
        expected = mpmath.mpf(valstr)
        assert mpmath.almosteq(chi2.logpdf(3, 5), expected)

        valstr = '-109.00728367462314733374258625716397761840084187637237'
        expected = mpmath.mpf(valstr)
        k = mpmath.mpf('31')/3
        assert mpmath.almosteq(chi2.logpdf(250, k), expected)


def test_logpdf_negx():
    with mpmath.workdps(50):
        assert chi2.logpdf(-3, 4) == mpmath.ninf


def test_basic_cdf():
    with mpmath.workdps(50):

        # Precomputed results using Wolfram
        #   CDF[ChiSquareDistribution[k], x]

        valstr = '0.82820285570326686493639334781694850021090176319403'
        expected = mpmath.mpf(valstr)
        assert mpmath.almosteq(chi2.cdf(5, 3), expected)

        valstr = '0.99999794808843309198843919723813688969884118187012'
        expected = mpmath.mpf(valstr)
        assert mpmath.almosteq(chi2.cdf(50, 25/2), expected)


def test_basic_sf():
    with mpmath.workdps(50):

        # Precomputed results using Wolfram
        #   SurvivalFunction[ChiSquareDistribution[k], x]

        valstr = '0.17179714429673313506360665218305149978909823680597'
        expected = mpmath.mpf(valstr)
        assert mpmath.almosteq(chi2.sf(5, 3), expected)

        valstr = '2.05191156690801156080276186311030115881812988312938e-6'
        expected = mpmath.mpf(valstr)
        assert mpmath.almosteq(chi2.sf(50, 25/2), expected)


def test_basic_mean_variance():
    with mpmath.workdps(50):
        assert chi2.mean(5) == 5
        assert chi2.variance(5) == 10


def test_mode():
    with mpmath.workdps(50):
        assert chi2.mode(5) == 3
        assert chi2.mode(1.5) == 0
