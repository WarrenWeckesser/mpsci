
import mpmath
from mpsci.distributions import ncf


def test_basic_pdf():
    with mpmath.workdps(50):

        # Precomputed results using Wolfram
        #   PDF[NoncentralFRatioDistribution[dfn, dfd, c], x]

        expected = mpmath.mpf('0.64212289998387725947981190753422224'
                              '030612639498189')
        assert mpmath.almosteq(ncf.pdf(1, 10, 12, 1/2), expected)

        expected = mpmath.mpf('0.00228946570447734902588902693024340'
                              '482823240415468206756')
        assert mpmath.almosteq(ncf.pdf(16, 2, 3, mpmath.mpf('1/10')),
                               expected)


def test_basic_cdf():
    with mpmath.workdps(50):

        # Precomputed results using Wolfram
        #   CDF[NoncentralFRatioDistribution[dfn, dfd, c], x]

        expected = mpmath.mpf('0.98736941909854022676388661536551212'
                              '857167482965985997')
        assert mpmath.almosteq(ncf.cdf(9, 4, 6, mpmath.mpf('1/3')), expected)

        expected = mpmath.mpf('6.24122219569310884295091344333840142'
                              '52693682110804522e-9')
        assert mpmath.almosteq(ncf.cdf(mpmath.mpf('1/100'), 10, 16,
                                       mpmath.mpf('1/4')),
                               expected)


def test_basic_mean():
    with mpmath.workdps(50):
        assert mpmath.almosteq(ncf.mean(10, 16, mpmath.mpf('1/4')),
                               mpmath.mpf('41/35'))


def test_basic_var():
    with mpmath.workdps(50):
        assert mpmath.almosteq(ncf.var(10, 16, mpmath.mpf('1/4')),
                               mpmath.mpf('4033/7350'))
