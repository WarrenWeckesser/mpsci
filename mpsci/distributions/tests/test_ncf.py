
from mpmath import mp
from mpsci.distributions import ncf


def test_basic_pdf():
    with mp.workdps(50):

        # Precomputed results using Wolfram
        #   PDF[NoncentralFRatioDistribution[dfn, dfd, c], x]

        expected = mp.mpf('0.64212289998387725947981190753422224'
                          '030612639498189')
        assert mp.almosteq(ncf.pdf(1, 10, 12, 1/2), expected)

        expected = mp.mpf('0.00228946570447734902588902693024340'
                          '482823240415468206756')
        assert mp.almosteq(ncf.pdf(16, 2, 3, mp.mpf('1/10')),
                           expected)


def test_basic_cdf():
    with mp.workdps(50):

        # Precomputed results using Wolfram
        #   CDF[NoncentralFRatioDistribution[dfn, dfd, c], x]

        expected = mp.mpf('0.98736941909854022676388661536551212'
                          '857167482965985997')
        assert mp.almosteq(ncf.cdf(9, 4, 6, mp.mpf('1/3')), expected)

        expected = mp.mpf('6.24122219569310884295091344333840142'
                          '52693682110804522e-9')
        assert mp.almosteq(ncf.cdf(mp.mpf('1/100'), 10, 16, mp.mpf('1/4')),
                           expected)


def test_basic_mean():
    with mp.workdps(50):
        assert mp.almosteq(ncf.mean(10, 16, mp.mpf('1/4')),
                           mp.mpf('41/35'))


def test_basic_var():
    with mp.workdps(50):
        assert mp.almosteq(ncf.var(10, 16, mp.mpf('1/4')),
                           mp.mpf('4033/7350'))
