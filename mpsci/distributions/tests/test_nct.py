import pytest
from mpmath import mp
from mpsci.distributions import nct


def test_basic_pdf():
    with mp.workdps(50):
        pdf = nct.pdf(3, 7, 0.5)
        # Wolfram Alpha:
        #   PDF[NoncentralStudentTDistribution[df, noncentrality], x]
        expected = mp.mpf('0.03603206821366978436669265469835680'
                          '31801809406337466')
        assert mp.almosteq(pdf, expected)


def test_basic_logpdf():
    with mp.workdps(50):
        logpdf = nct.logpdf(3, 7, 0.5)
        # Wolfram Alpha:
        #   ln[PDF[NoncentralStudentTDistribution[df, noncentrality], x]]
        expected = mp.mpf('-3.3233459533253355060314798326799414479'
                          '85315876425721246')
        assert mp.almosteq(logpdf, expected)


def test_basic_mean():
    with mp.workdps(50):
        m = nct.mean(7, 0.5)
        # Wolfram Alpha:
        #   Mean[NoncentralStudentTDistribution[7, 1/2]]
        assert mp.almosteq(m, 4*mp.sqrt(14/mp.pi)/15)


def test_basic_var():
    with mp.workdps(50):
        v = nct.var(7, 0.5)
        # Wolfram Alpha:
        #   Variance[NoncentralStudentTDistribution[7, 1/2]]
        expected = mp.mpf(7)/4 - 224/(225*mp.pi)
        assert mp.almosteq(v, expected)


# The exected values in the following were computed by numerical integration,
# e.g. for n=3, df=4, mu=1:
#
# >>> mp.dps = 60
# >>> mp.quad(lambda t: t**3*nct.pdf(t, 4, 1.0), [-mp.inf, 1, mp.inf])
# mpf('20.0530261970480040193261222784883620240558939248795065330393877')
#
# For this numerical integration, mp.dps generally needs to be at least
# twice as large as the actual desired precision of the result.
# Because the implementation of the pdf function is currently slow, the
# numerical integration to compute the noncentral moment is *extremely*
# slow, so the expected values have been precomputed.
#
@pytest.mark.parametrize('n, df, nc, val',
                         [(0, 8.0, 7.5, '1'),
                          (1, 4.0, 1.0, '1.253314137315500251207882642'),
                          (2, 9.0, 0.125, '1.305803571428571428571428571'),
                          (3, 3.5, 2.0, '127.8560034015895779185670077'),
                          (3, 4.0, 1.0, '20.05302619704800401932612228'),
                          (6, 9.0, 0.125, '109.050005558558872767857143')])
def test_noncentral_moment(n, df, nc, val):
    with mp.workdps(25):
        m = nct.noncentral_moment(n, df, nc)
        assert mp.almosteq(m, mp.mpf(val))
