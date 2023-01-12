
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
