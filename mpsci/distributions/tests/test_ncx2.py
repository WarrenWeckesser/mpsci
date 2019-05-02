
import mpmath
from mpsci.distributions import ncx2


def test_basic_pdf():
    with mpmath.workdps(50):

        # Precomputed results using Wolfram
        #   PDF[NoncentralChiSquareDistribution[k, lam], x}

        expected = mpmath.mpf('0.12876542477554595251711369063259389422017597538321')
        assert mpmath.almosteq(ncx2.pdf(1, 2, 3), expected)

        expected = mpmath.mpf('6.91617141220678829079562729724319046638618761057660e-17')
        assert mpmath.almosteq(ncx2.pdf(100, 2, 3), expected)


def test_basic_cdf_sf():
    with mpmath.workdps(50):

        # Precomputed results using Wolfram
        #   CDF{NoncentralChiSquareDistribution[k, lam], x}

        expected = mpmath.mpf('0.88747672169682687250365246005909913605814777942758')
        assert mpmath.almosteq(ncx2.cdf(10, 2, 3), expected)
        assert mpmath.almosteq(ncx2.sf(10, 2, 3), 1 - expected)

        k = mpmath.mpf('200')
        lam = mpmath.mpf('0.03')
        c = ncx2.cdf(1, k, lam)
        expected = mpmath.mpf('5.07601124317862768018428556288096760842982491912529e-189')
        assert mpmath.almosteq(c, expected)


def test_basic_mean_variance():
    with mpmath.workdps(50):
        assert ncx2.mean(2, 3) == 5
        assert ncx2.variance(2, 3) == 16
