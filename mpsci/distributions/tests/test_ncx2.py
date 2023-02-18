
from mpmath import mp
from mpsci.distributions import ncx2


@mp.workdps(50)
def test_basic_pdf():

    # Precomputed results using Wolfram
    #   PDF[NoncentralChiSquareDistribution[k, lam], x}

    expected = mp.mpf('0.12876542477554595251711369063259389422017597538321')
    assert mp.almosteq(ncx2.pdf(1, 2, 3), expected)

    expected = mp.mpf('6.91617141220678829079562729724319046638618761057660e-17')
    assert mp.almosteq(ncx2.pdf(100, 2, 3), expected)


@mp.workdps(50)
def test_basic_cdf_sf():

    # Precomputed results using Wolfram
    #   CDF{NoncentralChiSquareDistribution[k, lam], x}

    expected = mp.mpf('0.88747672169682687250365246005909913605814777942758')
    assert mp.almosteq(ncx2.cdf(10, 2, 3), expected)
    assert mp.almosteq(ncx2.sf(10, 2, 3), 1 - expected)

    k = mp.mpf('200')
    lam = mp.mpf('0.03')
    c = ncx2.cdf(1, k, lam)
    expected = mp.mpf('5.07601124317862768018428556288096760842982491912529e-189')
    assert mp.almosteq(c, expected)


@mp.workdps(50)
def test_basic_mean_variance():
    assert ncx2.mean(2, 3) == 5
    assert ncx2.variance(2, 3) == 16
