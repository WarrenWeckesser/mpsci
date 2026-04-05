from mpmath import mp
from mpsci.stats import pearsonr, pearsonr_ci


def test_pearsonr():
    x = [1, 2, 3, 5]
    y = [2, 2, 4, 7]
    r, p = pearsonr(x, y)
    # The reference values were computed with Wolfram Alpha:
    #    Correlation[(1, 2, 3, 5), (2, 2, 4, 7)]
    # Alpha reports only 6 digits, and does not provide the usual
    # "more digits" option that other Alpha computations provide.
    assert mp.almosteq(r, 0.970569, rel_eps=5e-6, abs_eps=0)
    assert mp.almosteq(p, 0.029431, rel_eps=5e-5, abs_eps=0)


def test_sample_length_2():
    x = [1.0, 2.0]
    y = [0.0, -1.0]
    r, p = pearsonr(x, y)
    assert r == -1
    assert p == 1


def test_pearsonr_ci():
    x = [0, 1, 3, 6, 10, 15, 21, 28, 36, 45]
    y = [0.0, 0.5625, 2.0625, 5.25, 11.25, 21.5625, 38.0625, 63.0,  99.0, 149.0625]
    r, _p = pearsonr(x, y)
    rlo, rhi = pearsonr_ci(r, len(x), alpha=0.05)
    # Reference values were checked with several independent sources.  They
    # were not high precision, so we check only about 8 significant digits.
    assert mp.almosteq(rlo, 0.87782892, rel_eps=5e-8, abs_eps=0)
    assert mp.almosteq(rhi, 0.99330131, rel_eps=5e-8, abs_eps=0)
