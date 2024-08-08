from mpmath import mp
from mpsci.stats import pearsonr


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
