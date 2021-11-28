
import mpmath
from mpsci.distributions import truncnorm, normal


def test_pdf():
    a = 3
    b = 5
    with mpmath.workdps(50):
        x = 4
        p = truncnorm.pdf(x, a, b)
        D = normal.cdf(b) - normal.cdf(a)
        assert mpmath.almosteq(p, normal.pdf(x)/D,
                               rel_eps=mpmath.mpf('1e-48'), abs_eps=0)


def test_cdf():
    a = 3
    b = 5
    with mpmath.workdps(50):
        x = 4
        p = truncnorm.cdf(x, a, b)
        N = normal.cdf(x) - normal.cdf(a)
        D = normal.cdf(b) - normal.cdf(a)
        assert mpmath.almosteq(p, N/D, rel_eps=mpmath.mpf('1e-48'), abs_eps=0)


def test_mean_with_quad():
    a = 1
    b = 3
    with mpmath.workdps(50):
        mean = truncnorm.mean(a, b)
        meanq = mpmath.quad(lambda x: x*truncnorm.pdf(x, a, b), [a, b])
        assert mpmath.almosteq(mean, meanq)


def test_var_with_quad():
    a = 1
    b = 4
    with mpmath.workdps(50):
        var = truncnorm.var(a, b)
        mean = truncnorm.mean(a, b)
        varq = mpmath.quad(lambda x: (x - mean)**2*truncnorm.pdf(x, a, b),
                           [a, b])
        assert mpmath.almosteq(var, varq)


def test_halfnormal_mean():
    with mpmath.workdps(50):
        meanr = truncnorm.mean(0, mpmath.inf)
        assert mpmath.almosteq(meanr, mpmath.sqrt(2/mpmath.pi))
        meanl = truncnorm.mean(-mpmath.inf, 0)
        assert mpmath.almosteq(meanl, -mpmath.sqrt(2/mpmath.pi))


def test_halfnormal_var():
    with mpmath.workdps(50):
        varr = truncnorm.var(0, mpmath.inf)
        assert mpmath.almosteq(varr, 1 - 2/mpmath.pi)
        varl = truncnorm.var(-mpmath.inf, 0)
        assert mpmath.almosteq(varl, 1 - 2/mpmath.pi)


def test_halfnormal_median():
    with mpmath.workdps(50):
        medianr = truncnorm.median(0, mpmath.inf)
        from_formula = mpmath.sqrt(2)*mpmath.erfinv(mpmath.mpf('0.5'))
        assert mpmath.almosteq(medianr, from_formula)
        medianl = truncnorm.median(-mpmath.inf, 0)
        assert mpmath.almosteq(medianl, -from_formula)


def test_symmetric():
    b = 2
    with mpmath.workdps(50):
        mean = truncnorm.mean(-b, b)
        assert mean == 0
        median = truncnorm.median(-b, b)
        assert median == 0
