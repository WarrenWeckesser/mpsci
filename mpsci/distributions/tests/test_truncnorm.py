
from mpmath import mp
from mpsci.distributions import truncnorm, normal


def test_pdf():
    a = 3
    b = 5
    with mp.workdps(50):
        x = 4
        p = truncnorm.pdf(x, a, b)
        D = normal.cdf(b) - normal.cdf(a)
        assert mp.almosteq(p, normal.pdf(x)/D,
                           rel_eps=mp.mpf('1e-48'), abs_eps=0)


def test_cdf():
    a = 3
    b = 5
    with mp.workdps(50):
        x = 4
        p = truncnorm.cdf(x, a, b)
        N = normal.cdf(x) - normal.cdf(a)
        D = normal.cdf(b) - normal.cdf(a)
        diff_ratio = N/D
        assert mp.almosteq(p, diff_ratio, rel_eps=mp.mpf('1e-48'), abs_eps=0)


def test_sf():
    a = 3
    b = 5
    with mp.workdps(50):
        x = 3.5
        p = truncnorm.sf(x, a, b)
        N = normal.cdf(b) - normal.cdf(x)
        D = normal.cdf(b) - normal.cdf(a)
        diff_ratio = N/D
        assert mp.almosteq(p, diff_ratio, rel_eps=mp.mpf('1e-47'), abs_eps=0)


def test_mean_with_quad():
    a = 1
    b = 3
    with mp.workdps(50):
        mean = truncnorm.mean(a, b)
        meanq = mp.quad(lambda x: x*truncnorm.pdf(x, a, b), [a, b])
        assert mp.almosteq(mean, meanq)


def test_var_with_quad():
    a = 1
    b = 4
    with mp.workdps(50):
        var = truncnorm.var(a, b)
        mean = truncnorm.mean(a, b)
        varq = mp.quad(lambda x: (x - mean)**2*truncnorm.pdf(x, a, b), [a, b])
        assert mp.almosteq(var, varq)


def test_halfnormal_mean():
    with mp.workdps(50):
        meanr = truncnorm.mean(0, mp.inf)
        assert mp.almosteq(meanr, mp.sqrt(2/mp.pi))
        meanl = truncnorm.mean(mp.ninf, 0)
        assert mp.almosteq(meanl, -mp.sqrt(2/mp.pi))


def test_halfnormal_var():
    with mp.workdps(50):
        varr = truncnorm.var(0, mp.inf)
        assert mp.almosteq(varr, 1 - 2/mp.pi)
        varl = truncnorm.var(mp.ninf, 0)
        assert mp.almosteq(varl, 1 - 2/mp.pi)


def test_halfnormal_median():
    with mp.workdps(50):
        medianr = truncnorm.median(0, mp.inf)
        from_formula = mp.sqrt(2)*mp.erfinv(mp.mpf('0.5'))
        assert mp.almosteq(medianr, from_formula)
        medianl = truncnorm.median(mp.ninf, 0)
        assert mp.almosteq(medianl, -from_formula)


def test_symmetric():
    b = 2
    with mp.workdps(50):
        mean = truncnorm.mean(-b, b)
        assert mean == 0
        median = truncnorm.median(-b, b)
        assert median == 0


def test_entropy_against_normal():
    with mp.workdps(50):
        e = truncnorm.entropy(-mp.inf, mp.inf)
        assert mp.almosteq(e, (mp.log(2*mp.pi) + 1)/2)


def test_entropy_with_quad():
    a = -0.5
    b = 2.5
    with mp.workdps(50):
        entr = truncnorm.entropy(a, b)
        q = mp.quad(lambda t: (-truncnorm.pdf(t, a, b) *
                               truncnorm.logpdf(t, a, b)),
                    [a, b])
        assert mp.almosteq(entr, q)
