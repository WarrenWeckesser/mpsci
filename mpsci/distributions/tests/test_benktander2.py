
import mpmath
from mpsci.distributions import benktander2


def test_pdf():
    with mpmath.workdps(50):
        x = mpmath.mpf('1.5')
        a = 2
        b = mpmath.mpf('0.75')
        p = benktander2.pdf(x, a, b)
        # Expected value computed with Wolfram Alpha:
        #    PDF[BenktanderWeibullDistribution[2, 3/4], 3/2]
        valstr = '0.6913485277470671248347955016586714957100372820095067311673'
        expected = mpmath.mpf(valstr)
        assert mpmath.almosteq(p, expected)


def test_logpdf():
    with mpmath.workdps(50):
        x = mpmath.mpf('1.5')
        a = 2
        b = mpmath.mpf('0.75')
        p = benktander2.logpdf(x, a, b)
        # Expected value computed with Wolfram Alpha:
        #    log(PDF[BenktanderWeibullDistribution[2, 3/4], 3/2])
        valstr = '-0.369111200683201817345059066451331575064497370791420217397'
        expected = mpmath.mpf(valstr)
        assert mpmath.almosteq(p, expected)


def test_cdf_invcdf():
    with mpmath.workdps(50):
        x = mpmath.mpf('1.5')
        a = 2
        b = mpmath.mpf('0.75')
        p = benktander2.cdf(x, a, b)
        # Expected value computed with Wolfram Alpha:
        #    CDF[BenktanderWeibullDistribution[2, 3/4], 3/2]
        valstr = '0.6497498357448767493508965922137966312235765166018499603595'
        expected = mpmath.mpf(valstr)
        assert mpmath.almosteq(p, expected)
        x1 = benktander2.invcdf(expected, a, b)
        assert mpmath.almosteq(x1, x)


def test_invcdf_b1():
    with mpmath.workdps(50):
        p = mpmath.mpf('0.001')
        a = 4
        b = 1
        x = benktander2.invcdf(p, a, b)
        valstr = '1.00025012508339588337503574556351708624018880131260860231'
        expected = mpmath.mpf(valstr)
        assert mpmath.almosteq(x, expected)


def test_sf_invsf():
    with mpmath.workdps(50):
        x = mpmath.mpf('1.5')
        a = 2
        b = mpmath.mpf('0.75')
        p = benktander2.sf(x, a, b)
        # Expected value computed with Wolfram Alpha:
        #    SurvivalFunction[BenktanderWeibullDistribution[2, 3/4], 3/2]
        valstr = '0.3502501642551232506491034077862033687764234833981500396405'
        expected = mpmath.mpf(valstr)
        assert mpmath.almosteq(p, expected)
        x1 = benktander2.invsf(expected, a, b)
        assert mpmath.almosteq(x1, x)


def test_invsf_b1():
    with mpmath.workdps(50):
        p = mpmath.mpf('0.999')
        a = 4
        b = 1
        x = benktander2.invsf(p, a, b)
        valstr = '1.00025012508339588337503574556351708624018880131260860231'
        expected = mpmath.mpf(valstr)
        assert mpmath.almosteq(x, expected)


def test_mean():
    with mpmath.workdps(50):
        a = 2
        b = mpmath.mpf('0.25')
        m = benktander2.mean(a, b)
        assert mpmath.almosteq(m, mpmath.mpf('1.5'))


def test_var():
    with mpmath.workdps(50):
        a = 2
        b = mpmath.mpf('0.25')
        m = benktander2.var(a, b)
        # Expected value computed with Wolfram Alpha:
        #    Var[BenktanderWeibullDistribution[2, 1/4]]
        expected = mpmath.mpf('251/512')
        assert mpmath.almosteq(m, expected)
