
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


def test_cdf():
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


def test_sf():
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
