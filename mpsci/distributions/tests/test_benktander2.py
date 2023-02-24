
import pytest
from mpmath import mp
from mpsci.distributions import benktander2


def test_pdf():
    with mp.workdps(50):
        x = mp.mpf('1.5')
        a = 2
        b = mp.mpf('0.75')
        p = benktander2.pdf(x, a, b)
        # Expected value computed with Wolfram Alpha:
        #    PDF[BenktanderWeibullDistribution[2, 3/4], 3/2]
        valstr = '0.6913485277470671248347955016586714957100372820095067311673'
        expected = mp.mpf(valstr)
        assert mp.almosteq(p, expected)


def test_logpdf():
    with mp.workdps(50):
        x = mp.mpf('1.5')
        a = 2
        b = mp.mpf('0.75')
        p = benktander2.logpdf(x, a, b)
        # Expected value computed with Wolfram Alpha:
        #    log(PDF[BenktanderWeibullDistribution[2, 3/4], 3/2])
        valstr = '-0.369111200683201817345059066451331575064497370791420217397'
        expected = mp.mpf(valstr)
        assert mp.almosteq(p, expected)


def test_cdf_invcdf():
    with mp.workdps(50):
        x = mp.mpf('1.5')
        a = 2
        b = mp.mpf('0.75')
        p = benktander2.cdf(x, a, b)
        # Expected value computed with Wolfram Alpha:
        #    CDF[BenktanderWeibullDistribution[2, 3/4], 3/2]
        valstr = '0.6497498357448767493508965922137966312235765166018499603595'
        expected = mp.mpf(valstr)
        assert mp.almosteq(p, expected)
        x1 = benktander2.invcdf(expected, a, b)
        assert mp.almosteq(x1, x)


def test_invcdf_b1():
    with mp.workdps(50):
        p = mp.mpf('0.001')
        a = 4
        b = 1
        x = benktander2.invcdf(p, a, b)
        valstr = '1.00025012508339588337503574556351708624018880131260860231'
        expected = mp.mpf(valstr)
        assert mp.almosteq(x, expected)


def test_sf_invsf():
    with mp.workdps(50):
        x = mp.mpf('1.5')
        a = 2
        b = mp.mpf('0.75')
        p = benktander2.sf(x, a, b)
        # Expected value computed with Wolfram Alpha:
        #    SurvivalFunction[BenktanderWeibullDistribution[2, 3/4], 3/2]
        valstr = '0.3502501642551232506491034077862033687764234833981500396405'
        expected = mp.mpf(valstr)
        assert mp.almosteq(p, expected)
        x1 = benktander2.invsf(expected, a, b)
        assert mp.almosteq(x1, x)


def test_invsf_b1():
    with mp.workdps(50):
        p = mp.mpf('0.999')
        a = 4
        b = 1
        x = benktander2.invsf(p, a, b)
        valstr = '1.00025012508339588337503574556351708624018880131260860231'
        expected = mp.mpf(valstr)
        assert mp.almosteq(x, expected)


@pytest.mark.parametrize('p, expected', [(0, 1), (1, 'inf')])
def test_invcdf_invsf_bounds(p, expected):
    with mp.workdps(50):
        x = benktander2.invcdf(p, 2, 0.25)
        assert x == mp.mpf(expected)
        x = benktander2.invsf(1 - p, 2, 0.25)
        assert x == mp.mpf(expected)


def test_mean():
    with mp.workdps(50):
        a = 2
        b = mp.mpf('0.25')
        m = benktander2.mean(a, b)
        assert mp.almosteq(m, mp.mpf('1.5'))


def test_var():
    with mp.workdps(50):
        a = 2
        b = mp.mpf('0.25')
        m = benktander2.var(a, b)
        # Expected value computed with Wolfram Alpha:
        #    Var[BenktanderWeibullDistribution[2, 1/4]]
        expected = mp.mpf('251/512')
        assert mp.almosteq(m, expected)
