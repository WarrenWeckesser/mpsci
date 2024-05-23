
import pytest
from mpmath import mp
from mpsci.distributions import benktander2


@pytest.mark.parametrize('a', [-1, 0])
def test_validate_a_positive(a):
    with pytest.raises(ValueError, match='must be positive'):
        benktander2._validate_ab(a, 1)


@pytest.mark.parametrize('b', [-1, 0, 1.25])
def test_validate_b_in_unit_interval(b):
    with pytest.raises(ValueError, match='must be in the interval'):
        benktander2._validate_ab(5, b)


@mp.workdps(25)
def test_support():
    # We haven't included a location parameter in the implementation,
    # so the support is always [1, inf).
    s = benktander2.support(2, 0.25)
    assert s == (mp.one, mp.inf)


@mp.workdps(25)
def test_pdf_outside_support():
    p = benktander2.pdf(0.5, 5, 0.5)
    assert p == mp.zero


@mp.workdps(25)
def test_logpdf_outside_support():
    p = benktander2.logpdf(0.5, 5, 0.75)
    assert p == mp.ninf


@mp.workdps(50)
def test_pdf():
    x = mp.mpf('1.5')
    a = 2
    b = mp.mpf('0.75')
    p = benktander2.pdf(x, a, b)
    # Expected value computed with Wolfram Alpha:
    #    PDF[BenktanderWeibullDistribution[2, 3/4], 3/2]
    valstr = '0.6913485277470671248347955016586714957100372820095067311673'
    expected = mp.mpf(valstr)
    assert mp.almosteq(p, expected)


@mp.workdps(50)
def test_logpdf():
    x = mp.mpf('1.5')
    a = 2
    b = mp.mpf('0.75')
    p = benktander2.logpdf(x, a, b)
    # Expected value computed with Wolfram Alpha:
    #    log(PDF[BenktanderWeibullDistribution[2, 3/4], 3/2])
    valstr = '-0.369111200683201817345059066451331575064497370791420217397'
    expected = mp.mpf(valstr)
    assert mp.almosteq(p, expected)


@mp.workdps(25)
def test_cdf_outside_support():
    p = benktander2.cdf(0.5, 5, 0.125)
    assert p == mp.zero


@mp.workdps(50)
def test_cdf_invcdf():
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


@mp.workdps(50)
def test_cdf_precision():
    x = mp.mpf('10000001/10000000')
    a = mp.mpf(0.5)
    b = mp.mpf('9999/10000')
    p = benktander2.cdf(x, a, b)
    # Expected value computed with Wolfram Alpha:
    #    CDF[BenktanderWeibullDistribution[1/2, 9999/10000], 1000001/1000000]
    valstr = '5.0009998748749970925010566936263405843182136875870792461e-8'
    expected = mp.mpf(valstr)
    assert mp.almosteq(p, expected)


@mp.workdps(50)
def test_invcdf_b1():
    p = mp.mpf('0.001')
    a = 4
    b = 1
    x = benktander2.invcdf(p, a, b)
    valstr = '1.00025012508339588337503574556351708624018880131260860231'
    expected = mp.mpf(valstr)
    assert mp.almosteq(x, expected)


@mp.workdps(25)
def test_sf_outside_support():
    p = benktander2.sf(0.5, 5, 0.975)
    assert p == mp.one


@mp.workdps(50)
def test_sf_invsf():
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


@mp.workdps(50)
def test_invsf_b1():
    p = mp.mpf('0.999')
    a = 4
    b = 1
    x = benktander2.invsf(p, a, b)
    valstr = '1.00025012508339588337503574556351708624018880131260860231'
    expected = mp.mpf(valstr)
    assert mp.almosteq(x, expected)


@pytest.mark.parametrize('p, expected', [(0, 1), (1, 'inf')])
@mp.workdps(50)
def test_invcdf_invsf_bounds(p, expected):
    x = benktander2.invcdf(p, 2, 0.25)
    assert x == mp.mpf(expected)
    x = benktander2.invsf(1 - p, 2, 0.25)
    assert x == mp.mpf(expected)


@mp.workdps(50)
def test_mean():
    a = 2
    b = mp.mpf('0.25')
    m = benktander2.mean(a, b)
    assert mp.almosteq(m, mp.mpf('1.5'))


@mp.workdps(50)
def test_var():
    a = 2
    b = mp.mpf('0.25')
    m = benktander2.var(a, b)
    # Expected value computed with Wolfram Alpha:
    #    Var[BenktanderWeibullDistribution[2, 1/4]]
    expected = mp.mpf('251/512')
    assert mp.almosteq(m, expected)
