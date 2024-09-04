
from itertools import product
import pytest
from mpmath import mp
from mpsci.distributions import nakagami


# Reference values are computed with Wolfram Alpha.
# Given x, nu and scale (loc=0), the PDF is
#     PDF[NakagamiDistribution[nu, scale**2], x]
#
@pytest.mark.parametrize(
    'x, nu, scale, ref',
    [(4.0, 3.0, 3.0,
      '0.18310447384012572818876579342477436028692518158528'),
     ('0.001', 2.0, 1.0,
      '7.9999840000159999893333386666645333340444442412698920e-9')]
)
@mp.workdps(48)
def test_pdf(x, nu, scale, ref):
    x = mp.mpf(x)
    nu = mp.mpf(nu)
    scale = mp.mpf(scale)
    ref = mp.mpf(ref)
    p = nakagami.pdf(x, nu, loc=0, scale=scale)
    assert mp.almosteq(p, ref)


# Reference values are computed with Wolfram Alpha.
# Given x, nu and scale (loc=0), the PDF is
#     Log[PDF[NakagamiDistribution[nu, scale**2], x]]
#
@pytest.mark.parametrize(
    'x, nu, scale, ref',
    [(4.0, 3.0, 3.0,
      '-1.6976983937382093133467478295191447665208036631990'),
     ('0.001', 2.0, 1.0,
      '-18.64382629526657522791022672778474816418341299457819')]
)
@mp.workdps(48)
def test_logpdf(x, nu, scale, ref):
    x = mp.mpf(x)
    nu = mp.mpf(nu)
    scale = mp.mpf(scale)
    ref = mp.mpf(ref)
    p = nakagami.logpdf(x, nu, loc=0, scale=scale)
    assert mp.almosteq(p, ref)


# Reference values are computed with Wolfram Alpha.
# Given x, nu and scale (loc=0), the CDF is
#     CDF[NakagamiDistribution[nu, scale**2], x]
#
@pytest.mark.parametrize(
    'x, nu, scale, ref',
    [(4.0, 3.0, 3.0,
      '0.90075880568235373130394041469653342777417629318376'),
     ('0.001', 2.0, 1.0,
      '1.99999733333533333226666711111095873020317459188712776e-12')]
)
@mp.workdps(48)
def test_cdf(x, nu, scale, ref):
    x = mp.mpf(x)
    nu = mp.mpf(nu)
    scale = mp.mpf(scale)
    ref = mp.mpf(ref)
    p = nakagami.cdf(x, nu, loc=0, scale=scale)
    assert mp.almosteq(p, ref)


# Reference values are computed with Wolfram Alpha.
# Given x, nu and scale (loc=0), the SF is
#     1 - CDF[NakagamiDistribution[nu, scale**2], x]
#
@pytest.mark.parametrize(
    'x, nu, scale, ref',
    [(4.0, 3.0, 3.0,
      '0.099241194317646268696059585303466572225823706816243'),
     ('0.001', 2.0, 1.0,
      '0.99999999999800000266666466666773333288888904126980')]
)
@mp.workdps(48)
def test_sf(x, nu, scale, ref):
    x = mp.mpf(x)
    nu = mp.mpf(nu)
    scale = mp.mpf(scale)
    ref = mp.mpf(ref)
    p = nakagami.sf(x, nu, loc=0, scale=scale)
    assert mp.almosteq(p, ref)


# Reference values are computed with Wolfram Alpha.
# Given nu and scale (loc=0), the mean is
#     Mean[NakagamiDistribution[nu, scale**2]]
#
@pytest.mark.parametrize(
    'nu, scale, ref',
    [(3.0, 3.0,
      '2.8781063660994988738487389450010574207394962311746'),
     (2.0, 1.0,
      '0.93998560298662518840591198180414196987762002772873')]
)
@mp.workdps(48)
def test_mean(nu, scale, ref):
    nu = mp.mpf(nu)
    scale = mp.mpf(scale)
    ref = mp.mpf(ref)
    m = nakagami.mean(nu, loc=0, scale=scale)
    assert mp.almosteq(m, ref)


# Reference values are computed with Wolfram Alpha.
# Given nu and scale (loc=0), the variance is
#     Variance[NakagamiDistribution[nu, scale**2]]
#
@pytest.mark.parametrize(
    'nu, scale, ref',
    [(3.0, 3.0,
      '0.71650374541753735952232701674349825455824474774142'),
     (2.0, 1.0,
      '0.11642706617787065168238154845263981381954610642575')]
)
@mp.workdps(48)
def test_var(nu, scale, ref):
    nu = mp.mpf(nu)
    scale = mp.mpf(scale)
    ref = mp.mpf(ref)
    v = nakagami.var(nu, loc=0, scale=scale)
    assert mp.almosteq(v, ref)


@pytest.mark.parametrize(
    'x',
    [[2, 4, 8, 16],
     [5.375, 4.625, 4.250, 5.125, 5.000, 5.125, 4.250, 4.500, 5.125, 5.500]]
)
@mp.workdps(40)
def test_mle(x):
    # This is a crude test of nakagami.mle().
    nu_hat, _, scale_hat = nakagami.mle(x, loc=0)
    nll = nakagami.nll(x, nu=nu_hat, loc=0, scale=scale_hat)
    delta = 1e-9
    n = 2
    dirs = set(product(*([[-1, 0, 1]]*n))) - set([(0,)*n])
    for d in dirs:
        nu = nu_hat + d[0]*delta
        scale = scale_hat + d[1]*delta
        assert nll < nakagami.nll(x, nu=nu, loc=0, scale=scale)
