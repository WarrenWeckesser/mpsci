import pytest
from mpmath import mp
from mpsci.distributions import gompertz
from ._utils import call_and_check_mle, check_mle
from ._expect import check_entropy_with_integral


@mp.workdps(60)
def test_pdf():
    x = mp.mpf(1.25)
    c = mp.mpf(3.0)
    scale = mp.mpf(0.5)
    pdf = gompertz.pdf(x, c, scale)
    # Expected value computed with Wolfram Alpha:
    #   PDF[GompertzDistribution[1/scale, c], 5/4]
    expected = mp.mpf(
        '1.96970502106856635803960342532108943718675835132009651139889e-13')
    assert mp.almosteq(pdf, expected)


@mp.workdps(60)
def test_logpdf():
    x = mp.mpf(1.25)
    c = mp.mpf(3.0)
    scale = mp.mpf(0.5)
    logpdf = gompertz.logpdf(x, c, scale)
    # Expected value computed with Wolfram Alpha:
    #   Log[PDF[GompertzDistribution[1/scale, c], 5/4]]
    expected = mp.mpf(
        '-29.25572241288236531339805049512319627682531267800647922882582')
    assert mp.almosteq(logpdf, expected)


@mp.workdps(60)
def test_cdf():
    x = mp.mpf(1.25)
    c = mp.mpf(3.0)
    scale = mp.mpf(0.5)
    cdf = gompertz.cdf(x, c, scale)
    expected = -mp.expm1(-c*mp.expm1(x/scale))
    assert mp.almosteq(cdf, expected)


@mp.workdps(60)
def test_invcdf():
    p = mp.mpf(0.25)
    c = mp.mpf(3.0)
    scale = mp.mpf(0.5)
    x = gompertz.invcdf(p, c, scale)
    # Quantile reported by Wolfram Alpha:
    #   Quantile[GompertzDistribution[1/scale, c], 1/4]
    expected = mp.log1p(mp.log(mp.mpf('4/3'))/3)/2
    assert mp.almosteq(x, expected)


@mp.workdps(60)
def test_sf():
    x = mp.mpf(1.25)
    c = mp.mpf(3.0)
    scale = mp.mpf(0.5)
    sf = gompertz.sf(x, c, scale)
    expected = mp.exp(-c*mp.expm1(x/scale))
    assert mp.almosteq(sf, expected)


@mp.workdps(60)
def test_invsf():
    p = mp.mpf(0.25)
    c = mp.mpf(3.0)
    scale = mp.mpf(0.5)
    x = gompertz.invsf(p, c, scale)
    expected = scale * mp.log1p(-mp.log(p)/c)
    assert mp.almosteq(x, expected)


@mp.workdps(60)
def test_mean():
    c = mp.mpf(3.0)
    scale = mp.mpf(0.5)
    m = gompertz.mean(c, scale)
    # Expected value computed with Wolfram Alpha:
    #   Mean[GompertzDistribution[1/scale, c]]
    expected = mp.mpf(
        '0.131041870127659248094359303011216347787736766926224587908757')
    assert mp.almosteq(m, expected)


@mp.workdps(60)
def test_entropy_against_integral():
    c = mp.mpf(3.0)
    scale = mp.mpf(0.5)
    # The calculation of the entropy with an integral over the
    # domain [0, inf] takes an *extremely* long time.  Instead we
    # use a large finite interval.
    lower = mp.zero
    upper = gompertz.mean(c, scale) + 100*mp.sqrt(gompertz.var(c, scale))
    check_entropy_with_integral(gompertz, (c, scale), support=(lower, upper))


@pytest.mark.parametrize(
    'x',
    [[2.2, 2.2, 3.1, 0.7, 3.6, 0.6, 0.8, 1.1, 0.1],
     [0.05, 0.09, 0.12, 0.06, 0.25, 0.30, 0.10, 0.41,
      0.02, 0.05, 0.11, 0.03, 0.07, 0.02, 0.13, 0.33]]
)
@mp.workdps(50)
def test_mle(x):
    call_and_check_mle(gompertz.mle, gompertz.nll, x)


@mp.workdps(50)
def test_mle_scale_fixed():
    x = [2.2, 2.2, 3.1, 0.7, 3.6, 0.6, 0.8, 1.1, 0.1]
    # Fix the scale to be 1.
    c1, scale1 = gompertz.mle(x, scale=1)
    assert scale1 == 1
    check_mle(lambda x, c: gompertz.nll(x, c, scale=1), x, (c1,))


@mp.workdps(50)
def test_mle_c_fixed():
    x = [2.2, 2.2, 3.1, 0.7, 3.6, 0.6, 0.8, 1.1, 0.1]
    # Fix c to be 0.125
    c1, scale1 = gompertz.mle(x, c=0.125)
    assert c1 == 0.125
    check_mle(lambda x, scale: gompertz.nll(x, c=0.125, scale=scale),
              x, (scale1,))
