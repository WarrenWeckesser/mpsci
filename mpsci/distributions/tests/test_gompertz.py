
import mpmath
from mpsci.distributions import gompertz


mpmath.mp.dps = 60


def test_pdf():
    x = mpmath.mpf(1.25)
    c = mpmath.mpf(3.0)
    scale = mpmath.mpf(0.5)
    pdf = gompertz.pdf(x, c, scale)
    # Expected value computed with Wolfram Alpha:
    #   PDF[GompertzDistribution[1/scale, c], 5/4]
    expected = mpmath.mpf(
        '1.96970502106856635803960342532108943718675835132009651139889e-13')
    assert mpmath.almosteq(pdf, expected)


def test_logpdf():
    x = mpmath.mpf(1.25)
    c = mpmath.mpf(3.0)
    scale = mpmath.mpf(0.5)
    logpdf = gompertz.logpdf(x, c, scale)
    # Expected value computed with Wolfram Alpha:
    #   Log[PDF[GompertzDistribution[1/scale, c], 5/4]]
    expected = mpmath.mpf(
        '-29.25572241288236531339805049512319627682531267800647922882582')
    assert mpmath.almosteq(logpdf, expected)


def test_cdf():
    x = mpmath.mpf(1.25)
    c = mpmath.mpf(3.0)
    scale = mpmath.mpf(0.5)
    cdf = gompertz.cdf(x, c, scale)
    expected = -mpmath.expm1(-c*mpmath.expm1(x/scale))
    assert mpmath.almosteq(cdf, expected)


def test_invcdf():
    p = mpmath.mpf(0.25)
    c = mpmath.mpf(3.0)
    scale = mpmath.mpf(0.5)
    x = gompertz.invcdf(p, c, scale)
    # Quantile reported by Wolfram Alpha:
    #   Quantile[GompertzDistribution[1/scale, c], 1/4]
    expected = mpmath.log(1 + mpmath.log(mpmath.mpf('4/3'))/3)/2
    assert mpmath.almosteq(x, expected)


def test_sf():
    x = mpmath.mpf(1.25)
    c = mpmath.mpf(3.0)
    scale = mpmath.mpf(0.5)
    sf = gompertz.sf(x, c, scale)
    expected = mpmath.exp(-c*mpmath.expm1(x/scale))
    assert mpmath.almosteq(sf, expected)


def test_invsf():
    p = mpmath.mpf(0.25)
    c = mpmath.mpf(3.0)
    scale = mpmath.mpf(0.5)
    x = gompertz.invsf(p, c, scale)
    expected = scale * mpmath.log(1 - mpmath.log(p)/c)
    assert mpmath.almosteq(x, expected)


def test_mean():
    c = mpmath.mpf(3.0)
    scale = mpmath.mpf(0.5)
    m = gompertz.mean(c, scale)
    # Expected value computed with Wolfram Alpha:
    #   Mean[GompertzDistribution[1/scale, c]]
    expected = mpmath.mpf(
        '0.131041870127659248094359303011216347787736766926224587908757')
    assert mpmath.almosteq(m, expected)
