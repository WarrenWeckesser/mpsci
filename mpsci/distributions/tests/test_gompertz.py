
from mpmath import mp
from mpsci.distributions import gompertz


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
    h = gompertz.entropy(c, scale)
    hi = -mp.quad(lambda t: (gompertz.pdf(t, c, scale) *
                             gompertz.logpdf(t, c, scale)),
                  [0, (gompertz.mean(c, scale)
                       + 100*mp.sqrt(gompertz.var(c, scale)))])
    assert mp.almosteq(h, hi)
