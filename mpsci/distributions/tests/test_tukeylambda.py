import pytest
from mpmath import mp
from mpsci.distributions import tukeylambda


# Reference values computed with Wolfram Alpha, e.g.:
#   PDF[TukeyLambdaDistribution[1/4], 3]
@pytest.mark.parametrize(
    'x, lam, ref',
    [(3, 0.25, '0.015209540163989895266168554585369291998966147023020353'),
     (2.5, -5, '0.0075727258100603655678595047600057319454228543921075618356')]
)
@mp.workdps(50)
def test_pdf_logpdf(x, lam, ref):
    ref = mp.mpf(ref)
    pdf = tukeylambda.pdf(x, lam)
    assert mp.almosteq(pdf, ref)
    logpdf = tukeylambda.logpdf(x, lam)
    assert mp.almosteq(logpdf, mp.log(ref))


# Reference values computed with Wolfram Alpha, e.g.:
#   CDF[TukeyLambdaDistribution[1/4], 3]
@pytest.mark.parametrize(
    'x, lam, ref',
    [(3, 0.25, '0.9961535901714251022090190749616906562977774265939088584'),
     (2.5, -5, '0.5193279875938647544212885839855118967385315393157213208')]
)
@mp.workdps(50)
def test_cdf_sf(x, lam, ref):
    ref = mp.mpf(ref)
    cdf = tukeylambda.cdf(x, lam)
    assert mp.almosteq(cdf, ref)
    sf = tukeylambda.sf(-x, lam)
    assert mp.almosteq(sf, ref)


@mp.workdps(50)
def test_invcdf_invsf():
    # Reference value from Wolfram Alpha:
    #    InverseCDF[TukeyLambdaDistribution[1/4], 1/8]
    ref = mp.mpf('-1.49025861052989648440072372962500018549911750201397045')
    p = mp.mpf('0.125')
    lam = 0.25
    invcdf = tukeylambda.invcdf(p, lam)
    assert mp.almosteq(invcdf, ref)
    invsf = tukeylambda.invsf(p, lam)
    assert mp.almosteq(invsf, -ref)


# Reference values computed with Wolfram Alpha, e.g.:
#   Variance[TukeyLambdaDistribution[1/4]]
@mp.workdps(50)
def test_var():
    lam = 0.25
    var = tukeylambda.var(lam)
    ref = mp.mpf('1.5565367754520328700389296299172261736789458824344863404')
    assert mp.almosteq(var, ref)
