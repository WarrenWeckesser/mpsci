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
     (2.5, -5, '0.5193279875938647544212885839855118967385315393157213208'),
     (-1.25, 0, '0.2227001388253088530004804523384582019709507184521052349878'),
     (-3, -1, '0.23240812075600178448012978875491734229145057102579229788159'),
     (-0.5, -2, '0.468989943540430815367258741152357011991279453295291106394098'),
     (0, -1.25, '0.5'),
     (100, -3, '0.850888295809637571379812810386688549623479342189446248413169'),
     (2000, -3, '0.944971501868898823441012561166301183368389695667748349929002'),
     (-5000, -4, '0.08408814797967718378182063883873424219506787762387318778480751')]
)
@mp.workdps(50)
def test_cdf_sf(x, lam, ref):
    ref = mp.mpf(ref)
    cdf = tukeylambda.cdf(x, lam)
    assert mp.almosteq(cdf, ref)
    sf = tukeylambda.sf(-x, lam)
    assert mp.almosteq(sf, ref)


@pytest.mark.parametrize(
    'lam, refstr',
    [(0.25, '-1.49025861052989648440072372962500018549911750201397045'),
     (0, '-1.945910149055313305105352743443179729637084729581861188')]
)
@mp.workdps(50)
def test_invcdf_invsf(lam, refstr):
    # Reference value from Wolfram Alpha:
    #    InverseCDF[TukeyLambdaDistribution[lam], 1/8]
    ref = mp.mpf(refstr)
    p = mp.mpf('0.125')
    invcdf = tukeylambda.invcdf(p, lam)
    assert mp.almosteq(invcdf, ref)
    invsf = tukeylambda.invsf(p, lam)
    assert mp.almosteq(invsf, -ref)


def test_mean():
    lam = 1.5
    loc = 3.0
    scale = 0.25
    m = tukeylambda.mean(lam, loc=loc, scale=scale)
    assert m == loc


def test_mean_lam_lt_neg1():
    lam = -2
    m = tukeylambda.mean(lam)
    assert mp.isnan(m)


# Reference values computed with Wolfram Alpha, e.g.:
#   Variance[TukeyLambdaDistribution[1/4]]
@mp.workdps(50)
def test_var():
    lam = 0.25
    var = tukeylambda.var(lam)
    ref = mp.mpf('1.5565367754520328700389296299172261736789458824344863404')
    assert mp.almosteq(var, ref)


@mp.workdps(50)
def test_var_lam_is_0():
    # When lambda is 0, the variance is pi**2/3.
    lam = 0
    var = tukeylambda.var(lam)
    ref = mp.pi**2 / 3
    assert mp.almosteq(var, ref)


def test_var_lam_lt_neg_half():
    lam = -2
    v = tukeylambda.var(lam)
    assert mp.isnan(v)


@pytest.mark.parametrize('lam', [-2.5, 0])
def test_support_lam_le_0(lam):
    # When lambda <= 0, the support is (-inf, inf).
    sup = tukeylambda.support(lam)
    assert sup == (mp.ninf, mp.inf)


@pytest.mark.parametrize('lam', [0.125, 256])
def test_support_lam_gt_0(lam):
    # When lambda > 0 with loc=0, scale=1, the support is (-1/lambda, 1/lambda).
    lam = mp.mpf(lam)
    sup = tukeylambda.support(lam)
    assert sup == (-1/lam, 1/lam)


@mp.workdps(40)
def test_noncentral_moment():
    # The reference value was computed using
    #   mp.dps = 100
    #   lam = mp.mpf('-0.1')
    #   t1 = mp.quad(lambda t: t**2 * tukeylambda.pdf(t, lam, loc=1, scale=3), [-1e9, 0])
    #   t2 = mp.quad(lambda t: t**2 * tukeylambda.pdf(t, lam, loc=1, scale=3), [0, 1e7])
    #   ref = t1 + t2
    # and then keeping the first 43 digits. (`check_noncentral_moment_with_integral` has
    # been a bit flaky when applied to tukeylambda.)
    ref = mp.mpf('44.02451960868272339412251903255121235441242')
    lam = mp.mpf('-0.1')
    m = tukeylambda.noncentral_moment(2, lam, loc=1, scale=3)
    assert mp.almosteq(m, ref)
