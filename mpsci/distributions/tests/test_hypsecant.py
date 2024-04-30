import pytest
from mpmath import mp
from mpsci.distributions import hypsecant
from ._utils import check_mle, call_and_check_mle


@mp.workdps(50)
def test_pdf_basic():
    pdf = hypsecant.pdf(5, loc=1, scale=2*3/mp.pi)
    # Expected value computed with Wolfram Alpha:
    #  PDF[SechDistribution[1, 3], 5]
    # The scale used in mpsci corresponds to 2/pi times the scale
    # used by Wolfram Alpha.
    expected = mp.mpf('0.040435054788310863393838723688535657960494051526683')
    assert mp.almosteq(pdf, expected)


@mp.workdps(50)
def test_cdf_basic():
    cdf = hypsecant.cdf(5, loc=1, scale=2*3/mp.pi)
    # Expected value computed with Wolfram Alpha:
    #  CDF[SechDistribution[1, 3], 5]
    # The scale used in mpsci corresponds to 2/pi times the scale
    # used by Wolfram Alpha.
    expected = mp.mpf('0.92199635863265696256466357196986332698026374561906')
    assert mp.almosteq(cdf, expected)


@mp.workdps(50)
def test_sf_basic():
    sf = hypsecant.sf(5, loc=1, scale=2*3/mp.pi)
    # Expected value computed with Wolfram Alpha:
    #  1 - CDF[SechDistribution[1, 3], 5]
    # The scale used in mpsci corresponds to 2/pi times the scale
    # used by Wolfram Alpha.
    expected = mp.mpf('0.0780036413673430374353364280301366730197362543809359')
    assert mp.almosteq(sf, expected)


@mp.workdps(50)
def test_invcdf_basic():
    x = hypsecant.invcdf(1/16, loc=1, scale=3*(2/mp.pi))
    # Reference value from Wolfram Alpha:
    #     InverseCDF[SechDistribution[1, 3], 1/16]
    ref = mp.mpf('-3.4266452065823773155746807670653531787193059074298')
    assert mp.almosteq(x, ref)


@mp.workdps(50)
def test_invsf_basic():
    x = hypsecant.invsf(15/16, loc=1, scale=3*(2/mp.pi))
    # Reference value from Wolfram Alpha:
    #     InverseCDF[SechDistribution[1, 3], 1/16]
    ref = mp.mpf('-3.4266452065823773155746807670653531787193059074298')
    assert mp.almosteq(x, ref)


def test_mean():
    mean = hypsecant.mean(loc=5, scale=7)
    assert mean == 5


@mp.workdps(50)
def test_var_with_integral():
    loc = 5
    scale = 7
    var = hypsecant.var(loc=loc, scale=scale)
    expected = mp.quad(lambda t: hypsecant.pdf(t, loc, scale)*(t - loc)**2,
                       [mp.ninf, mp.inf])
    assert mp.almosteq(var, expected)


@mp.workdps(100)
def test_entropy_with_integral():
    loc = 5
    scale = 7
    entr = hypsecant.entropy(loc, scale)
    intgrl = mp.quad(
        lambda t: hypsecant.pdf(t, loc, scale)*hypsecant.logpdf(t, loc, scale),
        [mp.ninf, mp.inf]
    )
    expected = -intgrl
    assert mp.almosteq(entr, expected)


@pytest.mark.parametrize(
    'x',
    [[-2, 4, -8, 16],
     [-4.87, -5.13, -5.09, -5.07, -4.94, -5.03, -4.91, -4.89, -4.59, -5.19]],
)
@mp.workdps(60)
def test_mle(x):
    call_and_check_mle(hypsecant.mle, hypsecant.nll, x)


@mp.workdps(60)
def test_mle_scale_fixed():
    x = [-25, -13, -2, 1.5, 3, 4, 16, 39]
    # Fix the scale to be 25.
    fscale = 25
    loc1, scale1 = hypsecant.mle(x, scale=fscale)
    assert scale1 == fscale
    check_mle(lambda x, loc: hypsecant.nll(x, loc=loc, scale=fscale),
              x, (loc1,))


@mp.workdps(60)
def test_mle_loc_fixed():
    x = [-25, -13, -2, 1.5, 3, 4, 16, 39]
    # Fix loc to be 0.
    floc = 0
    loc1, scale1 = hypsecant.mle(x, loc=floc)
    assert loc1 == floc
    check_mle(lambda x, scale: hypsecant.nll(x, loc=floc, scale=scale),
              x, (scale1,))


def test_mle_all_fixed():
    x = [1, 2, 3, 5, 8, 13, 21, 34, 55, 89]
    loc1, scale1 = hypsecant.mle(x, loc=1, scale=25)
    assert loc1 == 1 and scale1 == 25
