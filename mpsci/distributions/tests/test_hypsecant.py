
from mpmath import mp
from mpsci.distributions import hypsecant


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
