
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
