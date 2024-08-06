import pytest
from mpmath import mp
from mpsci.distributions import cauchy


@mp.workdps(50)
def test_pdf_logpdf():
    x = 11
    # From Wolfram Alpha:
    #     PDF[CauchyDistribution[0,1], 11]
    refstr = '0.002609097427735989110965307596270727246466551569515679'
    ref = mp.mpf(refstr)
    for x1 in [x, -x]:
        p = cauchy.pdf(x1)
        assert mp.almosteq(p, ref)
        logp = cauchy.logpdf(x1)
        assert mp.almosteq(logp, mp.log(ref))


@mp.workdps(50)
def test_cdf_sf():
    x = 11
    # From Wolfram Alpha:
    #     CDF[CauchyDistribution[0,1], 11]
    refstr = '0.9711420616236955222571082767738756646907856433035175462789577'
    ref = mp.mpf(refstr)
    cdf = cauchy.cdf(x)
    assert mp.almosteq(cdf, ref)
    sf = cauchy.sf(-x)
    assert mp.almosteq(sf, ref)


# First reference value from Wolfram Alpha:
# InverseCDF[CauchyDistribution[0, 1], 4503617641769005/9007199254740992]
@pytest.mark.parametrize(
    'p, ref',
    [('4503617641769005/9007199254740992',
      '6.283185307094160200386336023081100764370534528359140e-6'),
     ('1/4', -1),
     ('3/4', 1)]
)
@mp.workdps(50)
def test_invcdf(p, ref):
    p = mp.mpf(p)
    ref = mp.mpf(ref)
    x = cauchy.invcdf(p)
    assert mp.almosteq(x, ref)


@pytest.mark.parametrize(
    'p, ref',
    [('4503581612971987/9007199254740992',
      '6.283185307094160200386336023081100764370534528359140e-6'),
     ('1/4', 1),
     ('3/4', -1)]
)
@mp.workdps(50)
def test_invsf(p, ref):
    p = mp.mpf(p)
    ref = mp.mpf(ref)
    x = cauchy.invsf(p)
    assert mp.almosteq(x, ref)


@pytest.mark.parametrize('x', [-100, -7, 11, 1300])
@mp.workdps(50)
def test_cdf_invcdf_roundtrip(x):
    p = cauchy.cdf(x)
    x1 = cauchy.invcdf(p)
    assert mp.almosteq(x1, x)


@pytest.mark.parametrize('x', [-100, -7, 11, 1300])
@mp.workdps(50)
def test_sf_invsf_roundtrip(x):
    p = cauchy.sf(x)
    x1 = cauchy.invsf(p)
    assert mp.almosteq(x1, x)
