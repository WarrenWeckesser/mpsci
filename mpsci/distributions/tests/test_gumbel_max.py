import pytest
from mpmath import mp
from mpsci.distributions import gumbel_max
from ._utils import call_and_check_mle


def test_support():
    loc = 4
    scale = 0.25
    sup = gumbel_max.support(loc, scale)
    assert sup == (mp.ninf, mp.inf)


@pytest.mark.parametrize('scale', [1, 2])
@mp.workdps(50)
def test_pdf_scale(scale):
    x = mp.zero
    p = gumbel_max.pdf(x, 0, scale)
    assert mp.almosteq(p, 1/mp.e/scale)


@mp.workdps(50)
def test_pdf():
    x = mp.mpf(-1.0)
    loc = mp.zero
    scale = mp.mpf('0.1')
    p = gumbel_max.pdf(x, loc, scale)
    # Expected value was computed with Wolfram Alpha.  GumbelDistribution
    # in Alpha corresponds to the Gumbel distribution for minima, which in
    # SciPy is `scipy.stats.gumbel_l`.  With `loc=0`, we can simply flip
    # the sign of the x values to get the corresponding PDF values for
    # gumbel_max:
    #     PDF[GumbelDistribution[0, 1/10], 1]
    valstr = '2.3463581492019736117406984253693582114273335082428261e-9561'
    expected = mp.mpf(valstr)
    assert mp.almosteq(p, expected)


@mp.workdps(50)
def test_sf():
    x = mp.mpf(-1.0)
    loc = mp.zero
    scale = mp.mpf(3.0)
    p = gumbel_max.sf(x, loc, scale)
    # Expected value computed with Wolfram Alpha:
    #     CDF[GumbelDistribution[0, 3], 1]
    # The CDF and the negative of x are used because Wolfram Alpha's
    # Gumbel distribution is for the minima, like SciPy's gumbel_l.
    valstr = '0.75231869633420544657609353696784628454909394863346165'
    expected = mp.mpf(valstr)
    assert mp.almosteq(p, expected)


@mp.workdps(50)
def test_invsf():
    loc = 2
    scale = 5
    p = 1/8
    x = gumbel_max.invsf(p, loc, scale)
    # Given p, loc and scale, use Wolfram Alpha to compute the reference:
    #    -InverseCDF[GumbelDistribution[-loc, scale], p]
    # which in this case is
    #    -InverseCDF[GumbelDistribution[-2, 5], 1/8]
    valstr = '12.06709339019974022667775111351415772320947255750247036589'
    assert mp.almosteq(x, mp.mpf(valstr))


@mp.workdps(50)
def test_cdf():
    x = mp.mpf(-1.0)
    loc = mp.zero
    scale = mp.mpf(3.0)
    p = gumbel_max.cdf(x, loc, scale)
    # Expected value computed with Wolfram Alpha:
    #     1 - CDF[GumbelDistribution[0, 3], 1]
    # 1 - CDF and the negative of x are used because Wolfram Alpha's
    # Gumbel distribution is for the minima, like SciPy's gumbel_l.
    valstr = '0.24768130366579455342390646303215371545090605136653835'
    expected = mp.mpf(valstr)
    assert mp.almosteq(p, expected)


@mp.workdps(50)
def test_invcdf():
    loc = -2
    scale = 5
    p = 7/8
    x = gumbel_max.invcdf(p, loc, scale)
    # Reference value computed with Wolfram Alpha:
    #     InverseCDF[GumbelDistribution[2, 5], 1/8]
    # That is a value for gumbel_min.  We reuse this value here through the
    # identity
    #     gumbel_max.invcdf(p, loc, scale) = -gumbel_min.invcdf(1 - p, -loc, scale)
    valstr = '-8.067093390199740226677751113514157723209472557502470'
    assert mp.almosteq(x, -mp.mpf(valstr))


@mp.workdps(50)
def test_mean():
    loc = -2
    scale = 3
    m = gumbel_max.mean(loc, scale)
    # valstr comes from Wolfram Alpha:
    #     Mean[GumbelDistribution[2, 3]]
    # This is the mean for the gumbel_min.
    # We use the relation
    #     gumbel_min.mean(loc, scale) = -gumbel_max.mean(-loc, scale).
    valstr = '0.26835300529540141818046372975279270687352199218022920358'
    assert mp.almosteq(m, -mp.mpf(valstr))


@mp.workdps(50)
def test_var():
    loc = 2
    scale = 3
    v = gumbel_max.var(loc, scale)
    # Wolfram Alpha:
    #     Variance[GumbelDistribution[2, 3]]
    # (The min and max versions of Gumbel have the same variance.)
    valstr = '14.80440660163403792825173649981422670297054911086118593962'
    assert mp.almosteq(v, mp.mpf(valstr))


@pytest.mark.parametrize(
    'x',
    [[2, 4, 8, 16],
     [-990, -750, -375, -305, -300, -210, -60, 2]]
)
@mp.workdps(50)
def test_mle(x):
    call_and_check_mle(gumbel_max.mle, gumbel_max.nll, x)
