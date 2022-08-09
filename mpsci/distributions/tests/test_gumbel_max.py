

import pytest
import mpmath
from mpsci.distributions import gumbel_max
from ._utils import check_mle


@pytest.mark.parametrize('scale', [1, 2])
def test_pdf_scale(scale):
    with mpmath.workdps(50):
        x = mpmath.mp.zero
        p = gumbel_max.pdf(x, 0, scale)
        assert mpmath.almosteq(p, 1/mpmath.e/scale)


def test_pdf():
    with mpmath.workdps(50):
        x = mpmath.mpf(-1.0)
        loc = mpmath.mp.zero
        scale = mpmath.mpf('0.1')
        p = gumbel_max.pdf(x, loc, scale)
        # Expected value was computed with Wolfram Alpha.  GumbelDistribution
        # in Alpha corresponds to the Gumbel distributob for minima, which in
        # SciPy is `scipy.stats.gumbel_l`.  With `loc=0`, we can simply flip
        # the sign of the x values to get the corresponds PDF values for
        # gumbel_max:
        #     PDF[GumbelDistribution[0, 1/10], 1]
        valstr = '2.3463581492019736117406984253693582114273335082428261e-9561'
        expected = mpmath.mpf(valstr)
        assert mpmath.almosteq(p, expected)


def test_sf():
    with mpmath.workdps(50):
        x = mpmath.mpf(-1.0)
        loc = mpmath.mp.zero
        scale = mpmath.mpf(3.0)
        p = gumbel_max.sf(x, loc, scale)
        # Expected value computed with Wolfram Alpha:
        #     CDF[GumbelDistribution[0, 3], 1]
        # The CDF and the negative of x are used because Wolfram Alpha's
        # Gumbel distribution is for the minima, like SciPy's gumbel_l.
        valstr = '0.75231869633420544657609353696784628454909394863346165'
        expected = mpmath.mpf(valstr)
        assert mpmath.almosteq(p, expected)


def test_cdf():
    with mpmath.workdps(50):
        x = mpmath.mpf(-1.0)
        loc = mpmath.mp.zero
        scale = mpmath.mpf(3.0)
        p = gumbel_max.cdf(x, loc, scale)
        # Expected value computed with Wolfram Alpha:
        #     1 - CDF[GumbelDistribution[0, 3], 1]
        # 1 - CDF and the negative of x are used because Wolfram Alpha's
        # Gumbel distribution is for the minima, like SciPy's gumbel_l.
        valstr = '0.24768130366579455342390646303215371545090605136653835'
        expected = mpmath.mpf(valstr)
        assert mpmath.almosteq(p, expected)


@pytest.mark.parametrize(
    'x',
    [[2, 4, 8, 16],
     [-990, -750, -375, -305, -300, -210, -60, 2]]
)
def test_mle(x):
    check_mle(gumbel_max, x)
