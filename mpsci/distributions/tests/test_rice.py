import pytest
from mpmath import mp
from mpsci.distributions import rice


@mp.workdps(55)
def test_pdf_logpdf():
    x = 5
    nu = 2
    sigma = 3
    p = rice.pdf(x, nu, sigma)
    # Reference value computed with Wolfram Alpha:
    #    PDF[RiceDistribution[2, 3], 5]
    val = '0.147895622189182736724316687543112924251334758218654595827535'
    assert mp.almosteq(p, mp.mpf(val))

    logp = rice.logpdf(x, nu, sigma)
    assert mp.almosteq(logp, mp.log(mp.mpf(val)))


@mp.workdps(55)
def test_cdf_sf():
    x = 5
    nu = 2
    sigma = 3
    cdf = rice.cdf(x, nu, sigma)
    # Reference value computed with Wolfram Alpha:
    #    CDF[RiceDistribution[2, 3], 5]
    val = '0.676335066267619908085529790477967022985718296948574400046663'
    assert mp.almosteq(cdf, mp.mpf(val))

    sf = rice.sf(x, nu, sigma)
    assert mp.almosteq(sf, 1 - mp.mpf(val))


@mp.workdps(55)
def test_mean():
    nu = 2
    sigma = 3
    m = rice.mean(nu, sigma)
    # Expected value computed with Wolfram Alpha:
    #   Mean[RiceDistribution[2, 3]]
    val = '4.16652436436299795861868470929154000443627136280337922061687'
    assert mp.almosteq(m, mp.mpf(val))


@mp.workdps(55)
def test_var():
    nu = 2
    sigma = 3
    v = rice.var(nu, sigma)
    # Expected value computed with Wolfram Alpha:
    #   Variance[RiceDistribution[2, 3]]
    val = '4.64007472116951582653420522824375044825113849704596201317590'
    assert mp.almosteq(v, mp.mpf(val))


@pytest.mark.parametrize('nu, sigma',
                         [(0.5, 3.0),
                          (0.125, 25.0),
                          (10, 4),
                          (240, 725)])
def test_mean_with_integral(nu, sigma):
    m = rice.mean(nu, sigma)
    q = mp.quad(lambda t: t*rice.pdf(t, nu, sigma), [0, mp.inf])
    assert mp.almosteq(m, q)


# Reference values were computed with Wolfram Alpha.  The input
#     moments RiceDistribution 7 3
# generates a table of raw moments.  When n is even, the moment
# is a rational.  When n is odd, Wolfram Alpha expressed thre moment
# a linear combination of the Bessel functions I0 and I1.  In the
# following, the reference value in the odd case holds the coefficients
# of the linear combination.
@pytest.mark.parametrize(
    'n, nu, sigma, ref',
    [(1, 7, 3, ('67/6', '49/6')),
     (2, 7, 3, 67),
     (3, 7, 3, ('5533/6', '4165/6')),
     (4, 7, 3, 6577),
     (5, 7, 3, ('310325/3', '234122/3')),
     (6, 7, 3, 827371),
     (7, 7, 3, ('86237725/6', '65080183/6')),
     (8, 7, 3, 125611393),
     (9, 7, 3, ('14180521075/6', '10702014883/6')),
     (1, '2/3', 5, ('227/45', '2/45')),
     (2, '2/3', 5, '454/9'),
     (3, '2/3', 5, ('154583/405', '1808/405')),
     (4, '2/3', 5, '412216/81')]
)
@mp.workdps(50)
def test_noncentral_moment(n, nu, sigma, ref):
    nu = mp.mpf(nu)
    sigma = mp.mpf(sigma)
    moment = rice.noncentral_moment(n, nu, sigma)
    if isinstance(ref, tuple):
        c0, c1 = mp.mpf(ref[0]), mp.mpf(ref[1])
        r = nu**2/sigma**2/4
        ref = mp.sqrt(mp.pi/2)*(c0*mp.besseli(0, r)
                                + c1*mp.besseli(1, r))/mp.exp(r)
    else:
        ref = mp.mpf(ref)
    assert mp.almosteq(moment, ref)
