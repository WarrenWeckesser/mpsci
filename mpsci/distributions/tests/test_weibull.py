from itertools import product
import pytest
from mpmath import mp
from mpsci.distributions import weibull_max, weibull_min
from ._expect import check_entropy_with_integral


@pytest.mark.parametrize('dist, xsign', [(weibull_min, 1), (weibull_max, -1)])
@mp.workdps(50)
def test_pdf(dist, xsign):
    k = 1.25
    loc = 1
    scale = 3
    x = 2.5
    p = dist.pdf(xsign*x, k, xsign*loc, scale)
    # Expected value was computed with Wolfram Alpha:
    #   PDF[WeibullDistribution[5/4, 3, 1], 5/2]
    valstr = '0.23010863853495101956594599926808749710908978279269136511'
    expected = mp.mpf(valstr)
    assert mp.almosteq(p, expected)


@pytest.mark.parametrize('dist', [weibull_min, weibull_max])
@pytest.mark.parametrize('k', [0.125, 1, 2.25])
@mp.workdps(50)
def test_pdf_loc_edge_case(dist, k):
    loc = 0
    scale = mp.mpf(0.25)
    p = dist.pdf(loc, k, loc, scale)
    expected = mp.inf if k < 1 else 1/scale if k == 1 else 0
    assert mp.almosteq(p, expected)


@pytest.mark.parametrize('dist', [weibull_min, weibull_max])
@mp.workdps(50)
def test_cdf_sf(dist):
    k = 1.25
    loc = 1
    scale = 3
    x = 2.5
    if dist == weibull_min:
        cdf = dist.cdf(x, k, loc, scale)
        sf = dist.sf(x, k, loc, scale)
    else:
        cdf = dist.sf(-x, k, -loc, scale)
        sf = dist.cdf(-x, k, -loc, scale)
    # Expected value computed with Wolfram Alpha:
    #   CDF[WeibullDistribution[5/4, 3, 1], 5/2]
    valstr = '0.34324760759355263295507068694174586069396497366940988665'
    expected = mp.mpf(valstr)
    assert mp.almosteq(cdf, expected)
    assert mp.almosteq(sf, 1 - expected)


@pytest.mark.parametrize('dist, sign', [(weibull_min, 1), (weibull_max, -1)])
@mp.workdps(50)
def test_invcdf(dist, sign):
    k = 1.5
    loc = 1
    scale = 3
    p = mp.mpf('1/16')
    if sign == -1:
        p = 1 - p
    x = dist.invcdf(p, k, sign*loc, scale)
    # Expected value computed with Wolfram Alpha:
    #   InverseCDF[WeibullDistribution[3/2, 3, 1], 1/16]
    valstr = '1.48268884363357472595994253221918943182906779611027773721'
    expected = mp.mpf(valstr)
    assert mp.almosteq(x, sign * expected)


@pytest.mark.parametrize('dist, sign', [(weibull_min, 1), (weibull_max, -1)])
@mp.workdps(50)
def test_mean(dist, sign):
    k = 1.5
    loc = 1
    scale = 3
    m = dist.mean(k, sign*loc, scale)
    # Expected value computed with Wolfram Alpha:
    #   Mean[WeibullDistribution[3/2, 3, 1]]
    valstr = '3.708235878852800833890576056309027571038654532113587396788'
    expected = mp.mpf(valstr)
    assert mp.almosteq(m, sign*expected)


@pytest.mark.parametrize('dist', [weibull_min, weibull_max])
@pytest.mark.parametrize('k', [1.125, 4.75])
@mp.workdps(50)
def test_mode_k_gt_1(dist, k):
    loc = 1
    scale = 0.125
    m = dist.mode(k, loc, scale)
    # A crude test of the mode.
    pm = dist.pdf(m, k, loc, scale)
    delta = mp.sqrt(mp.eps)
    left = (1 - mp.sign(m) * delta) * m
    right = (1 + mp.sign(m) * delta) * m
    assert dist.pdf(left, k, loc, scale) < pm
    assert dist.pdf(right, k, loc, scale) < pm


@pytest.mark.parametrize('dist, sign', [(weibull_min, 1), (weibull_max, -1)])
@mp.workdps(50)
def test_var(dist, sign):
    k = 1.5
    loc = 1
    scale = 3
    m = dist.var(k, sign*loc, scale)
    # Expected value computed with Wolfram Alpha:
    #   Variance[WeibullDistribution[3/2, 3, 1]]
    valstr = '3.38121256332538801963394969804653523502508870632016051797309'
    expected = mp.mpf(valstr)
    assert mp.almosteq(m, expected)


@pytest.mark.parametrize('dist, sign', [(weibull_min, 1), (weibull_max, -1)])
@mp.workdps(50)
def test_skewness(dist, sign):
    k = 1.25
    loc = 1
    scale = 3
    skew = dist.skewness(k, loc, scale)
    # Expected value computed with Wolfram Alpha:
    #   Skewness[WeibullDistribution[5/4, 3, 1]]
    valstr = '1.429545236590974853525527387620583784997166374292021040338'
    expected = mp.mpf(valstr)
    assert mp.almosteq(skew, sign*expected)


@pytest.mark.parametrize('dist', [weibull_min, weibull_max])
@mp.workdps(50)
def test_kurtosis(dist):
    k = 1.25
    loc = 1
    scale = 3
    kurt = dist.kurtosis(k, loc, scale)
    # Expected value computed with Wolfram Alpha:
    #   ExcessKurtosis[WeibullDistribution[5/4, 3, 1]]
    valstr = '2.8021519350984650074697694858304410798423229238041266467027'
    expected = mp.mpf(valstr)
    assert mp.almosteq(kurt, expected)


@pytest.mark.parametrize('dist', [weibull_min, weibull_max])
@mp.workdps(50)
def test_entropy(dist):
    check_entropy_with_integral(dist, (1.25, 1, 3))


@pytest.mark.parametrize('dist, sgn', [(weibull_min, 1), (weibull_max, -1)])
@pytest.mark.parametrize(
    'x',
    [[2, 4, 8, 16],
     [5.43, 4.78, 3.38, 4.71, 4.64, 4.76, 5.45, 5.33, 4.64, 3.60,
      5.02, 4.93, 3.40, 5.37, 4.36, 4.08, 4.97, 5.65, 5.10, 4.48,
      5.44, 5.59, 4.64, 5.36, 4.99]],
)
def test_mle(dist, sgn, x):
    # This is a crude test of dist.mle().
    x = [sgn*t for t in x]
    k_hat, _, scale_hat = dist.mle(x, loc=0)
    nll = dist.nll(x, k=k_hat, loc=0, scale=scale_hat)
    delta = 1e-9
    n = 2
    dirs = set(product(*([[-1, 0, 1]]*n))) - set([(0,)*n])
    for d in dirs:
        k = k_hat + d[0]*delta
        scale = scale_hat + d[1]*delta
        assert nll < dist.nll(x, k=k, loc=0, scale=scale)


@pytest.mark.parametrize('dist, sgn', [(weibull_min, 1), (weibull_max, -1)])
@pytest.mark.parametrize(
    'x',
    [[2, 4, 8, 16],
     [5.43, 4.78, 3.38, 4.71, 4.64, 4.76, 5.45, 5.33, 4.64, 3.60,
      5.02, 4.93, 3.40, 5.37, 4.36, 4.08, 4.97, 5.65, 5.10, 4.48,
      5.44, 5.59, 4.64, 5.36, 4.99]],
)
@pytest.mark.parametrize('scale', [5, 8])
def test_mle_fixed_scale(dist, sgn, x, scale):
    # This is a crude test of dist.mle().
    x = [sgn*t for t in x]
    k_hat, loc_hat, scale_hat = dist.mle(x, loc=0, scale=scale)
    assert loc_hat == 0
    assert scale_hat == scale
    nll = dist.nll(x, k=k_hat, loc=0, scale=scale)
    delta = 1e-9
    assert nll < dist.nll(x, k_hat + delta, loc=0, scale=scale)
    assert nll < dist.nll(x, k_hat - delta, loc=0, scale=scale)
