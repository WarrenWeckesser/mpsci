
import pytest
from mpmath import mp
from mpsci.distributions import burr12, Initial
from ._utils import check_mle


@mp.workdps(25)
def test_pdf_outside_supprt():
    assert burr12.pdf(-1, 2, 3, 4) == 0


@mp.workdps(25)
def test_logpdf_outside_supprt():
    assert burr12.logpdf(-1, 2, 3, 4) == mp.ninf


@mp.workdps(25)
def test_pdf_basic():
    c = 2
    d = 3
    scale = 4
    x = 8
    p = burr12.pdf(x, c, d, scale)
    assert mp.almosteq(p, mp.mpf('3/625'))


@mp.workdps(25)
def test_logpdf_basic():
    c = 2
    d = 3
    scale = 4
    x = 8
    logp = burr12.logpdf(x, c, d, scale)
    assert mp.almosteq(logp, mp.log(mp.mpf('3/625')))


@mp.workdps(25)
def test_cdf_outside_supprt():
    assert burr12.cdf(-1, 2, 3, 4) == 0


@pytest.mark.parametrize(
    'x, c, d, scale, ref',
    [(3, 1, 2, 4, '33/49'),
     (3, 2, 1, 6, '1/5')])
@mp.workdps(25)
def test_cdf_basic(x, c, d, scale, ref):
    cdf = burr12.cdf(x, c, d, scale)
    assert mp.almosteq(cdf, mp.mpf(ref))


@mp.workdps(25)
def test_invcdf_basic():
    c = 1
    d = 2
    scale = 4
    p = mp.mpf('33/49')
    x = burr12.invcdf(p, c, d, scale)
    assert mp.almosteq(x, 3)


@mp.workdps(25)
def test_logcdf_basic():
    x = 3
    c = 2
    d = 1
    scale = 6
    logp = burr12.logcdf(x, c, d, scale)
    ref = mp.log(mp.mpf('0.2'))
    assert mp.almosteq(logp, ref)


@mp.workdps(25)
def test_sf_outside_supprt():
    assert burr12.sf(-1, 2, 3, 4) == 1


@mp.workdps(25)
def test_logsf_outside_supprt():
    assert burr12.logsf(-1, 2, 3, 4) == 0


@mp.workdps(25)
def test_sf_basic():
    c = 1
    d = 2
    scale = 4
    x = 3
    sf = burr12.sf(x, c, d, scale)
    assert mp.almosteq(sf, mp.mpf('16/49'))


@mp.workdps(25)
def test_invsf_basic():
    c = 1
    d = 2
    scale = 4
    p = mp.mpf('16/49')
    x = burr12.invsf(p, c, d, scale)
    assert mp.almosteq(x, 3)


@mp.workdps(25)
def test_logsf_basic():
    c = 1
    d = 2
    scale = 4
    x = 3
    logsf = burr12.logsf(x, c, d, scale)
    assert mp.almosteq(logsf, mp.log(mp.mpf('16/49')))


@mp.workdps(25)
def test_mean_basic():
    c = mp.mpf(2)
    d = mp.mpf(3)
    scale = mp.mpf(7)
    mean = burr12.mean(c, d, scale)
    assert mp.almosteq(mean, scale*d*mp.beta(d - 1/c, 1 + 1/c))


@mp.workdps(25)
def test_var_basic():
    c = mp.mpf(2)
    d = mp.mpf(3)
    scale = mp.mpf(7)
    mu1 = burr12.mean(c, d, 1)
    mu2 = d*mp.beta((c*d - 2)/c, (c + 2)/c)
    var = burr12.var(c, d, scale)
    assert mp.almosteq(var, scale**2 * (-mu1**2 + mu2))


@mp.workdps(25)
def test_median_basic():
    c = mp.mpf(2)
    d = mp.mpf(3)
    scale = mp.mpf(7)
    median = burr12.median(c, d, scale)
    assert mp.almosteq(median, scale*mp.powm1(2, 1/d)**(1/c))


@mp.workdps(25)
def test_mode_basic():
    c = mp.mpf(2)
    d = mp.mpf(3)
    scale = mp.mpf(7)
    mode = burr12.mode(c, d, scale)
    assert mp.almosteq(mode, scale*((c - 1)/(d*c + 1))**(1/c))


@mp.workdps(25)
def test_mode0_basic():
    c = mp.mpf(0.5)
    d = mp.mpf(3)
    scale = mp.mpf(7)
    mode = burr12.mode(c, d, scale)
    assert mode == 0


@pytest.mark.parametrize(
    'x, c0, d0, scale0',
    [([1, 2, 3, 4, 5, 8, 13], 1.6, 3.7, 11.2),
     ([0.090, 0.053, 0.202, 0.165, 0.853, 0.310, 1.181, 0.136, 0.248,
       0.030, 0.128, 0.130, 0.175, 0.270, 0.377, 0.178, 0.815, 0.223,
       0.147, 0.627, 1.514, 0.397, 0.014, 0.199, 0.206, 0.391, 1.160,
       0.210, 0.184, 0.311, 0.210, 0.062, 0.345, 0.847, 0.398, 0.159,
       0.220, 0.173, 0.061, 0.324, 0.145, 0.190, 0.143, 0.411, 0.317,
       0.130, 0.296, 0.008, 0.320, 0.271, 0.134, 0.077, 0.189, 0.263,
       0.554, 0.780, 0.066, 0.670, 1.235, 0.240],
      1.5, 1.5, 0.25)]
)
@mp.workdps(50)
def test_mle(x, c0, d0, scale0):
    p_hat = burr12.mle(x, c=Initial(c0), d=Initial(d0), scale=Initial(scale0))
    check_mle(burr12.nll, x, p_hat)


@pytest.mark.parametrize(
    'x, c, d0, scale0',
    [([1, 1, 2, 3, 5, 8, 13, 21],
      1.5, 1.125, 4.5),
     ([5.00, 2.25, 2.75, 0.25, 3.50, 1.00, 6.00, 3.75,
       4.75, 2.75, 12.0, 5.50, 7.00, 3.50, 2.75, 1.50],
      2, 5, 10),
     ([5.25, 0.625, 1.25, 0.625, 0.375, 25.0, 24.125, 0.125,
       0.25, 0.375, 3.375, 600.125, 2.125, 5.125, 2.25, 44.375,
       0.25, 0.5, 0.75, 1.125, 4.875, 20.125, 0.875, 0.125],
      0.75, 1.0, 2.0),
     ([18.2500, 16.5625, 19.6250, 18.3125, 17.8125,
       16.5000, 18.0000, 19.1875, 18.9375, 14.8750,
       16.2500, 19.1875, 17.2500, 17.7500, 19.2500,
       18.6250, 20.2500, 17.4375, 14.8750, 18.0000,
       14.8125, 15.2500, 20.0625, 18.0000, 16.1250],
      16, 5, 20)]
)
@mp.workdps(50)
def test_mle_c_fixed(x, c, d0, scale0):
    p_hat = burr12.mle(x, c=c, d=Initial(d0), scale=Initial(scale0))
    check_mle(lambda x, d, scale: burr12.nll(x, c, d, scale), x, p_hat[1:])


@pytest.mark.parametrize(
    'x, c, d',
    [([1, 1, 2, 3, 5, 8, 13, 21], 1.5, 1.125),
     ([1.48016889e-96, 3.54563945e-77, 4.68530581e-99, 3.00079303e-73,
       8.55871035e-91, 4.65244972e-63, 5.22063674e-59, 3.66513438e-68,
       4.46739790e-82, 9.63256794e-61, 3.11272713e-94, 4.01509576e-66],
      0.03125, 200)]
)
@mp.workdps(50)
def test_mle_c_and_d_fixed(x, c, d):
    p_hat = burr12.mle(x, c=c, d=d)
    check_mle(lambda x, scale: burr12.nll(x, c, d, scale), x, p_hat[2:])


@pytest.mark.parametrize(
    'x, d, scale',
    [([0.5, 1.0, 1.5, 3.0, 8.0, 24.0], 0.5, 2.0),
     ([0.335, 0.595, 0.048, 0.515, 0.043, 0.129, 0.541, 0.276, 0.258,
       0.056, 0.251, 0.074, 0.011, 0.126, 0.578, 0.263, 0.555, 0.260],
      20, 5)]
)
@mp.workdps(50)
def test_mle_d_and_scale_fixed(x, d, scale):
    p_hat = burr12.mle(x, d=d, scale=scale)
    check_mle(lambda x, c: burr12.nll(x, c, d, scale), x, p_hat[:1])


@pytest.mark.parametrize(
    'x, scale',
    [([0.243, 1.181, 0.330, 0.172, 0.496, 0.122, 0.380, 0.019, 0.337,
       0.112, 0.237, 0.537, 0.036, 1.837, 0.148, 0.564, 0.696, 0.435,
       1.108, 0.134, 0.005, 2.136, 0.075, 0.253], 5),
     ([2.200, 9.368, 6.731, 4.264, 4.629, 4.355, 5.841, 2.551, 3.809], 0.125)]
)
@mp.workdps(50)
def test_mle_scale_fixed(x, scale):
    p_hat = burr12.mle(x, scale=scale)
    check_mle(lambda x, c, d: burr12.nll(x, c, d, scale), x, p_hat[:2])


@pytest.mark.parametrize(
    'x, d, c0, scale0',
    [([0.116, 0.034, 0.081, 0.051, 0.171, 0.107, 0.171, 0.106, 0.116],
      1.5, 3.25, 0.12),
     ([12.5, 18.75, 6.25, 125.0, 37.5],
      0.125, 6.2, 6.5)]
)
@mp.workdps(50)
def test_mle_d_fixed(x, d, c0, scale0):
    p_hat = burr12.mle(x, c=Initial(c0), d=d, scale=Initial(scale0))
    check_mle(lambda x, c, scale: burr12.nll(x, c, d, scale), x,
              [p_hat[0], p_hat[2]])
