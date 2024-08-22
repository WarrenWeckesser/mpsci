
import pytest
from mpmath import mp
from mpsci.distributions import burr12


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
