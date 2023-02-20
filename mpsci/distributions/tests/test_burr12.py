
from mpmath import mp
from mpsci.distributions import burr12


def test_pdf_basic():
    with mp.workdps(25):
        c = 2
        d = 3
        scale = 4
        x = 8
        p = burr12.pdf(x, c, d, scale)
        assert mp.almosteq(p, mp.mpf('3/625'))


def test_logpdf_basic():
    with mp.workdps(25):
        c = 2
        d = 3
        scale = 4
        x = 8
        logp = burr12.logpdf(x, c, d, scale)
        assert mp.almosteq(logp, mp.log(mp.mpf('3/625')))


def test_cdf_basic():
    with mp.workdps(25):
        c = 1
        d = 2
        scale = 4
        x = 3
        cdf = burr12.cdf(x, c, d, scale)
        assert mp.almosteq(cdf, mp.mpf('33/49'))


def test_invcdf_basic():
    with mp.workdps(25):
        c = 1
        d = 2
        scale = 4
        p = mp.mpf('33/49')
        x = burr12.invcdf(p, c, d, scale)
        assert mp.almosteq(x, 3)


def test_sf_basic():
    with mp.workdps(25):
        c = 1
        d = 2
        scale = 4
        x = 3
        sf = burr12.sf(x, c, d, scale)
        assert mp.almosteq(sf, mp.mpf('16/49'))


def test_invsf_basic():
    with mp.workdps(25):
        c = 1
        d = 2
        scale = 4
        p = mp.mpf('16/49')
        x = burr12.invsf(p, c, d, scale)
        assert mp.almosteq(x, 3)


def test_logsf_basic():
    with mp.workdps(25):
        c = 1
        d = 2
        scale = 4
        x = 3
        logsf = burr12.logsf(x, c, d, scale)
        assert mp.almosteq(logsf, mp.log(mp.mpf('16/49')))


def test_mean_basic():
    with mp.workdps(25):
        c = mp.mpf(2)
        d = mp.mpf(3)
        scale = mp.mpf(7)
        mean = burr12.mean(c, d, scale)
        assert mp.almosteq(mean, scale*d*mp.beta(d - 1/c, 1 + 1/c))


def test_var_basic():
    with mp.workdps(25):
        c = mp.mpf(2)
        d = mp.mpf(3)
        scale = mp.mpf(7)
        mu1 = burr12.mean(c, d, 1)
        mu2 = d*mp.beta((c*d - 2)/c, (c + 2)/c)
        var = burr12.var(c, d, scale)
        assert mp.almosteq(var, scale**2 * (-mu1**2 + mu2))


def test_median_basic():
    with mp.workdps(25):
        c = mp.mpf(2)
        d = mp.mpf(3)
        scale = mp.mpf(7)
        median = burr12.median(c, d, scale)
        assert mp.almosteq(median, scale*mp.powm1(2, 1/d)**(1/c))


def test_mode_basic():
    with mp.workdps(25):
        c = mp.mpf(2)
        d = mp.mpf(3)
        scale = mp.mpf(7)
        mode = burr12.mode(c, d, scale)
        assert mp.almosteq(mode, scale*((c - 1)/(d*c + 1))**(1/c))


def test_mode0_basic():
    with mp.workdps(25):
        c = mp.mpf(0.5)
        d = mp.mpf(3)
        scale = mp.mpf(7)
        mode = burr12.mode(c, d, scale)
        assert mode == 0
