
from mpmath import mp
from mpsci.distributions import gamma_gompertz


def test_basic_pdf_logpdf_cdf():
    with mp.workdps(50):
        x = 3 * mp.log(5)
        c = 2
        beta = 4
        scale = 3

        # For the above values, the exact value of the PDF is 10/96.
        p = gamma_gompertz.pdf(x, c, beta, scale)
        assert mp.almosteq(p, mp.mpf('10/96'))

        logp = gamma_gompertz.logpdf(x, c, beta, scale)
        assert mp.almosteq(logp, mp.log(mp.mpf('10/96')))

        # For the above values, the exact value of the CDF is 3/4.
        p = gamma_gompertz.cdf(x, c, beta, scale)
        assert mp.almosteq(p, mp.mpf('3/4'))


def test_beta_and_scale_1():
    # With beta == 1 and scale == 1, the PDF is simply c*exp(-c*x)
    # and the CDF is 1 - exp(-c*x).

    with mp.workdps(50):
        x = mp.mpf('11.3')
        c = mp.mpf('8.5')
        beta = mp.one
        scale = mp.one

        p = gamma_gompertz.pdf(x, c, beta, scale)
        expected = c * mp.exp(-c * x)
        assert mp.almosteq(p, expected)

        p = gamma_gompertz.cdf(x, c, beta, scale)
        expected = 1 - mp.exp(-c * x)
        assert mp.almosteq(p, expected)

        p = gamma_gompertz.sf(x, c, beta, scale)
        expected = mp.exp(-c * x)
        assert mp.almosteq(p, expected)


def test_sf_invsf_roundtrip():
    with mp.workdps(50):
        rel_eps = mp.mpf('1e-48')

        x = mp.mpf('13.5')
        p = gamma_gompertz.sf(x, 4, 3, 2)
        x1 = gamma_gompertz.invsf(p, 4, 3, 2)
        assert mp.almosteq(x1, x, rel_eps=rel_eps)

        p = mp.mpf('2/3')
        x = gamma_gompertz.invsf(p, 5, 2, 3)
        p1 = gamma_gompertz.sf(x, 5, 2, 3)
        assert mp.almosteq(p1, p, rel_eps=rel_eps)


def test_cdf_invcdf_roundtrip():
    with mp.workdps(50):
        rel_eps = mp.mpf('1e-48')

        x = mp.mpf('13.5')
        p = gamma_gompertz.cdf(x, 4, 3, 2)
        x1 = gamma_gompertz.invcdf(p, 4, 3, 2)
        assert mp.almosteq(x1, x, rel_eps=rel_eps)

        p = mp.mpf('2/3')
        x = gamma_gompertz.invcdf(p, 5, 2, 3)
        p1 = gamma_gompertz.cdf(x, 5, 2, 3)
        assert mp.almosteq(p1, p, rel_eps=rel_eps)
