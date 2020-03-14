
import mpmath
from mpsci.distributions import gamma_gompertz


def test_basic_pdf_cdf():
    with mpmath.workdps(50):
        x = 3 * mpmath.log(5)
        c = 2
        beta = 4
        scale = 3

        # For the above values, the exact value of the PDF is 10/96.
        p = gamma_gompertz.pdf(x, c, beta, scale)
        assert mpmath.almosteq(p, mpmath.mp.mpf('10/96'))

        # For the above values, the exact value of the CDF is 3/4.
        p = gamma_gompertz.cdf(x, c, beta, scale)
        assert mpmath.almosteq(p, mpmath.mp.mpf('3/4'))


def test_beta_and_scale_1():
    # With beta == 1 and scale == 1, the PDF is simply c*exp(-c*x)
    # and the CDF is 1 - exp(-c*x).

    with mpmath.workdps(50):
        x = mpmath.mp.mpf('11.3')
        c = mpmath.mp.mpf('8.5')
        beta = mpmath.mp.one
        scale = mpmath.mp.one

        p = gamma_gompertz.pdf(x, c, beta, scale)
        expected = c * mpmath.exp(-c * x)
        assert mpmath.almosteq(p, expected)

        p = gamma_gompertz.cdf(x, c, beta, scale)
        expected = 1 - mpmath.exp(-c * x)
        assert mpmath.almosteq(p, expected)

        p = gamma_gompertz.sf(x, c, beta, scale)
        expected = mpmath.exp(-c * x)
        assert mpmath.almosteq(p, expected)


def test_sf_invsf_roundtrip():
    with mpmath.workdps(50):
        rel_eps = mpmath.mp.mpf('1e-48')

        x = mpmath.mp.mpf('13.5')
        p = gamma_gompertz.sf(x, 4, 3, 2)
        x1 = gamma_gompertz.invsf(p, 4, 3, 2)
        assert mpmath.almosteq(x1, x, rel_eps=rel_eps)

        p = mpmath.mp.mpf('2/3')
        x = gamma_gompertz.invsf(p, 5, 2, 3)
        p1 = gamma_gompertz.sf(x, 5, 2, 3)
        assert mpmath.almosteq(p1, p, rel_eps=rel_eps)


def test_cdf_invcdf_roundtrip():
    with mpmath.workdps(50):
        rel_eps = mpmath.mp.mpf('1e-48')

        x = mpmath.mp.mpf('13.5')
        p = gamma_gompertz.cdf(x, 4, 3, 2)
        x1 = gamma_gompertz.invcdf(p, 4, 3, 2)
        assert mpmath.almosteq(x1, x, rel_eps=rel_eps)

        p = mpmath.mp.mpf('2/3')
        x = gamma_gompertz.invcdf(p, 5, 2, 3)
        p1 = gamma_gompertz.cdf(x, 5, 2, 3)
        assert mpmath.almosteq(p1, p, rel_eps=rel_eps)
