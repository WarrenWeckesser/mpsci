
from mpmath import mp
from mpsci.distributions import genhyperbolic as gh
from ._expect import noncentral_moment_with_integral


@mp.workdps(40)
def test_pdf_is_normalized():
    p = 0.25
    a = 2.5
    b = 0.75
    loc = 0
    scale = 5
    m = gh.mean(p, a, b, loc=loc, scale=scale)
    q = mp.quad(lambda t: gh.pdf(t, p, a, b, loc=loc, scale=scale),
                [mp.ninf, m, mp.inf])
    assert mp.almosteq(q, 1)


@mp.workdps(40)
def test_mean_var_with_integral():
    p = 0.25
    a = 2.5
    b = 0.75
    loc = 0
    scale = 5

    m = gh.mean(p, a, b, loc=loc, scale=scale)
    mu1 = noncentral_moment_with_integral(1, gh, (p, a, b, loc, scale))
    assert mp.almosteq(m, mu1)

    v = gh.var(p, a, b, loc=loc, scale=scale)
    mu2 = noncentral_moment_with_integral(2, gh, (p, a, b, loc, scale))
    assert mp.almosteq(v, mu2 - mu1**2)
