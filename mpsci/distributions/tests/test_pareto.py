from mpmath import mp
from mpsci.distributions import pareto
from ._utils import check_mle


@mp.workdps(40)
def test_pdf_loc0():
    b = 5
    scale = 3
    x = 9
    p = pareto.pdf(x, b=b, scale=scale)
    # From Wolfram Alpha:
    #   PDF[ParetoDistribution[3, 5], 9]
    expected = mp.mpf('5/2187')
    assert mp.almosteq(p, expected)


@mp.workdps(40)
def test_logpdf_loc0():
    b = 5
    scale = 3
    x = 9
    logp = pareto.logpdf(x, b=b, scale=scale)
    # From Wolfram Alpha:
    #   PDF[ParetoDistribution[3, 5], 9]
    expected = mp.log(mp.mpf('5/2187'))
    assert mp.almosteq(logp, expected)


@mp.workdps(40)
def test_cdf_loc0():
    b = 5
    scale = 3
    x = 9
    cdf = pareto.cdf(x, b=b, scale=scale)
    # From Wolfram Alpha:
    #   CDF[ParetoDistribution[3, 5], 9]
    expected = mp.mpf('242/243')
    assert mp.almosteq(cdf, expected)


@mp.workdps(40)
def test_invcdf_loc0():
    b = 5
    scale = 3
    p = mp.mpf('242/243')
    x = pareto.invcdf(p, b=b, scale=scale)
    # From Wolfram Alpha:
    #   CDF[ParetoDistribution[3, 5], 9]
    expected = mp.mpf('9')
    assert mp.almosteq(x, expected)


@mp.workdps(40)
def test_sf_loc0():
    b = 5
    scale = 3
    x = 9
    sf = pareto.sf(x, b=b, scale=scale)
    # From Wolfram Alpha:
    #   1 - CDF[ParetoDistribution[3, 5], 9]
    expected = mp.mpf('1/243')
    assert mp.almosteq(sf, expected)


@mp.workdps(40)
def test_invsf_loc0():
    b = 5
    scale = 3
    p = mp.mpf('1/243')
    x = pareto.invsf(p, b=b, scale=scale)
    # From Wolfram Alpha:
    #   CDF[ParetoDistribution[3, 5], 9]
    expected = mp.mpf('9')
    assert mp.almosteq(x, expected)


@mp.workdps(40)
def test_mean_loc0():
    b = 5
    scale = 3
    m = pareto.mean(b=b, loc=0, scale=scale)
    # From Wolfram Alpha:
    #   Mean[ParetoDistribution[3, 5]]
    expected = mp.mpf('15/4')
    assert mp.almosteq(m, expected)


@mp.workdps(40)
def test_var_loc0():
    b = 5
    scale = 3
    m = pareto.var(b=b, loc=0, scale=scale)
    # From Wolfram Alpha:
    #   Variance[ParetoDistribution[3, 5]]
    expected = mp.mpf('15/16')
    assert mp.almosteq(m, expected)


@mp.workdps(50)
def test_entropy_with_integral():
    b = 5
    loc = -1
    scale = 3.25
    entr = pareto.entropy(b, loc=loc, scale=scale)

    with mp.extradps(mp.dps):

        def integrand(t):
            return (pareto.logpdf(t, b, loc, scale) *
                    pareto.pdf(t, b, loc, scale))

        intgrl = -mp.quad(integrand, [loc + scale, mp.inf])

    assert mp.almosteq(entr, intgrl)


def test_mle_fixed_loc():
    x = [1.25, 1.5, 1.5, 7]
    b_hat, loc, scale_hat = pareto.mle(x, loc=0)
    assert loc == 0
    # Use check_mle() for the b parameter only.
    check_mle(lambda x, b: pareto.nll(x, b=b, scale=scale_hat),
              x, (b_hat,))
    assert scale_hat == min(x)
