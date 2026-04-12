from mpmath import mp
from mpsci.distributions import multivariate_hypergeometric
from mpsci.stats import mean


@mp.workdps(50)
def test_mean():
    colors = [10, 10, 25, 25, 10]
    nsample = 16
    m = multivariate_hypergeometric.mean(colors, nsample)

    sup = list(multivariate_hypergeometric.support(colors, nsample))
    pmf = [multivariate_hypergeometric.pmf(x, colors, nsample) for x in sup]
    ref = [mean([x[i] for x in sup], weights=pmf) for i in range(len(colors))]

    assert all(mp.almosteq(m[i], ref[i]) for i in range(len(colors)))
