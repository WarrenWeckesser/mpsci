import pytest
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


@pytest.mark.parametrize('colors, nsample',
                         [([10, 10, 25, 25, 10], 16),
                          ([5, 6, 7], 5),
                          ([10, 0, 3, 1], 4)])
@mp.workdps(50)
def test_cov(colors, nsample):
    cov = mp.matrix(multivariate_hypergeometric.cov(colors, nsample))

    n = len(colors)
    sup = list(multivariate_hypergeometric.support(colors, nsample))
    pmf = [multivariate_hypergeometric.pmf(x, colors, nsample) for x in sup]
    mu = [mean([x[i] for x in sup], weights=pmf) for i in range(len(colors))]
    for i in range(n):
        for j in range(i, n):
            prods = [(x[i] - mu[i]) * (x[j] - mu[j])for x in sup]
            m = mean(prods, weights=pmf)
            assert mp.almosteq(cov[i, j], m)
