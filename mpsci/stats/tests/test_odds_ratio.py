import math
from mpmath import mp
from mpsci.stats import odds_ratio


@mp.workdps(40)
def test_odds_ratio_with_scipy():
    # Test that the function call works, and that the
    # odds ratio is correct to 4 significant digits.
    # TODO: Create a test with higher precision required.
    table = [[10, 11], [37, 8]]
    result = odds_ratio(table)
    # Reference value computed with scipy.stats.contingency.odds_ratio.
    assert math.isclose(float(round(result.odds_ratio, 4)), 0.2024, rel_tol=5e-5)


@mp.workdps(40)
def test_odds_ratio_with_r_fisher_test():
    # Test that the odds ratio and confidence interval agree with
    # R's fisher.test, allowing a relative error of 2e-4.
    table = [[7, 11], [6, 21]]

    #
    # Reference values computed with the R function fisher.test:
    #
    #    > m <- matrix(c(7, 6, 11, 21), nrow=2)
    #    > result = fisher.test(m)
    #    > result
    #
    #            Fisher's Exact Test for Count Data
    #
    #    data:  m
    #    p-value = 0.3172
    #    alternative hypothesis: true odds ratio is not equal to 1
    #    95 percent confidence interval:
    #      0.4920595 10.1444865
    #    sample estimates:
    #    odds ratio
    #      2.186158
    #
    result = odds_ratio(table)
    assert mp.almosteq(result.odds_ratio, 2.186158, rel_eps=2e-4)
    # result.pvalue is a Fraction.
    assert mp.almosteq(float(result.pvalue), 0.3172, rel_eps=2e-4)
    ci = result.odds_ratio_ci(0.95)
    assert mp.almosteq(ci.low, 0.4920595, rel_eps=2e-4)
    assert mp.almosteq(ci.high, 10.1444865, rel_eps=2e-4)

    #
    # Reference values computed with the R function fisher.test:
    #
    #    > m <- matrix(c(7, 6, 11, 21), nrow=2)
    #    > result = fisher.test(m, alternative='less')
    #    > result
    #
    #            Fisher's Exact Test for Count Data
    #
    #    data:  m
    #    p-value = 0.938
    #    alternative hypothesis: true odds ratio is less than 1
    #    95 percent confidence interval:
    #     0.000000 8.137102
    #    sample estimates:
    #    odds ratio
    #      2.186158
    #
    result = odds_ratio(table, alternative='less')
    assert mp.almosteq(result.odds_ratio, 2.186158, rel_eps=2e-4)
    # result.pvalue is a Fraction.
    assert mp.almosteq(float(result.pvalue), 0.938, rel_eps=2e-4)
    ci = result.odds_ratio_ci(0.95)
    assert ci.low == 0
    assert mp.almosteq(ci.high, 8.137102, rel_eps=2e-4)

    #
    # Reference values computed with the R function fisher.test:
    #
    #    > m <- matrix(c(7, 6, 11, 21), nrow=2)
    #    > result = fisher.test(m, alternative='greater')
    #    > result
    #
    #            Fisher's Exact Test for Count Data
    #
    #    data:  m
    #    p-value = 0.191
    #    alternative hypothesis: true odds ratio is greater than 1
    #    95 percent confidence interval:
    #     0.6057588       Inf
    #    sample estimates:
    #    odds ratio
    #      2.186158
    #
    result = odds_ratio(table, alternative='greater')
    assert mp.almosteq(result.odds_ratio, 2.186158, rel_eps=2e-4)
    # result.pvalue is a Fraction.
    assert mp.almosteq(float(result.pvalue), 0.191, rel_eps=2e-4)
    ci = result.odds_ratio_ci(0.95)
    assert mp.almosteq(ci.low, 0.6057588, rel_eps=2e-4)
    assert ci.high == mp.inf
