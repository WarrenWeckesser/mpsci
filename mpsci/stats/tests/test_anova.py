from mpmath import mp
from mpsci.stats import anova_oneway


@mp.workdps(20)
def test_anova_oneway_against_R():
    """
    The reference values for this test were computed with the aov() function
    in R:

        x <- c(5, 6, 7, 4, 5, 6, 3, 5, 7)
        g <- factor(c(1, 1, 1, 2, 2, 2, 3, 3, 3))
        result <- aov(x ~ g)
        options(digits=15)
        print(summary(result)[[1]][1, "F value"])
        print(summary(result)[[1]][1, "Pr(>F)"])

    Output:

        [1] 0.5
        [1] 0.629737609329447
    """
    a = [5, 6, 7]
    b = [4, 5, 6]
    c = [3, 5, 7]
    result = anova_oneway(a, b, c)
    assert result.f == 0.5
    assert mp.almosteq(result.p, 0.629737609329447, rel_eps=1e-14)
