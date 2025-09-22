
from mpmath import mp
from mpsci.distributions import laplace


@mp.workdps(40)
def test_mle():
    x = [1, 2, 4, 4, 6, 8, 8, 9]
    mu_hat, scale_hat = laplace.mle(x)
    # The MLE for mu is the median.
    assert mu_hat == 5
    # The MLE for scale is the mean absolute deviation from the median:
    # scale_hat = mean(|1-5|, |2-5|, |4-5|, |4-5|, |6-5|,
    #                  |8-5|, |8-5|, |9-5|)
    #           = mean(4, 3, 1, 1, 1, 3, 3, 4)
    #           = 20/8 = 2.5
    assert scale_hat == 2.5


@mp.workdps(50)
def test_interval_prob():
    x1 = -1001
    x2 = -1000
    p = laplace.interval_prob(x1, x2)
    # s computed with Wolfram Alpha:
    #   CDF[LaplaceDistribution[0, 1], -1000] - CDF[LaplaceDistribution[0, 1], -1001]
    s = "1.6043089874548760814516280731829919474907824618811554928501e-435"
    ref = mp.mpf(s)
    assert mp.almosteq(p, ref)

    # Reuse above result, by symmetry.
    p = laplace.interval_prob(-x2, -x1)
    assert mp.almosteq(p, ref)

    x1 = -mp.mpf(1)/10000000
    x2 = mp.mpf(2)/10000000
    p = laplace.interval_prob(x1, x2)
    # Wolfram Alpha:
    #    CDF[LaplaceDistribution[0, 1], 2/10000000]
    #      - CDF[LaplaceDistribution[0, 1], -1/10000000]
    s = "1.4999998750000074999996458333470833328819444572420631733631023e-7"
    ref = mp.mpf(s)
    assert mp.almosteq(p, ref)
