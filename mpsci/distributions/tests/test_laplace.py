
from mpmath import mp
from mpsci.distributions import laplace


def test_mle():
    with mp.workdps(40):
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
