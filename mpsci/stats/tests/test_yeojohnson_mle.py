import pytest
from mpmath import mp
from mpsci.stats import yeojohnson_llf, yeojohnson_mle


def test_empty_data():
    with pytest.raises(ValueError, match='must have at least one element'):
        yeojohnson_llf(2, [])


@pytest.mark.parametrize('x, lam0',
                         [([1, 2, 3, 5, 8, 13], -0.1),
                          ([2003, 1950, 1997, 2000, 2009, 2009,
                            1980, 1999, 2007, 1991], 100)])
@mp.workdps(60)
def test_yeojohnson_mle(x, lam0):
    # This is a relativity crude test. It checks that values of
    # `yeojohnson_llf` near the computed optimal value are less
    # than the log-likelihood at the computed value.
    # The code assumes that the computed optimal value is not 0.
    lam = yeojohnson_mle(x, lam0)
    llf = yeojohnson_llf(lam, x)

    lam1 = 1.00001*lam
    llf1 = yeojohnson_llf(lam1, x)
    assert llf1 < llf

    lam2 = 0.99999*lam
    llf2 = yeojohnson_llf(lam2, x)
    assert llf2 < llf
