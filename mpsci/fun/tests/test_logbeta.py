
from mpmath import mp
from mpsci.fun import logbeta, multivariate_logbeta


def test_logbeta_basic():
    with mp.workdps(50):
        lnb = logbeta(250, 350)
        expected = mp.mpf('-409.087820730822707552263110'
                          '5941596488100197539984096')
        assert mp.almosteq(lnb, expected)


def test_multivariate_logbeta_basic():
    with mp.workdps(50):
        alpha = [1.0, 2.0, 5.0, 0.5]
        result = multivariate_logbeta(alpha)
        expected = (mp.fsum([mp.loggamma(t) for t in alpha]
                            + [-mp.loggamma(mp.fsum(alpha))]))
        assert mp.almosteq(result, expected)
