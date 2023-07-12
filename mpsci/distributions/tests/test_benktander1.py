
import pytest
from mpmath import mp
from mpsci.distributions import benktander1
from ._utils import check_mle


def test_pdf():
    with mp.workdps(50):
        x = mp.mpf('1.5')
        p = benktander1.pdf(x, 2, 3)
        # Expected value computed with Wolfram Alpha:
        #    PDF[BenktanderGibratDistribution[2, 3], 3/2]
        valstr = '1.090598817302604549131682068809802266147250025484891499295'
        expected = mp.mpf(valstr)
        assert mp.almosteq(p, expected)


def test_logpdf():
    with mp.workdps(50):
        x = mp.mpf('1.5')
        p = benktander1.logpdf(x, 2, 3)
        # Expected value computed with Wolfram Alpha:
        #    log(PDF[BenktanderGibratDistribution[2, 3], 3/2])
        valstr = '0.086726919062697113736142804022160705324241157062981346304'
        expected = mp.mpf(valstr)
        assert mp.almosteq(p, expected)


def test_cdf_invcdf():
    with mp.workdps(50):
        x = mp.mpf('1.5')
        p = benktander1.cdf(x, 2, 3)
        # Expected value computed with Wolfram Alpha:
        #    CDF[BenktanderGibratDistribution[2, 3], 3/2]
        valstr = '0.59896999842391210365289674809988804989249935760023852777'
        expected = mp.mpf(valstr)
        assert mp.almosteq(p, expected)
        x1 = benktander1.invcdf(expected, 2, 3)
        assert mp.almosteq(x1, x)


def test_sf_invsf():
    with mp.workdps(50):
        x = mp.mpf('1.5')
        p = benktander1.sf(x, 2, 3)
        # Expected value computed with Wolfram Alpha:
        #    SurvivalFunction[BenktanderGibratDistribution[2, 3], 3/2]
        valstr = '0.40103000157608789634710325190011195010750064239976147223'
        expected = mp.mpf(valstr)
        assert mp.almosteq(p, expected)
        x1 = benktander1.invsf(expected, 2, 3)
        assert mp.almosteq(x1, x)


@pytest.mark.parametrize('p, expected', [(0, 1), (1, 'inf')])
def test_invcdf_invsf_bounds(p, expected):
    with mp.workdps(50):
        x = benktander1.invcdf(p, 2, 3)
        assert x == mp.mpf(expected)
        x = benktander1.invsf(1 - p, 2, 3)
        assert x == mp.mpf(expected)


def test_mean():
    with mp.workdps(50):
        a = 2
        b = 3
        m = benktander1.mean(a, b)
        assert mp.almosteq(m, mp.mpf('1.5'))


def test_var():
    with mp.workdps(50):
        a = 2
        b = 3
        m = benktander1.var(a, b)
        # Expected value computed with Wolfram Alpha:
        #    Var[BenktanderGibratDistribution[2, 3]]
        valstr = '0.129886916731278610514259475545032373691162070980680465530'
        expected = mp.mpf(valstr)
        assert mp.almosteq(m, expected)


@pytest.mark.parametrize(
    'x',
    [[1.3, 1.5, 2.8, 1.7, 2.4, 6.6, 2.5, 1.4, 1.8, 1.8, 2.0, 2.3, 1.2,
      7.8, 38.3, 1.4, 1.4, 1.3, 10.5, 1.3, 1.3, 2.2, 8.6, 1.3, 1.6],
     [1.58, 1.20, 1.13, 1.86, 1.15, 1.30, 1.06, 1.07, 1.07, 1.63,
      1.67, 1.25, 1.01, 1.37, 1.01, 1.57, 1.23, 1.53, 1.03, 1.06],
     [1.014, 1.035, 1.122, 1.043, 1.025, 1.049,
      1.002, 1.008, 1.007, 1.019, 1.020, 1.044]],
)
@mp.workdps(50)
def test_mle(x):
    # benktander1.mle() is not reliable--it typically fails--but for the
    # data sets in this test, it works.
    p = benktander1.mle(x)
    check_mle(benktander1.nll, x, p)
