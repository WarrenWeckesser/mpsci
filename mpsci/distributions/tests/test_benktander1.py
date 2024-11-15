
import pytest
from mpmath import mp
from mpsci.distributions import benktander1, Initial
from ._utils import check_mle


@pytest.mark.parametrize('a', [-1, 0])
def test_validate_a_positive(a):
    with pytest.raises(ValueError, match='must be positive'):
        benktander1._validate_ab(a, 1)


@pytest.mark.parametrize('b', [-1, 0])
def test_validate_b_positive(b):
    with pytest.raises(ValueError, match='must be positive'):
        benktander1._validate_ab(5, b)


def test_validate_ab_values():
    with pytest.raises(ValueError, match='must not be greater than'):
        benktander1._validate_ab(3, 10)


@mp.workdps(25)
def test_support():
    # We haven't included a location parameter in the implementation,
    # so the support is always [1, inf).
    s = benktander1.support(2, 1)
    assert s == (mp.one, mp.inf)


@mp.workdps(25)
def test_pdf_outside_support():
    p = benktander1.pdf(0.5, 5, 11)
    assert p == mp.zero


@mp.workdps(25)
def test_logpdf_outside_support():
    p = benktander1.logpdf(0.5, 5, 11)
    assert p == mp.ninf


@mp.workdps(50)
def test_pdf():
    x = mp.mpf('1.5')
    p = benktander1.pdf(x, 2, 3)
    # Expected value computed with Wolfram Alpha:
    #    PDF[BenktanderGibratDistribution[2, 3], 3/2]
    valstr = '1.090598817302604549131682068809802266147250025484891499295'
    expected = mp.mpf(valstr)
    assert mp.almosteq(p, expected)


@mp.workdps(50)
def test_logpdf():
    x = mp.mpf('1.5')
    p = benktander1.logpdf(x, 2, 3)
    # Expected value computed with Wolfram Alpha:
    #    log(PDF[BenktanderGibratDistribution[2, 3], 3/2])
    valstr = '0.086726919062697113736142804022160705324241157062981346304'
    expected = mp.mpf(valstr)
    assert mp.almosteq(p, expected)


@mp.workdps(25)
def test_cdf_outside_support():
    p = benktander1.cdf(0.5, 5, 11)
    assert p == mp.zero


@mp.workdps(50)
def test_cdf_invcdf():
    x = mp.mpf('1.5')
    p = benktander1.cdf(x, 2, 3)
    # Expected value computed with Wolfram Alpha:
    #    CDF[BenktanderGibratDistribution[2, 3], 3/2]
    valstr = '0.59896999842391210365289674809988804989249935760023852777'
    expected = mp.mpf(valstr)
    assert mp.almosteq(p, expected)
    x1 = benktander1.invcdf(expected, 2, 3)
    assert mp.almosteq(x1, x)


@mp.workdps(25)
def test_sf_outside_support():
    p = benktander1.sf(0.5, 5, 11)
    assert p == mp.one


@mp.workdps(50)
def test_sf_invsf():
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
@mp.workdps(50)
def test_invcdf_invsf_bounds(p, expected):
    x = benktander1.invcdf(p, 2, 3)
    assert x == mp.mpf(expected)
    x = benktander1.invsf(1 - p, 2, 3)
    assert x == mp.mpf(expected)


@mp.workdps(50)
def test_mean():
    a = 2
    b = 3
    m = benktander1.mean(a, b)
    assert mp.almosteq(m, mp.mpf('1.5'))


@mp.workdps(50)
def test_var():
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
      1.002, 1.008, 1.007, 1.019, 1.020, 1.044],
     [1.072, 1.564, 1.063, 2.435, 1.734, 2.412, 1.727, 1.73, 1.847, 1.536,
      11.625, 1.297, 2.068, 1.4, 1.306, 1.016, 1.205, 1.429, 8.622, 2.57,
      1.995, 1.138, 1.623, 1.513, 1.339, 1.149, 2.108, 1.138, 1.878, 1.022,
      2.491, 1.109, 3.394, 1.295, 1.489, 1.143, 1.565, 1.13, 1.305, 1.156,
      5.385, 3.254, 2.244, 2.874, 1.605, 1.201, 2.486, 1.853, 1.536, 1.309]]
)
@mp.workdps(50)
def test_mle(x):
    # benktander1.mle() is not reliable--it typically fails--but for the
    # data sets in this test, it works.
    p = benktander1.mle(x)
    check_mle(benktander1.nll, x, p)


@mp.workdps(50)
def test_mle_a_initial():
    x = [1.23, 1.12, 1.06, 1.04, 1.59, 1.13, 1.05, 1.56, 1.16, 1.02]
    p = benktander1.mle(x, a=Initial(5))
    check_mle(benktander1.nll, x, p)


@mp.workdps(50)
def test_mle_b_initial():
    x = [2.288, 1.331, 1.806, 1.170, 1.084, 1.203, 2.641, 1.640, 1.364, 1.495,
         1.107, 1.119, 1.192, 1.918, 1.708, 1.014, 1.009, 1.255, 1.966, 1.883,
         1.295, 1.185, 1.179, 1.013, 1.039, 1.796, 1.652, 1.335, 1.368, 1.784]
    p = benktander1.mle(x, b=Initial(1.5))
    check_mle(benktander1.nll, x, p)
