
import mpmath
from mpsci.distributions import benktander1


def test_pdf():
    with mpmath.workdps(50):
        x = mpmath.mpf('1.5')
        p = benktander1.pdf(x, 2, 3)
        # Expected value computed with Wolfram Alpha:
        #    PDF[BenktanderGibratDistribution[2, 3], 3/2]
        valstr = '1.090598817302604549131682068809802266147250025484891499295'
        expected = mpmath.mpf(valstr)
        assert mpmath.almosteq(p, expected)


def test_logpdf():
    with mpmath.workdps(50):
        x = mpmath.mpf('1.5')
        p = benktander1.logpdf(x, 2, 3)
        # Expected value computed with Wolfram Alpha:
        #    log(PDF[BenktanderGibratDistribution[2, 3], 3/2])
        valstr = '0.086726919062697113736142804022160705324241157062981346304'
        expected = mpmath.mpf(valstr)
        assert mpmath.almosteq(p, expected)


def test_cdf_invcdf():
    with mpmath.workdps(50):
        x = mpmath.mpf('1.5')
        p = benktander1.cdf(x, 2, 3)
        # Expected value computed with Wolfram Alpha:
        #    CDF[BenktanderGibratDistribution[2, 3], 3/2]
        valstr = '0.59896999842391210365289674809988804989249935760023852777'
        expected = mpmath.mpf(valstr)
        assert mpmath.almosteq(p, expected)
        x1 = benktander1.invcdf(expected, 2, 3)
        assert mpmath.almosteq(x1, x)


def test_sf_invsf():
    with mpmath.workdps(50):
        x = mpmath.mpf('1.5')
        p = benktander1.sf(x, 2, 3)
        # Expected value computed with Wolfram Alpha:
        #    SurvivalFunction[BenktanderGibratDistribution[2, 3], 3/2]
        valstr = '0.40103000157608789634710325190011195010750064239976147223'
        expected = mpmath.mpf(valstr)
        assert mpmath.almosteq(p, expected)
        x1 = benktander1.invsf(expected, 2, 3)
        assert mpmath.almosteq(x1, x)


def test_mean():
    with mpmath.workdps(50):
        a = 2
        b = 3
        m = benktander1.mean(a, b)
        assert mpmath.almosteq(m, mpmath.mpf('1.5'))


def test_var():
    with mpmath.workdps(50):
        a = 2
        b = 3
        m = benktander1.var(a, b)
        # Expected value computed with Wolfram Alpha:
        #    Var[BenktanderGibratDistribution[2, 3]]
        valstr = '0.129886916731278610514259475545032373691162070980680465530'
        expected = mpmath.mpf(valstr)
        assert mpmath.almosteq(m, expected)
