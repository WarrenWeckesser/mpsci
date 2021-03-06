
import mpmath
from mpsci.distributions import multivariate_t


# Some values from Wolfram Alpha
# PDF[MultivariateTDistribution[{{1, 0},{0,3}}, 3], {3, 5}]
#     = (81 sqrt(3/61))/(7442 π)
#     ≈ 0.0007683183228324574166316130914574759225242355775144734517564...
# PDF[MultivariateTDistribution[{{1, 1/10},{1/10,3}}, 3], {0, 0}]
#     = 5/(sqrt(299) π)
#     ≈ 0.0920416800862802853500744637173704102495845924002081767282224...
# PDF[MultivariateTDistribution[{{1,1/10,1/25}, {1/10,3,0}, {1/25,0,2}}, 9],
#                               {0,0,0}]
#     = (3200 sqrt(2/7469))/(189 π^2)
#     ≈ 0.0280719251273608075819567901706183442574259902571887726816305...

mpmath.mp.dps = 100


def test_pdf2a():
    A = mpmath.matrix(2)
    A[0, 0] = mpmath.mpf(1)
    A[1, 1] = mpmath.mpf(3)
    x = [mpmath.mpf(3), mpmath.mpf(5)]
    df = 3
    p = multivariate_t.pdf(x, df, [0, 0], A)
    valstr = '0.0007683183228324574166316130914574759225242355775144734517564'
    expected = mpmath.mpf(valstr)
    assert mpmath.almosteq(p, expected)


def test_pdf2b():
    A = mpmath.matrix(2)
    A[0, 0] = mpmath.mpf(1)
    A[0, 1] = mpmath.mpf('0.1')
    A[1, 0] = A[0, 1]
    A[1, 1] = mpmath.mpf(3)
    x = [0, 0]
    df = 3
    p = multivariate_t.pdf(x, df, [0, 0], A)
    valstr = '0.0920416800862802853500744637173704102495845924002081767282224'
    expected = mpmath.mpf(valstr)
    assert mpmath.almosteq(p, expected)


def test_pdf3():
    A = mpmath.matrix(3)
    A[0, 0] = mpmath.mpf(1)
    A[0, 1] = mpmath.mpf('0.1')
    A[1, 0] = A[0, 1]
    A[0, 2] = mpmath.mpf('0.04')
    A[2, 0] = A[0, 2]
    A[1, 1] = mpmath.mpf(3)
    A[1, 2] = 0
    A[2, 1] = 0
    A[2, 2] = 2
    x = [0, 0, 0]
    df = 9
    p = multivariate_t.pdf(x, df, [0, 0, 0], A)
    valstr = '0.0280719251273608075819567901706183442574259902571887726816305'
    expected = mpmath.mpf(valstr)
    assert mpmath.almosteq(p, expected)
