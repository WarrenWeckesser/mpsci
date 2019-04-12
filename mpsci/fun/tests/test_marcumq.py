
import mpmath
from mpsci.fun import marcumq, cmarcumq


def test_sum():
    with mpmath.workdps(50):
        for m in [1, 10, 25, 100]:
            q1 = marcumq(m, 0.5, 2)
            q2 = cmarcumq(m, 0.5, 2)
            assert mpmath.almosteq(q1 + q2, 1)


def test_bounds_for_Q1():
    """
    These bounds are from:

        Jiangping Wang and Dapeng Wu,
        "Tight bounds for the first order Marcum Q-function"
        Wireless Communications and Mobile Computing,
        Volume 12, Issue 4, March 2012, Pages 293-301.

    """
    with mpmath.workdps(50):
        sqrt2 = mpmath.sqrt(2)
        sqrthalfpi = mpmath.sqrt(mpmath.pi/2)
        for b in [mpmath.mp.mpf(1), mpmath.mp.mpf(10)]:
            for a in [b/16, 0.5*b, 0.875*b, b]:
                q1 = marcumq(1, a, b)

                sinhab = mpmath.sinh(a*b)
                i0ab = mpmath.besseli(0, a*b)

                # lower bound when 0 <= a <= b
                lb = (mpmath.sqrt(mpmath.pi/8) * b * i0ab / sinhab *
                      (mpmath.erfc((b - a)/sqrt2)
                       - mpmath.erfc((b + a)/sqrt2)))
                assert lb < q1

                # upper bound when 0 <= a <= b
                ub = ((i0ab + 3)/(mpmath.exp(a*b) + 3) *
                      (mpmath.exp(-(b - a)**2/2)
                       + a*sqrthalfpi*mpmath.erfc((b - a)/sqrt2)
                       + 3*mpmath.exp(-(a**2 + b**2)/2)))
                assert q1 < ub, ("marcumq(1, a, b) < ub for a = %s,  b = %s" %
                                 (a, b))

        for a in [mpmath.mp.mpf(1), mpmath.mp.mpf(10)]:
            for b in [b/16, 0.5*b, 0.875*b, b]:
                q1 = marcumq(1, a, b)

                sinhab = mpmath.sinh(a*b)
                i0ab = mpmath.besseli(0, a*b)

                # lower bound when 0 <= b <= a
                lb = (1 - sqrthalfpi * b * i0ab / sinhab *
                          (mpmath.erf(a/sqrt2)
                           - mpmath.erf((a - b)/sqrt2)/2
                           - mpmath.erf((a + b)/sqrt2)/2))
                assert lb < q1

                # upper bound when 0 <= b <= a
                ub = (1 - i0ab/(mpmath.exp(a*b) + 3) *
                          (4*mpmath.exp(-a**2/2)
                           - mpmath.exp(-(b - a)**2/2)
                           - 3*mpmath.exp(-(a**2 + b**2)/2)
                           + a*sqrthalfpi *
                             (mpmath.erfc(-a/sqrt2)
                              - mpmath.erfc((b - a)/sqrt2))))
                assert q1 < ub, ("marcumq(1, a, b) < ub for a = %s,  b = %s" %
                                 (a, b))


def test_Q1_identities():
    with mpmath.workdps(50):
        for a in [mpmath.mp.mpf(1), mpmath.mp.mpf(13)]:
            q1 = marcumq(1, a, a)
            expected = 0.5*(1 + mpmath.exp(-a**2)*mpmath.besseli(0, a**2))
            assert mpmath.almosteq(q1, expected)

        for a, b in [(1, 2), (10, 3.5)]:
            total = marcumq(1, a, b) + marcumq(1, b, a)
            expected = 1 + mpmath.exp(-(a**2 + b**2)/2)*mpmath.besseli(0, a*b)
            assert mpmath.almosteq(total, expected)


def test_specific_values_from_wolfram():
    # The function is MarcumQ[m, a, b] in Wolfram.
    with mpmath.workdps(50):
        values = [
            [(1, 1, 2),
             mpmath.mpf('0.269012060035909996678516959220271087421337500744873384155')],
            [(4, 2, 2),
             mpmath.mpf('0.961002639616864002974959310625974743642314730388958932865')],
            [(10, 2, 7),
             mpmath.mpf('0.003419880268711142942584170929968429599523954396941827739')],
            [(10, 2, 20),
             mpmath.mpf('1.1463968305097683198312254572377449552982286967700350e-63')],
            [(25, 3, 1),
             mpmath.mpf('0.999999999999999999999999999999999985610186498600062604510')],
            [(100, 5, 25),
             mpmath.mpf('6.3182094741234192918553754494227940871375208133861094e-36')],
        ]
        for (m, a, b), expected in values:
            q = marcumq(m, a, b)
            assert mpmath.almosteq(q, expected)
