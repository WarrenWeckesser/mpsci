from mpmath import mp
from mpsci.fun import marcumq, cmarcumq


def test_sum():
    # Test that marcumq(m, 0.5, 2) + cmarcumq(m, 0.5, 2) == 1.
    # The individual values are not checked.
    with mp.workdps(50):
        for m in [1, 10, 25, 100]:
            q1 = marcumq(m, 0.5, 2)
            q2 = cmarcumq(m, 0.5, 2)
            assert mp.almosteq(q1 + q2, 1)


def test_bounds_for_Q1():
    """
    These bounds are from:

        Jiangping Wang and Dapeng Wu,
        "Tight bounds for the first order Marcum Q-function"
        Wireless Communications and Mobile Computing,
        Volume 12, Issue 4, March 2012, Pages 293-301.

    """
    with mp.workdps(50):
        sqrt2 = mp.sqrt(2)
        sqrthalfpi = mp.sqrt(mp.pi/2)
        for b in [mp.mpf(1), mp.mpf(10)]:
            for a in [b/16, 0.5*b, 0.875*b, b]:
                q1 = marcumq(1, a, b)

                sinhab = mp.sinh(a*b)
                i0ab = mp.besseli(0, a*b)

                # lower bound when 0 <= a <= b
                lb = (mp.sqrt(mp.pi/8) * b * i0ab / sinhab *
                      (mp.erfc((b - a)/sqrt2)
                       - mp.erfc((b + a)/sqrt2)))
                assert lb < q1

                # upper bound when 0 <= a <= b
                ub = ((i0ab + 3)/(mp.exp(a*b) + 3) *
                      (mp.exp(-(b - a)**2/2)
                       + a*sqrthalfpi*mp.erfc((b - a)/sqrt2)
                       + 3*mp.exp(-(a**2 + b**2)/2)))
                assert q1 < ub, f"marcumq(1, a, b) < ub for {a = },  {b = }"

        for a in [mp.mpf(1), mp.mpf(10)]:
            for b in [b/16, 0.5*b, 0.875*b, b]:
                q1 = marcumq(1, a, b)

                sinhab = mp.sinh(a*b)
                i0ab = mp.besseli(0, a*b)

                # lower bound when 0 <= b <= a
                lb = (1 - sqrthalfpi * b * i0ab / sinhab *
                          (mp.erf(a/sqrt2)
                           - mp.erf((a - b)/sqrt2)/2
                           - mp.erf((a + b)/sqrt2)/2))
                assert lb < q1

                # upper bound when 0 <= b <= a
                ub = (1 - i0ab/(mp.exp(a*b) + 3) *
                          (4*mp.exp(-a**2/2)
                           - mp.exp(-(b - a)**2/2)
                           - 3*mp.exp(-(a**2 + b**2)/2)
                           + (a*sqrthalfpi *
                              (mp.erfc(-a/sqrt2) - mp.erfc((b - a)/sqrt2)))))
                assert q1 < ub, f"marcumq(1, a, b) < ub for {a = },  {b = }"


def test_Q1_identities():
    with mp.workdps(50):
        for a in [mp.mpf(1), mp.mpf(13)]:
            q1 = marcumq(1, a, a)
            expected = 0.5*(1 + mp.exp(-a**2)*mp.besseli(0, a**2))
            assert mp.almosteq(q1, expected)

        for a, b in [(1, 2), (10, 3.5)]:
            total = marcumq(1, a, b) + marcumq(1, b, a)
            expected = 1 + mp.exp(-(a**2 + b**2)/2)*mp.besseli(0, a*b)
            assert mp.almosteq(total, expected)


def test_specific_values_from_wolfram():
    # The function is MarcumQ[m, a, b] in Wolfram.
    with mp.workdps(50):
        values = [
            [(1, 1, 2),
             '0.269012060035909996678516959220271087421337500744873384155'],
            [(4, 2, 2),
             '0.961002639616864002974959310625974743642314730388958932865'],
            [(10, 2, 7),
             '0.003419880268711142942584170929968429599523954396941827739'],
            [(10, 2, 20),
             '1.1463968305097683198312254572377449552982286967700350e-63'],
            [(25, 3, 1),
             '0.999999999999999999999999999999999985610186498600062604510'],
            [(100, 5, 25),
             '6.3182094741234192918553754494227940871375208133861094e-36'],
        ]
        for (m, a, b), expected in values:
            q = marcumq(m, a, b)
            assert mp.almosteq(q, mp.mpf(expected))

        values = [
            [(0.5, 1, 2),
             '0.839994848036912854058580730864442945100083598568223741623'],
            [(0.5, 5, 13),
             '0.99999999999999937790394257282158764840048274118115775113'],
            [(0.5, 5, mp.mpf('1/13')),
             '2.3417168332795098320214996797055274037932802604709459e-7'],
            [(0.75, 5, mp.mpf('0.001')),
             '7.6243509472783061143887290663443105556790864676878435e-11'],
            [(51/2, 5, mp.mpf('0.1')),
             '9.9527768099307883918427141035705197017192239618506806e-91']
        ]
        for (m, a, b), expected in values:
            c = cmarcumq(m, a, b)
            assert mp.almosteq(c, mp.mpf(expected))
