import pytest
from mpmath import mp
from mpsci.fun import marcumq, cmarcumq


@mp.workdps(50)
def test_sum():
    # Test that marcumq(m, 0.5, 2) + cmarcumq(m, 0.5, 2) == 1.
    # The individual values are not checked.
    for m in [1, 10, 25, 100]:
        q1 = marcumq(m, 0.5, 2)
        q2 = cmarcumq(m, 0.5, 2)
        assert mp.almosteq(q1 + q2, 1)


@mp.workdps(50)
def test_bounds_for_Q1():
    """
    These bounds are from:

        Jiangping Wang and Dapeng Wu,
        "Tight bounds for the first order Marcum Q-function"
        Wireless Communications and Mobile Computing,
        Volume 12, Issue 4, March 2012, Pages 293-301.

    """
    sqrt2 = mp.sqrt(2)
    sqrthalfpi = mp.sqrt(mp.pi/2)
    for b in [mp.mpf(1), mp.mpf(10)]:
        for a in [b/16, 0.5*b, 0.875*b, b]:
            q1 = marcumq(1, a, b)

            sinhab = mp.sinh(a*b)
            i0ab = mp.besseli(0, a*b)

            # lower bound when 0 <= a <= b
            lb = (mp.sqrt(mp.pi/8) * b * i0ab / sinhab *
                  (mp.erfc((b - a)/sqrt2) - mp.erfc((b + a)/sqrt2)))
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


@mp.workdps(50)
def test_Q1_identities():
    for a in [mp.mpf(1), mp.mpf(13)]:
        q1 = marcumq(1, a, a)
        expected = 0.5*(1 + mp.exp(-a**2)*mp.besseli(0, a**2))
        assert mp.almosteq(q1, expected)

    for a, b in [(1, 2), (10, 3.5)]:
        total = marcumq(1, a, b) + marcumq(1, b, a)
        expected = 1 + mp.exp(-(a**2 + b**2)/2)*mp.besseli(0, a*b)
        assert mp.almosteq(total, expected)


@mp.workdps(50)
def test_edge_cases():
    y = marcumq(1, 0.5, 0)
    assert y == 1
    y = cmarcumq(1, 0.5, 0)
    assert y == 0


@pytest.mark.parametrize(
    'm, a, b, ref',
    [(1, 1, 2,
      '0.269012060035909996678516959220271087421337500744873384155'),
     (4, 2, 2,
      '0.961002639616864002974959310625974743642314730388958932865'),
     (10, 2, 7,
      '0.003419880268711142942584170929968429599523954396941827739'),
     (10, 2, 20,
      '1.1463968305097683198312254572377449552982286967700350e-63'),
     (25, 3, 1,
      '0.999999999999999999999999999999999985610186498600062604510'),
     (100, 5, 25,
      '6.3182094741234192918553754494227940871375208133861094e-36'),
     (1, 0, 3,
      '0.01110899653824230649614313428693052777153926750577133022641'),
     (1.5, 0, 3,
      '0.02929088653488823210691774571191602221467112714280755827194')]
)
@mp.workdps(50)
def test_marcumq_from_wolfram(m, a, b, ref):
    # The function is MarcumQ[m, a, b] in Wolfram.
    m = mp.mpf(m)
    a = mp.mpf(a)
    b = mp.mpf(b)
    q = marcumq(m, a, b)
    assert mp.almosteq(q, mp.mpf(ref))


@pytest.mark.parametrize(
    'm, a, b, ref',
    [(0.5, 1, 2,
      '0.839994848036912854058580730864442945100083598568223741623'),
     (0.5, 5, 13,
      '0.99999999999999937790394257282158764840048274118115775113'),
     (0.5, 5, '1/13',
      '2.3417168332795098320214996797055274037932802604709459e-7'),
     (0.75, 5, '0.001',
      '7.6243509472783061143887290663443105556790864676878435e-11'),
     (51/2, 5, '0.1',
      '9.9527768099307883918427141035705197017192239618506806e-91'),
     (1, 0, 3,
      '0.9888910034617576935038568657130694722284607324942286697736'),
     (1.5, 0, 3,
      '0.970709113465111767893082254288083977785328872857192441728')]
)
@mp.workdps(50)
def test_cmarcumq_from_wolfram(m, a, b, ref):
    m = mp.mpf(m)
    a = mp.mpf(a)
    b = mp.mpf(b)
    c = cmarcumq(m, a, b)
    assert mp.almosteq(c, mp.mpf(ref))
