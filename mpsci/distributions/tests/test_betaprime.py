import pytest
from mpmath import mp
from mpsci.distributions import betaprime, Initial
from ._utils import check_mle, call_and_check_mle
from ._expect import check_kurtosis_with_integral


# Expected values were computed with Wolfram Alpha, e.g.
#     PDF[BetaPrimeDistribution[2, 7/2], 3] = 189/8192


@pytest.mark.parametrize('scale', [1, 3])
def test_pdf(scale):
    with mp.workdps(80):
        x = 3.0
        a = 2
        b = 3.5
        p = betaprime.pdf(x*scale, a, b, scale=scale)*scale
        expected = mp.mpf('189/8192')
        assert mp.almosteq(p, expected)


@pytest.mark.parametrize(
    'x, a, b, p',
    [(3.0, 2.0, 3.5, '995/1024'),
     ('1e-12', 1.25, 2.5,
      '2.936625089475862805103544292783441344239148068331725458'
      '237324122914262883428805256602650237662010653e-15'),
     (10**25, 0.5, 0.125,
      '0.999355535410004101045571916149514907513532205970073693'
      '3856311738452896343983734668')]
)
@pytest.mark.parametrize('scale', [1, 3])
def test_cdf_invcdf(x, a, b, p, scale):
    with mp.workdps(80):
        x = mp.mpf(x)
        a = mp.mpf(a)
        b = mp.mpf(b)
        p = mp.mpf(p)
        scale = mp.mpf(scale)
        p_computed = betaprime.cdf(x*scale, a, b, scale=scale)
        assert mp.almosteq(p_computed, p)
        x_computed = betaprime.invcdf(p, a, b, scale=scale)/scale
        assert mp.almosteq(x_computed, x, rel_eps=mp.mpf('1e-77'))


@pytest.mark.parametrize(
    'x, a, b, p',
    [(3.0, 2.0, 3.5, '29/1024'),
     (0.125, 1.25, 2.5,
      '0.828718068951133944885573567547680175087963872519064943'
      '1122121393516469130876181115151511585080223107'),
     (10**25, 0.5, 0.125,
      '0.000644464589995898954428083850485092486467794029926306'
      '61436882615471036560162653318007898')]
)
@pytest.mark.parametrize('scale', [1, 3])
def test_sf(x, a, b, p, scale):
    with mp.workdps(80):
        x = mp.mpf(x)
        a = mp.mpf(a)
        b = mp.mpf(b)
        p = mp.mpf(p)
        scale = mp.mpf(scale)
        p_computed = betaprime.sf(x*scale, a, b, scale=scale)
        assert mp.almosteq(p_computed, p)
        x_computed = betaprime.invsf(p, a, b, scale=scale)/scale
        assert mp.almosteq(x_computed, x)


@pytest.mark.parametrize('scale', [1, 3])
def test_mode(scale):
    with mp.workdps(80):
        a = 2
        b = 3.5
        m = betaprime.mode(a, b, scale=scale)/scale
        expected = mp.mpf('2/9')
        assert mp.almosteq(m, expected)


@pytest.mark.parametrize('scale', [1, 3])
def test_mean(scale):
    with mp.workdps(80):
        a = 2
        b = 3.5
        m = betaprime.mean(a, b, scale=scale)/scale
        expected = mp.mpf('0.8')
        assert mp.almosteq(m, expected)


@pytest.mark.parametrize('scale', [1, 3])
def test_var(scale):
    with mp.workdps(80):
        a = 2
        b = 3.5
        v = betaprime.var(a, b, scale=scale)/scale**2
        expected = mp.mpf('0.96')
        assert mp.almosteq(v, expected)


@pytest.mark.parametrize('scale', [1, 3])
def test_skewness(scale):
    with mp.workdps(80):
        a = 2
        b = 3.5
        s = betaprime.skewness(a, b, scale=scale)
        # Skewness is independent of scale.
        expected = 13*mp.sqrt(mp.mpf('2/3'))
        assert mp.almosteq(s, expected)


# Reference values were computed with Wolfram Alpha, e.g.
#   ExcessKurtosis[BetaPrimeDistribution[2, 9/2]]
@pytest.mark.parametrize('a, b, scale, ref',
                         [(2, '9/2', 1, '1257/11'),
                          ('1/4', 10, 1, '11811/259')])
@mp.workdps(80)
def test_kurtosis(a, b, scale, ref):
    a = mp.mpf(a)
    b = mp.mpf(b)
    scale = mp.mpf(scale)
    ref = mp.mpf(ref)
    k = betaprime.kurtosis(a, b, scale=scale)
    assert mp.almosteq(k, ref)


@mp.workdps(50)
def test_kurtosis_with_integral():
    a = 5
    b = 16
    scale = 0.25
    check_kurtosis_with_integral(betaprime, (a, b, scale))


# Reference values were computed with Wolfram Alpha, e.g, the input
#     moment BetaPrimeDistribution 2 7
# generates a table of raw moments.
@pytest.mark.parametrize(
    'order, a, b, ref',
    [(1, 2, 7, '1/3'),
     (2, 2, 7, '1/5'),
     (3, 2, 7, '1/5'),
     (4, 2, 7, '1/3'),
     (5, 2, 7, 1),
     (6, 2, 7, 7)]
)
@pytest.mark.parametrize('scale', [1, 3])
def test_noncentral_moment(order, a, b, ref, scale):
    moment = betaprime.noncentral_moment(order, a, b, scale=scale)
    assert mp.almosteq(moment/scale**order, mp.mpf(ref))


@pytest.mark.parametrize(
    'x',
    [[0.03125, 0.0625, 0.125, 0.25, 0.5, 1],
     [1, 2, 3, 5, 8, 13, 21, 34, 55, 89]]
)
@mp.workdps(50)
def test_mle(x):
    call_and_check_mle(
        lambda x: betaprime.mle(x, scale=1)[:2],
        lambda x, a, b: betaprime.nll(x, a, b, scale=1),
        x,
    )


@mp.workdps(50)
def test_mle_scale_free():
    x = [1, 2, 3, 5, 8, 13, 21, 34, 55, 89]
    # This dataset needs a good initial guess.
    a1, b1, scale1 = betaprime.mle(x, a=Initial(0.9), b=Initial(3.4),
                                   scale=Initial(65))
    check_mle(betaprime.nll, x, (a1, b1, scale1))


@mp.workdps(50)
def test_mle_scale_fixed():
    x = [1, 2, 3, 5, 8, 13, 21, 34, 55, 89]
    # Fix the scale to be 25.
    a1, b1, scale1 = betaprime.mle(x, scale=25)
    assert scale1 == 25
    check_mle(lambda x, a, b: betaprime.nll(x, a, b, scale=25), x, (a1, b1))


@mp.workdps(50)
def test_mle_a_fixed():
    x = [1, 2, 3, 5, 8, 13, 21, 34, 55, 89]
    # Fix `a` to be 1.25.
    a1, b1, scale1 = betaprime.mle(x, a=1.25)
    assert a1 == 1.25
    check_mle(lambda x, b, scale: betaprime.nll(x, a=1.25, b=b, scale=scale),
              x, (b1, scale1))


@mp.workdps(50)
def test_mle_b_fixed():
    x = [1, 2, 3, 5, 8, 13, 21, 34, 55, 89]
    # Fix `b` to be 2.5.
    a1, b1, scale1 = betaprime.mle(x, b=2.5, scale=Initial(40))
    assert b1 == 2.5
    check_mle(lambda x, a, scale: betaprime.nll(x, a=a, b=2.5, scale=scale),
              x, (a1, scale1))


@mp.workdps(50)
def test_mle_a_and_b_fixed():
    x = [1, 2, 3, 5, 8, 13, 21, 34, 55, 89]
    # Fix `a` to be 0.5 and `b` to be 2.5.
    a1, b1, scale1 = betaprime.mle(x, a=0.5, b=2.5, scale=Initial(40))
    assert a1 == 0.5 and b1 == 2.5
    check_mle(lambda x, scale: betaprime.nll(x, a=0.5, b=2.5, scale=scale),
              x, (scale1,))


@mp.workdps(50)
def test_mle_a_and_scale_fixed():
    x = [1, 2, 3, 5, 8, 13, 21, 34, 55, 89]
    # Fix `a` to be 1.25 and `scale` to be 50.
    a1, b1, scale1 = betaprime.mle(x, a=1.25, scale=50)
    assert a1 == 1.25 and scale1 == 50
    check_mle(lambda x, b: betaprime.nll(x, a=1.25, b=b, scale=50),
              x, (b1,))


@mp.workdps(50)
def test_mle_b_and_scale_fixed():
    x = [1, 2, 3, 5, 8, 13, 21, 34, 55, 89]
    # Fix `b` to be 3 and `scale` to be 50.
    a1, b1, scale1 = betaprime.mle(x, b=3, scale=50)
    assert b1 == 3 and scale1 == 50
    check_mle(lambda x, a: betaprime.nll(x, a=a, b=3, scale=50),
              x, (a1,))


@mp.workdps(50)
def test_mle_all_fixed():
    x = [1, 2, 3, 5, 8, 13, 21, 34, 55, 89]
    # Fix `a` to be 1, `b` to be 3.0 and `scale` to be 50.
    a1, b1, scale1 = betaprime.mle(x, a=1, b=3, scale=50)
    assert a1 == 1 and b1 == 3 and scale1 == 50
