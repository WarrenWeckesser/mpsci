from mpmath import mp
from mpsci.distributions import chi2
import pytest


@mp.workdps(50)
def test_basic_pdf():
    # Precomputed results using Wolfram
    #   PDF[ChiSquareDistribution[k], x]

    valstr = '0.15418032980376927681122055778139364630515950769552'
    expected = mp.mpf(valstr)
    assert mp.almosteq(chi2.pdf(3, 5), expected)

    valstr = '0.073224912809632435566001478191006253712616408469863'
    expected = mp.mpf(valstr)
    assert mp.almosteq(chi2.pdf(5, 3), expected)

    valstr = '8.2025293495250421352683463106457295660525245864016e-7'
    expected = mp.mpf(valstr)
    assert mp.almosteq(chi2.pdf(50, 25/2), expected)


@mp.workdps(50)
def test_basic_logpdf():
    # Precomputed results using Wolfram
    #   log(PDF[ChiSquareDistribution[k], x])

    valstr = '-1.8696323888706178960827071179443547875376521947264'
    expected = mp.mpf(valstr)
    assert mp.almosteq(chi2.logpdf(3, 5), expected)

    valstr = '-109.00728367462314733374258625716397761840084187637237'
    expected = mp.mpf(valstr)
    k = mp.mpf('31')/3
    assert mp.almosteq(chi2.logpdf(250, k), expected)


@mp.workdps(50)
def test_logpdf_negx():
    assert chi2.logpdf(-3, 4) == mp.ninf


@mp.workdps(50)
def test_basic_cdf():
    # Precomputed results using Wolfram
    #   CDF[ChiSquareDistribution[k], x]

    valstr = '0.82820285570326686493639334781694850021090176319403'
    expected = mp.mpf(valstr)
    assert mp.almosteq(chi2.cdf(5, 3), expected)

    valstr = '0.99999794808843309198843919723813688969884118187012'
    expected = mp.mpf(valstr)
    assert mp.almosteq(chi2.cdf(50, 25/2), expected)


@pytest.mark.parametrize('p, k', [('0.125', '27'),
                                  ('0.975', '13.5'),
                                  ('0.5', '8.75'),
                                  ('1e-12', '98')])
@mp.workdps(50)
def test_invcdf_cdf_roundtrip(p, k):
    p = mp.mpf(p)
    k = mp.mpf(k)
    x = chi2.invcdf(p, k)
    assert mp.almosteq(chi2.cdf(x, k), p)


@mp.workdps(50)
def test_basic_sf():
    # Precomputed results using Wolfram
    #   SurvivalFunction[ChiSquareDistribution[k], x]

    valstr = '0.17179714429673313506360665218305149978909823680597'
    expected = mp.mpf(valstr)
    assert mp.almosteq(chi2.sf(5, 3), expected)

    valstr = '2.05191156690801156080276186311030115881812988312938e-6'
    expected = mp.mpf(valstr)
    assert mp.almosteq(chi2.sf(50, 25/2), expected)


@pytest.mark.parametrize('p, k', [('0.125', '27'),
                                  ('0.975', '13.5'),
                                  ('0.5', '8.75'),
                                  ('1e-12', '98')])
@mp.workdps(50)
def test_invsf_sf_roundtrip(p, k):
    p = mp.mpf(p)
    k = mp.mpf(k)
    x = chi2.invsf(p, k)
    assert mp.almosteq(chi2.sf(x, k), p)


@mp.workdps(50)
def test_basic_mean_var():
    assert chi2.mean(5) == 5
    assert chi2.var(5) == 10


@mp.workdps(50)
def test_mode():
    assert chi2.mode(5) == 3
    assert chi2.mode(1.5) == 0


@mp.workdps(50)
def test_noncentral_moment():
    k = 3
    m = [chi2.noncentral_moment(t, k) for t in range(1, 5)]
    # References values were double-checked with Wolfram Alpha:
    #     moments ChiSquareDistribution 3
    assert m == [3, 15, 105, 945]
