from mpmath import mp
from mpsci.distributions import invchi2


@mp.workdps(50)
def test_basic_pdf():
    # Precomputed results using Wolfram
    #   PDF[InverseChiSquareDistribution[k], x]

    valstr = '0.10373303650472434764451353404955302024945975867374'
    expected = mp.mpf(valstr)
    assert mp.almosteq(invchi2.pdf(1.5, 3), expected)

    valstr = '2.34106001613725228245145002508895597667308289018820e-14'
    expected = mp.mpf(valstr)
    assert mp.almosteq(invchi2.pdf(81/4, 25/2), expected)


@mp.workdps(50)
def test_basic_logpdf():
    # Precomputed results using Wolfram
    #   log(PDF[InverseChiSquareDistribution[k], x])

    valstr = '-3.7700120335846911034319542107866986558441878469126'
    expected = mp.mpf(valstr)
    assert mp.almosteq(invchi2.logpdf(1.5, 5), expected)

    valstr = '-31.385587476770552920353333056344968208697046093867'
    expected = mp.mpf(valstr)
    assert mp.almosteq(invchi2.logpdf(81/4, 25/2), expected)


@mp.workdps(50)
def test_logpdf_negx():
    assert invchi2.logpdf(-3, 4) == mp.ninf


@mp.workdps(50)
def test_basic_cdf():
    # Precomputed results using Wolfram
    #   CDF[InverseChiSquareDistribution[k], x]

    valstr = '0.98474787901850902869682324882702477083525051202079'
    expected = mp.mpf(valstr)
    assert mp.almosteq(invchi2.cdf(1.5, 5), expected)

    valstr = '0.99999999999992389055640916792063614896647765633255'
    expected = mp.mpf(valstr)
    assert mp.almosteq(invchi2.cdf(81/4, 25/2), expected)


@mp.workdps(50)
def test_basic_sf():
    # Precomputed results using Wolfram
    #   SurvivalFunction[InverseChiSquareDistribution[k], x]

    valstr = '0.01525212098149097130317675117297522916474948797921'
    expected = mp.mpf(valstr)
    assert mp.almosteq(invchi2.sf(1.5, 5), expected)

    valstr = '7.61094435908320793638510335223436674450440059228449e-14'
    expected = mp.mpf(valstr)
    assert mp.almosteq(invchi2.sf(81/4, 25/2), expected)


@mp.workdps(50)
def test_mean():
    assert invchi2.mean(4) == mp.mpf('0.5')


@mp.workdps(50)
def test_var():
    assert mp.almosteq(invchi2.var(6), mp.one/16)


@mp.workdps(50)
def test_mode():
    assert invchi2.mode(6) == mp.mpf('0.125')
