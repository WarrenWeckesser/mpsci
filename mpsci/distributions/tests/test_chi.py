
from mpmath import mp
from mpsci.distributions import chi


def test_basic_pdf():
    with mp.workdps(50):
        # Precomputed results using Wolfram
        #   PDF[ChiDistribution[k], x]

        valstr = '0.0042825672244763312567703836301019479421534053439844360'
        expected = mp.mpf(valstr)
        assert mp.almosteq(chi.pdf(4, 3), expected)

        valstr = '3.71073289303285830890274027770297984404243487230242288e-8'
        expected = mp.mpf(valstr)
        assert mp.almosteq(chi.pdf(mp.mpf('0.01'), 9/2), expected)


def test_basic_logpdf():
    with mp.workdps(50):
        # Precomputed results using Wolfram
        #   log(PDF[ChiDistribution[k], x])

        valstr = '-17.109451341550648019199921622226468826451063970292087'
        expected = mp.mpf(valstr)
        assert mp.almosteq(chi.logpdf(mp.mpf('0.01'), 9/2), expected)


def test_basic_cdf():
    with mp.workdps(50):
        # Precomputed results using Wolfram
        #   CDF[ChiDistribution[k], x]

        valstr = '0.9988660157102146773432998625790302104175739819130433'
        expected = mp.mpf(valstr)
        assert mp.almosteq(chi.cdf(4, 3), expected)


def test_basic_sf():
    with mp.workdps(50):
        # Precomputed results using Wolfram
        #   SurvivalFunction[ChiDistribution[k], x]

        valstr = '0.0011339842897853226567001374209697895824260180869566620'
        expected = mp.mpf(valstr)
        assert mp.almosteq(chi.sf(4, 3), expected)


def test_basic_mean():
    with mp.workdps(50):
        assert mp.almosteq(chi.mean(3), 2*mp.sqrt(2/mp.pi))


def test_basic_var():
    with mp.workdps(50):
        assert mp.almosteq(chi.var(3), 3 - 8/mp.pi)


def test_basic_mode():
    with mp.workdps(50):
        assert mp.almosteq(chi.mode(3), mp.sqrt(2))


@mp.workdps(50)
def test_noncentral_moment():
    k = 3
    m = [chi.noncentral_moment(t, k) for t in range(1, 5)]
    # References values were computed with Wolfram Alpha:
    #     moments ChiDistribution 3
    sqrt2opi = mp.sqrt(2/mp.pi)
    ref = [2*sqrt2opi, 3, 8*sqrt2opi, 15]
    for mn, refn in zip(m, ref):
        assert mp.almosteq(mn, refn)
