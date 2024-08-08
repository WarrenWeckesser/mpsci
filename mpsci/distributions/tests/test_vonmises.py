
from mpmath import mp
from mpsci.distributions import vonmises


@mp.workdps(50)
def test_pdf_logpdf():
    x = mp.mpf(1)/4
    mu = 0
    kappa = mp.mpf(2)
    p = vonmises.pdf(x, kappa, mu)
    # Reference value computed with Wolfram Alpha:
    #   PDF[VonMisesDistribution[0, 2], 1/4]
    refstr = '0.4847869492413841637467261534666108952610517554232914'
    ref = mp.mpf(refstr)
    assert mp.almosteq(p, ref)

    logp = vonmises.logpdf(x, kappa, mu)
    assert mp.almosteq(logp, mp.log(ref))
