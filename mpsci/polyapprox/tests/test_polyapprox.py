
from mpmath import mp
from mpsci.polyapprox import inverse_pade, inverse_taylor


mp.dps = 50


def test_inverse_pade():
    # This test is also in the docstring of inverse_pade.
    with mp.extradps(10):
        pa1 = inverse_pade(mp.sin, 1, 5, 4)
        c = mp.taylor(mp.asin, mp.sin(1), 10)
        pa2 = mp.pade(c, 5, 4)
    for polys in zip(pa1, pa2):
        for coeffs in zip(polys[0], polys[1]):
            assert mp.almosteq(coeffs[0], coeffs[1])


def test_inverse_taylor():
    # This test is also in the docstring of inverse_taylor.
    ts1 = inverse_taylor(mp.sin, 1, 5)
    ts2 = mp.taylor(mp.asin, mp.sin(1), 5)
    for coeffs in zip(ts1, ts2):
        assert mp.almosteq(coeffs[0], coeffs[1])
