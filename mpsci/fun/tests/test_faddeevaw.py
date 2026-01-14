from mpmath import mp
from mpsci.fun import faddeevaw
import pytest


@mp.workdps(25)
def test_faddeevaw_zero():
    w = faddeevaw(0j)
    assert w == mp.mpc(1)


@mp.workdps(50)
@pytest.mark.parametrize('z', [1.0, -3 + 2j, 5j])
def test_faddeevaw_conj_property(z):
    # This tests the property w(z.conj()).conj() = w(-z).
    # It does not verify that the values computed are correct.
    w = faddeevaw(z.conjugate()).conjugate()
    assert mp.almosteq(w, faddeevaw(-z))
