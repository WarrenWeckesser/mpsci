import pytest
from mpmath import mp
from mpsci.fun import roots_legendre


@mp.workdps(50)
@pytest.mark.parametrize('n', [1, 2, 3, 4, 5, 13, 21])
def test_roots_legendre(n):
    # Note: This doesn't test the weights.  The function
    # test_gauss_legendre_quadrature below tests both roots and weights.
    roots, weights = roots_legendre(n)
    for root in roots:
        assert mp.almosteq(mp.legendre(n, root), 0, rel_eps=0, abs_eps=2*mp.eps)

def test_roots_legendre_bad_n():
    with pytest.raises(ValueError, match='n must be a positive integer'):
        roots_legendre(0)

@mp.workdps(50)
@pytest.mark.parametrize('n', [3, 9, 14, 21])
def test_gausss_legendre_quadrature(n):
    roots, weights = roots_legendre(n)
    coeffs = range(1, 2*n)
    with mp.extradps(5):
        integral = mp.quad(lambda t: mp.polyval(coeffs, t, asc=True), [-1, 1])
    # glsum is the  result of the Gauss-Legendre quadrature.
    glsum = mp.fsum([w * mp.polyval(coeffs, x, asc=True)
                     for x, w in zip(roots, weights)])
    assert mp.almosteq(glsum, integral)
