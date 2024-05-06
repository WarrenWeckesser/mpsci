import pytest
import mpmath
from mpmath import mp
from mpsci.fun import wright_bessel


def test_rho_validation():
    with pytest.raises(ValueError, match='must be greater than -1'):
        wright_bessel(3.0, -1.0, 4+5j)


@mp.workdps(25)
def test_z_zero():
    z = 0
    rho = 3
    beta = 2 + 4j
    w = wright_bessel(z, rho, beta)
    assert mp.almosteq(w, 1/mp.gamma(beta))


@pytest.mark.parametrize(
    'z, rho, beta',
    [(2+1j, 2.5, 0.25-3j),
     (2+1j, -0.5, 0.25-3j),
     (12.0, 6.0, 4.0),
     (0.0, 0.75, 2.0),
     (1+2j, 1.0, -1-1j),
     (1+2j, 1.0, -3),
     (1+3j, 0.0, 0.0),
     (9-1j, 0.0, 10-1j),
     (3-10j, 1.5, -3.0)])
@mp.workdps(30)
def test_recurrence_relation(z, rho, beta):
    # Verify that the recurrence relation given as equation F.7 in
    # https://appliedmath.brown.edu/sites/default/files/fractional/
    #    36%20TheWrightFunctions.pdf
    # is satisfied at several sample points.
    z = mpmath.mpmathify(z)
    rho = mpmath.mpmathify(rho)
    beta = mpmath.mpmathify(beta)
    w0 = wright_bessel(z, rho, beta)
    w_beta_minus_1 = wright_bessel(z, rho, beta - 1)
    w_beta_plus_rho = wright_bessel(z, rho, beta + rho)
    assert mp.almosteq(rho*z*w_beta_plus_rho,
                       w_beta_minus_1 + (1 - beta)*w0)
