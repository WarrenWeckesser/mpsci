import pytest
from mpmath import mp
from mpsci.fun import (spherical_besselj, spherical_bessely,
                       spherical_besseli, spherical_besselk)


@mp.extradps(5)
def _j0(x):
    return mp.sin(x) / x


@mp.extradps(5)
def _j1(x):
    return (mp.sin(x) - x * mp.cos(x)) / x**2


@mp.extradps(5)
def _j2(x):
    return ((3 - x**2) * mp.sin(x) - 3 * x * mp.cos(x)) / x**3


_sph_besselj_funcs = [_j0, _j1, _j2]


@pytest.mark.parametrize('n, x', [(0, 0.125), (0, 75),
                                  (1, 0.125), (1, 75),
                                  (2, 0.125), (2, 75)])
@mp.workdps(50)
def test_spherical_besselj(n, x):
    x = mp.mpf(x)
    y = spherical_besselj(n, x)
    y2 = _sph_besselj_funcs[n](x)
    assert mp.almosteq(y, y2)


@mp.extradps(5)
def _y0(x):
    return -mp.cos(x) / x


@mp.extradps(5)
def _y1(x):
    return (-mp.cos(x) - x * mp.sin(x)) / x**2


@mp.extradps(5)
def _y2(x):
    return ((x**2 - 3) * mp.cos(x) - 3 * x * mp.sin(x)) / x**3


_sph_bessely_funcs = [_y0, _y1, _y2]


@pytest.mark.parametrize('n, x', [(0, 0.125), (0, 75),
                                  (1, 0.125), (1, 75),
                                  (2, 0.125), (2, 75)])
@mp.workdps(50)
def test_spherical_bessely(n, x):
    x = mp.mpf(x)
    y = spherical_bessely(n, x)
    y2 = _sph_bessely_funcs[n](x)
    assert mp.almosteq(y, y2)


@mp.extradps(5)
def _i0(x):
    return mp.sinh(x) / x


@mp.extradps(5)
def _i1(x):
    return (x * mp.cosh(x) - mp.sinh(x)) / x**2


@mp.extradps(5)
def _i2(x):
    return ((x**2 + 3)* mp.sinh(x) - 3 * x * mp.cosh(x)) / x**3


_sph_besseli_funcs = [_i0, _i1, _i2]


@pytest.mark.parametrize('n, x', [(0, 0.125), (0, 75),
                                  (1, 0.125), (1, 75),
                                  (2, 0.125), (2, 75)])
@mp.workdps(50)
def test_spherical_besseli(n, x):
    x = mp.mpf(x)
    y = spherical_besseli(n, x)
    y2 = _sph_besseli_funcs[n](x)
    assert mp.almosteq(y, y2)



@mp.extradps(5)
def _k0(x):
    return (mp.pi / 2) * mp.exp(-x) / x


@mp.extradps(5)
def _k1(x):
    return (mp.pi / 2) * mp.exp(-x) * (x + 1) / x**2


@mp.extradps(5)
def _k2(x):
    return (mp.pi / 2)* mp.exp(-x) * (x**2 + 3 * x + 3) / x**3


_sph_besselk_funcs = [_k0, _k1, _k2]


@pytest.mark.parametrize('n, x', [(0, 0.125), (0, 75),
                                  (1, 0.125), (1, 75),
                                  (2, 0.125), (2, 75)])
@mp.workdps(50)
def test_spherical_besselk(n, x):
    x = mp.mpf(x)
    y = spherical_besselk(n, x)
    y2 = _sph_besselk_funcs[n](x)
    assert mp.almosteq(y, y2)
