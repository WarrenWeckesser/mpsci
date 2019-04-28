"""
Some signal functions.
"""

from __future__ import division
from functools import reduce
import mpmath


__all__ = ['butter_lp', 'butter_bp', 'butter_lp_ord', 'cheby1_lp',
           'freqz', 'zpkfreqz', 'poly_from_zeros']


def _prod(seq):
    """Returns the product of the elements in the sequence `seq`."""
    return reduce(lambda x, y: x*y, seq, 1)


def _relative_degree(z, p):
    """
    Return relative degree of transfer function from zeros and poles.

    This is simply len(p) - len(z), which must be nonnegative.
    A ValueError is raised if len(p) < len(z).
    """
    degree = len(p) - len(z)
    if degree < 0:
        raise ValueError("Improper transfer function. "
                         "Must have at least as many poles as zeros.")
    return degree


def _zpkbilinear(z, p, k, fs):
    """Bilinear transformation to convert a filter from analog to digital."""

    degree = _relative_degree(z, p)

    fs2 = 2*fs

    # Bilinear transform the poles and zeros
    z_z = [(fs2 + z1) / (fs2 - z1) for z1 in z]
    p_z = [(fs2 + p1) / (fs2 - p1) for p1 in p]

    # Any zeros that were at infinity get moved to the Nyquist frequency
    z_z.extend([-1] * degree)

    # Compensate for gain change
    numer = _prod(fs2 - z1 for z1 in z)
    denom = _prod(fs2 - p1 for p1 in p)
    k_z = k * numer / denom

    return z_z, p_z, k_z.real


def _zpklp2lp(z, p, k, wo=1):
    """Transform a lowpass filter to a different cutoff frequency."""

    # Scale all points radially from origin to shift cutoff frequency
    z_lp = [wo * z1 for z1 in z]
    p_lp = [wo * p1 for p1 in p]

    # Each shifted pole decreases gain by wo, each shifted zero increases it.
    # Cancel out the net change to keep overall gain the same
    degree = _relative_degree(z, p)
    k_lp = k * wo**degree

    return z_lp, p_lp, k_lp


def _zpklp2bp(z, p, k, wo, bw):

    # Scale poles and zeros to desired bandwidth
    z_lp = [zero*bw/2 for zero in z]
    p_lp = [pole*bw/2 for pole in p]

    # Duplicate poles and zeros and shift from baseband to +wo and -wo
    z_bp = ([zero + mpmath.sqrt(zero**2 - wo**2) for zero in z_lp] +
            [zero - mpmath.sqrt(zero**2 - wo**2) for zero in z_lp])

    p_bp = ([pole + mpmath.sqrt(pole**2 - wo**2) for pole in p_lp] +
            [pole - mpmath.sqrt(pole**2 - wo**2) for pole in p_lp])

    degree = _relative_degree(z, p)

    # Move degree zeros to origin, leaving degree zeros at infinity for BPF
    z_bp.extend([0]*degree)

    # Cancel out gain change from frequency scaling
    k_bp = k * bw**degree

    return z_bp, p_bp, k_bp


def _butter_analog_poles(n):
    """
    Poles of an analog Butterworth lowpass filter.

    This is the same calculation as scipy.signal.buttap(n) or
    scipy.signal.butter(n, 1, analog=True, output='zpk'), but mpmath is used,
    and only the poles are returned.
    """
    poles = []
    for k in range(-n+1, n, 2):
        poles.append(-mpmath.exp(1j*mpmath.pi*k/(2*n)))
    return poles


def butter_lp(n, Wn):
    """
    Lowpass Butterworth digital filter design.

    This computes the same result as scipy.signal.butter(n, Wn, output='zpk'),
    but it uses mpmath, and the results are returned in lists instead of numpy
    arrays.
    """
    zeros = []
    poles = _butter_analog_poles(n)
    k = 1
    fs = 2
    warped = 2 * fs * mpmath.tan(mpmath.pi * Wn / fs)
    z, p, k = _zpklp2lp(zeros, poles, k, wo=warped)
    z, p, k = _zpkbilinear(z, p, k, fs=fs)
    return z, p, k


def butter_bp(n, wlo, whi):
    """
    Bandpass Butterworth filter design.
    """
    zeros = []
    poles = _butter_analog_poles(n)
    k = 1
    fs = 2

    warpedlo = 2 * fs * mpmath.tan(mpmath.pi * wlo / fs)
    warpedhi = 2 * fs * mpmath.tan(mpmath.pi * whi / fs)
    bw = warpedhi - warpedlo
    wo = mpmath.sqrt(warpedlo * warpedhi)
    z, p, k = _zpklp2bp(zeros, poles, k, wo=wo, bw=bw)
    z, p, k = _zpkbilinear(z, p, k, fs=fs)
    return z, p, k


def butter_lp_ord(wp, ws, deltap, deltas, fs=1):
    """
    (deltap and deltas are not in dB!)
    """
    r = ((1/deltas)**2 - 1) / ((1/(1-deltap))**2 - 1)
    print(r)
    t = mpmath.tan(mpmath.pi*ws/fs)/mpmath.tan(mpmath.pi*wp/fs)
    print(t)
    n = mpmath.log(r) / (2*mpmath.log(t))
    return n


def cheby1_lp(N, rp, Wn):
    """
    Chebyshev Type I lowpass digital filter design.

    This function computes the same result as

        scipy.signal.cheby1(n, rp, Wn, output='zpk')

    but it uses mpmath, and the results are returned in lists instead of
    numpy arrays.
    """
    zeros = []

    # Ripple factor (epsilon)
    rp = mpmath.mp.mpf(rp)
    eps = mpmath.sqrt(mpmath.power(10, (rp/10)) - 1)
    mu = mpmath.asinh(1 / eps) / N

    # Arrange poles in an ellipse on the left half of the S-plane
    poles = []
    k = mpmath.mp.mpf(1)
    for m in range(-N+1, N, 2):
        theta = mpmath.pi * m / (2*N)
        pole = -mpmath.sinh(mu + 1j*theta)
        poles.append(pole)
        k *= -pole

    if N % 2 == 0:
        k = k / mpmath.sqrt(1 + eps * eps)

    fs = mpmath.mp.mpf(2)
    warped = 2 * fs * mpmath.tan(mpmath.pi * Wn / fs)
    z, p, k = _zpklp2lp(zeros, poles, k, wo=warped)
    z, p, k = _zpkbilinear(z, p, k, fs=fs)
    return z, p, k


def zpkfreqz(z, p, k, worN=None):
    """
    Frequency response of a filter in zpk format.

    This is the same calculation as scipy.signal.freqz, but the input is in
    zpk format, and the results are returned in lists instead of numpy arrays.

    *See also:* `mpsci.signal.freqz`
    """
    if worN is None or isinstance(worN, int):
        N = worN or 512
        ws = [mpmath.pi * mpmath.mpf(j) / N for j in range(N)]
    else:
        ws = worN

    h = []
    for wk in ws:
        zm1 = mpmath.exp(1j * wk)
        numer = _prod([zm1 - t for t in z])
        denom = _prod([zm1 - t for t in p])
        hk = k * numer / denom
        h.append(hk)
    return ws, h


def freqz(b, a=1, worN=None):
    """
    Frequency response of a filter in (b, a) format (i.e. transfer function).

    This function is similar to `scipy.signal.freqz`, but the results are stored
    in lists.

    *See also:* `mpsci.signal.zpkfreqz`
    """
    if worN is None or isinstance(worN, int):
        N = worN or 512
        ws = [mpmath.pi * mpmath.mpf(j) / N for j in range(N)]
    else:
        ws = worN

    # This assumes b and a contain real values.
    try:
        len(b)
    except TypeError:
        b = [b]
    b = [mpmath.mp.mpf(t) for t in b]
    try:
        len(a)
    except TypeError:
        a = [a]
    a = [mpmath.mp.mpf(t) for t in a]

    h = []
    for wk in ws:
        z = mpmath.exp(-1j * wk)
        hk = mpmath.polyval(b[::-1], z) / mpmath.polyval(a[::-1], z)
        h.append(hk)

    return ws, h


def _convolve(seq1, seq2):
    # XXX Fix this; it currently does more than necessary.
    c = []
    for i in range(-len(seq2) + 1, len(seq1)):
        val = 0
        for k in range(len(seq2)):
            if 0 <= (i + k) < len(seq1):
                val += seq1[i+k] * seq2[len(seq2) - 1 - k]
        c.append(val)
    return c


def poly_from_zeros(z):
    """
    Convert the zeros of a polynomial to the coefficients.

    Coefficients are ordered from highest degree term to lowest.

    The leading coefficient will be 1.

    This is the same operation as performed by `numpy.poly`.

    Examples
    --------
    Convert the zeros [2, 3] to the polynomial coefficients,

        (x - 2)*(x - 3) = x**2 - 5*x + 6

    >>> poly_from_zeros([2, 3])
    [1, -5, 6]

    """
    if len(z) == 0:
        return [1]
    p = [1, -z[0]]
    for k in range(1, len(z)):
        p = _convolve(p, [1, -z[k]])
    return p

_math = """
.. math::

   (x - 2)(x - 3) = x^2 - 5x + 6

"""

poly_from_zeros._docstring_re_subs = [
    ('[ ]+\(x.*6', _math, 0, 0)
]
