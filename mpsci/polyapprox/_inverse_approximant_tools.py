
import mpmath


# The function _bell_incomplete_poly was copied from sympy
# and modified to use only mpmath.

def _bell_incomplete_poly(n, k, symbols):
    r"""
    The second kind of Bell polynomials (incomplete Bell polynomials).
    Calculated by recurrence formula:

    .. math:: B_{n,k}(x_1, x_2, \dotsc, x_{n-k+1}) =
            \sum_{m=1}^{n-k+1}
            \x_m \binom{n-1}{m-1} B_{n-m,k-1}(x_1, x_2, \dotsc, x_{n-m-k})
    where
        `B_{0,0} = 1;`
        `B_{n,0} = 0; for n \ge 1`
        `B_{0,k} = 0; for k \ge 1`
    """
    if (n == 0) and (k == 0):
        return mpmath.mp.one
    elif (n == 0) or (k == 0):
        return mpmath.mp.zero
    s = mpmath.mp.zero
    a = mpmath.mp.one
    for m in range(1, n - k + 2):
        s += a * _bell_incomplete_poly(n - m, k - 1, symbols) * symbols[m - 1]
        a = a * (n - m) / m
    return s


def revert(coeffs, taylor=True):
    """
    Series reversion.

    Given `coeffs`, the Taylor coefficients of a function, find the
    Taylor coefficients of the compositional inverse of the function.

    The calculation here is based on the formulas for the coefficients
    of the inverse in terms of Bell polynomials:
    https://en.wikipedia.org/wiki/Bell_polynomials#Reversion_of_series

    If the argument `taylor` is True, the coefficients must be the
    Taylor series coefficients.  If it is not True, the coefficients
    are the derivatives.  That is, they are the Taylor coefficients
    multiplied by the factorial terms.
    """
    c0 = coeffs[0]
    if coeffs[1] == 0:
        raise ValueError('coeffs[1] must be nonzero.')

    if taylor:
        # Rescale the coefficients to the derivative values.
        f = [c * mpmath.factorial(i) for i, c in enumerate(coeffs)]
    else:
        f = coeffs

    m = len(f)
    f_hat = [f[k+1] / ((k + 1) * f[1]) for k in range(1, m-1)]

    # inv_derivs will be the series of derivatives of the inverse.
    g = [0]*m
    g[1] = 1/f[1]

    for n in range(2, m):
        s = sum((-1)**k * mpmath.rf(n, k) * _bell_incomplete_poly(n-1, k, f_hat[:n-k])
                for k in range(1, n))
        g[n] = s / f[1]**n

    if taylor:
        # Convert the derivatives to the Taylor coefficients
        inv_coeffs = [c/mpmath.factorial(k) for k, c in enumerate(g)]
    else:
        inv_coeffs = g

    return inv_coeffs, c0


def inverse_taylor(f, x0, n):
    """
    Taylor polynomial coefficients of the inverse of f.

    Given a callable f, and a point x0, find the Taylor polynomial of degree n
    of the inverse of f at x0.

    If y0 = f(x0), and if the inverse of f is g, this function returns
    the Taylor polynomial coefficients of g(y) at y0.

    f'(x0) must be nonzero.

    Examples
    --------
    >>> import mpmath
    >>> mpmath.mp.dps = 40

    Compute the Taylor coefficients of the inverse of the sine function
    sin(x) at x=1.

    >>> inverse_taylor(mpmath.sin, 1, 5)
    [mpf('1.0'),
     mpf('1.850815717680925617911753241398650193470396'),
     mpf('2.667464736243829370645086306803786566557799'),
     mpf('8.745566949501434796799480049601499630239969'),
     mpf('34.55691117453807764026147509020588920253199'),
     mpf('152.9343377104818039879748855586655382173672')]

    Compare that to computing the Taylor polynomial coefficients of
    the arcsine function directly:

    >>> mpmath.taylor(mpmath.asin, mpmath.sin(1), 5)
    [mpf('1.0'),
     mpf('1.850815717680925617911753241398650193470396'),
     mpf('2.667464736243829370645086306803786566557799'),
     mpf('8.745566949501434796799480049601499630240153'),
     mpf('34.55691117453807764026147509020588920253199'),
     mpf('152.9343377104818039879748855586655382173702')]
    """
    x0 = mpmath.mpf(x0)
    c = mpmath.taylor(f, x0, n)
    r, c0 = revert(c)
    r[0] = x0
    return [mpmath.mpf(t) for t in r]

inverse_taylor._docstring_re_subs = [
    (r"g\(y\)", r':math:`g(y)`', 0, 0),
    (r"f(')?\(x0\)", r':math:`f\1(x_0)`', 0, 0),
    (r' f(\W)', r' :math:`f`\1', 0, 0),
    (r' g(\W)', r' :math:`g`\1', 0, 0),
    (r'([xy])0', r':math:`\1_0`', 0, 0),
]

def inverse_pade(f, x0, m, n):
    """
    Padé approximant coefficients of the inverse of f.

    Given a callable f, and a point x0, find the Padé approximant of degree
    (m, n) of the inverse of f at x0.

    If y0 = f(x0), and if the inverse of f is g, this function returns
    the Padé approximant coefficients of g(y) at y0.

    f'(x0) must be nonzero.

    Examples
    --------
    >>> import mpmath
    >>> mpmath.mp.dps = 40

    Compute the Padé approximant to the inverse of sin(x) at x=1.

    >>> inverse_pade(mpmath.sin, 1, 5, 4)
    ([mpf('1.0'),
      mpf('-5.428836087225345782152614868037223199487785'),
      mpf('-14.59025586448337482707922792297134121701713'),
      mpf('76.66727054306441691994858675947043862200347'),
      mpf('20.92630843471146736348587129663693571301545'),
      mpf('-91.95538065543221755259217919809770565490541')],
     [mpf('1.0'),
      mpf('-7.279651804906271400064368109435873392958164'),
      mpf('-3.784426620962357975179374311522526358975659'),
      mpf('94.34419434777145262733458338059694153109452'),
      mpf('-114.4848137234397209897780633142520390436954')])

    Compare that to computing the Padé approximant of the arcsine
    function directly.

    >>> c = mpmath.taylor(mpmath.asin, mpmath.sin(1), 10)
    >>> mpmath.pade(c, 5, 4)
    ([mpf('1.0'),
      mpf('-5.428836087225345782152614868037223199022362'),
      mpf('-14.59025586448337482707922792297134122127003'),
      mpf('76.66727054306441691994858675947043863057723'),
      mpf('20.92630843471146736348587129663693571903839'),
      mpf('-91.95538065543221755259217919809770566742443')],
     [mpf('1.0'),
      mpf('-7.279651804906271400064368109435873392492793'),
      mpf('-3.784426620962357975179374311522526364090111'),
      mpf('94.34419434777145262733458338059694154789217'),
      mpf('-114.4848137234397209897780633142520390591904')])

    """
    d = m + n + 1
    c = inverse_taylor(f, x0, d)
    pc, qc = mpmath.pade(c, m, n)
    return pc, qc


inverse_pade._docstring_re_subs = [
    (r"g\(y\)", r':math:`g(y)`', 0, 0),
    (r"f(')?\(x0\)", r':math:`f\1(x_0)`', 0, 0),
    (r' f(\W)', r' :math:`f`\1', 0, 0),
    (r' g(\W)', r' :math:`g`\1', 0, 0),
    (r'([xy])0', r':math:`\1_0`', 0, 0),
]