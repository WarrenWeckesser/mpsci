from mpmath import mp


def mp_bisect(func, x0, x1):
    x0 = mp.mpf(x0)
    x1 = mp.mpf(x1)
    y0 = func(x0)
    if y0 == 0:
        return x0
    y1 = func(x1)
    if y1 == 0:
        return x1
    if (y0 > 0 and y1 > 0) or (y0 < 0 and y1 < 0):
        print(f'{y0 = }')
        print(f'{y1 = }')
        raise ValueError('func(x0) and func(x1) have the same sign')
    while True:
        xmid = (x0 + x1)/2
        ymid = func(xmid)
        if ymid == 0 or (xmid == x0 or xmid == x1):
            return xmid
        if mp.sign(ymid) == mp.sign(y0):
            x0 = xmid
            y0 = ymid
        else:
            x1 = xmid
            y1 = ymid


def betaincinv(a, b, y, method='findroot', complement=False):
    """
    Inverse of the regularized incomplete beta function.

    If `complement` is True, the inverse of the complement of the
    regularized incomplete beta function is computed.

    The function name and first three parameters match those of
    `scipy.special.betaincinv`.  With `complement=True`, the function
    computes the equivalent of `scipy.special.betainccinv`.

    The `method` parameter has several options:

    * `"findroot"`:
        Use `mpmath.mp.findroot`, with initial guess 0.5.
    * `("findroot", x0)`:
        Use `mpmath.mp.findroot`, with initial guess `x0`.
    * `"bisect"`:
        Use a bisection method, starting with the root bracket
        [0, 1].
    * `("bisect", [xa, xb])`:
        Use a bisection method; start the bisection with the
        bracket `[xa, xb]`.  `xa` and `xb` must be chosen so that
        the inverse is in the interval `[xa, xb]`.

    The numerical method that is used to find the root (either
    `mpmath.mp.findroot` or the bisection method) might fail for some
    inputs. If that happens, try changing the method, or try changing
    the initial guess or initial bracket.

    Examples
    --------
    >>> from mpmath import mp
    >>> from mpsci.fun import betaincinv

    >>> mp.dps = 125
    >>> a = mp.mpf('0.001')
    >>> b = mp.mpf(2500)
    >>> y = mp.mpf('0.995')

    >>> x = betaincinv(a, b, y, method='bisect)
    >>> with mp.workdps(50):
    ...     print(x)
    0.0000015015140517221501171052646134215218978294984794701

    Verify the inverse:

    >>> y1 = mp.betainc(a, b, 0, x, regularized=True)
    >>> with mp.workdps(50):
    ...    print(y1)
    ...
    0.995

    """
    if y < 0 or y > 1:
        return mp.nan
    if y == 0:
        return mp.one if complement else mp.zero
    if y == 1:
        return mp.zero if complement else mp.one
    y = mp.mpf(y)

    x0 = 0.5
    bracket = [0, 1]
    if isinstance(method, tuple):
        if method[0] == 'bisect':
            bracket = [mp.mpf(t) for t in method[1]]
            method = method[0]
        elif method[0] == 'findroot':
            x0 = mp.mpf(method[1])
            method = method[0]
    if method not in ['findroot', 'bisect']:
        raise ValueError('invalid method given')

    if complement:

        def func(t):
            return mp.betainc(a, b, t, 1, regularized=True) - y

    else:

        def func(t):
            return mp.betainc(a, b, 0, t, regularized=True) - y

    if method == 'bisect':
        try:
            return mp_bisect(func, bracket[0], bracket[1])
        except Exception:
            raise RuntimeError('failed to find the inverse with bisection')
    else:
        try:
            return mp.findroot(func, x0)
        except Exception:
            raise RuntimeError('failed to find the inverse with '
                               'mpmath.mp.findroot')
