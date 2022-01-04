
import operator
import mpmath


def _root_approx(n, k):
    # Tricomi approximation
    with mpmath.extradps(5):
        n = mpmath.mpf(n)
        k = mpmath.mpf(k)
        c = 1 + (1/(8*n**2)) * (-1 + 1/n)
        a = mpmath.pi*(4*k - 1)/(4*n + 2)
        return c*mpmath.cos(a)


def roots_legendre(n):
    """
    Compute the roots of the Legendre polynomial, and quadrature weights.

    Warning: not tested beyond n=21.

    Examples
    --------
    >>> import mpmath
    >>> mpmath.mp.dps = 40
    >>> from mpsci.fun import roots_legendre

    >>> roots, weights = roots_legendre(7)
    >>> roots
    [mpf('-0.9491079123427585245261896840478512624007709'),
    mpf('-0.7415311855993944398638647732807884070741476'),
    mpf('-0.405845151377397166906606412076961463347382'),
    mpf('0.0'),
    mpf('0.405845151377397166906606412076961463347382'),
    mpf('0.7415311855993944398638647732807884070741476'),
    mpf('0.9491079123427585245261896840478512624007709')]
    >>> weights
    [mpf('0.1294849661688696932706114326790820183285874'),
    mpf('0.2797053914892766679014677714237795824869251'),
    mpf('0.3818300505051189449503697754889751338783651'),
    mpf('0.4179591836734693877551020408163265306122449'),
    mpf('0.3818300505051189449503697754889751338783651'),
    mpf('0.2797053914892766679014677714237795824869251'),
    mpf('0.1294849661688696932706114326790820183285874')]
    """
    n = operator.index(n)
    if n < 1:
        raise ValueError('n must be a positive integer.')
    if n == 1:
        return [mpmath.mp.zero], [mpmath.mpf(2)]
    elif n == 2:
        x = 1/mpmath.sqrt(3)
        w = mpmath.mp.one
        return [-x, x], [w, w]
    elif n == 3:
        x = mpmath.sqrt('0.6')
        wx = mpmath.mpf('5/9')
        w0 = mpmath.mpf('8/9')
        return [-x, mpmath.mp.zero, x], [wx, w0, wx]
    approx_roots = [_root_approx(n, k) for k in range(1, n//2 + 1)]
    with mpmath.extradps(5):
        roots = [mpmath.findroot(lambda x: mpmath.legendre(n, x), x0)
                 for x0 in approx_roots]
        derivs = [mpmath.diff(lambda x: mpmath.legendre(n, x), root)
                  for root in roots]
        weights = [2/((1-root**2)*deriv**2)
                   for root, deriv in zip(roots, derivs)]
        if n & 1:
            z = mpmath.mp.zero
            root0 = [z]
            deriv = mpmath.diff(lambda x: mpmath.legendre(n, x), z)
            weight0 = [2/deriv**2]
        else:
            root0 = []
            weight0 = []
        return ([-r for r in roots] + root0 + roots[::-1],
                weights + weight0 + weights[::-1])
