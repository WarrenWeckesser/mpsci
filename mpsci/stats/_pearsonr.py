

import mpmath


def pearsonr(x, y):
    """
    Pearson's correlation coefficient.

    Returns the correlation coefficient r and the p-value.

    x and y must be one-dimensional sequences with the same lengths.

    The function assumes all the values in x and y are finite
    (no `inf`, no `nan`).
    """
    if len(x) != len(y):
        raise ValueError('lengths of x and y must be the same.')

    if all(x[0] == t for t in x[1:]) or all(y[0] == t for t in y[1:]):
        return mpmath.nan, mpmath.nan

    if len(x) == 2:
        return mpmath.sign(x[1] - x[0])*mpmath.sign(y[1] - y[0]), mpmath.mpf(1)

    x = [mpmath.mp.mpf(float(t)) for t in x]
    y = [mpmath.mp.mpf(float(t)) for t in y]

    xmean = sum(x) / len(x)
    ymean = sum(y) / len(y)

    xm = [t - xmean for t in x]
    ym = [t - ymean for t in y]

    num = sum(s*t for s, t in zip(xm, ym))
    den = mpmath.sqrt(sum(t**2 for t in xm) * sum(t**2 for t in ym))
    r = num / den

    n = len(x)
    a = mpmath.mpf(float(n))/2 - 1
    p = 2*mpmath.betainc(a, a, x2=0.5*(1-abs(r)))/mpmath.beta(a, a)

    return r, p
