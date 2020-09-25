

import mpmath
from ..distributions import normal


__all__ = ['pearsonr', 'pearsonr_ci']


def pearsonr(x, y, alternative='two-sided'):
    """
    Pearson's correlation coefficient.

    Returns the correlation coefficient r and the p-value.

    x and y must be one-dimensional sequences with the same lengths.

    The function assumes all the values in x and y are finite
    (no `inf`, no `nan`).

    Examples
    --------
    >>> from mpsci.stats import pearsonr
    >>> import mpmath
    >>> mpmath.mp.dps = 25
    >>> x = [1, 2, 3, 5, 8, 10]
    >>> y = [0.25, 2, 2, 2.5, 2.4, 5.5]

    Compute the correlation coefficent and p-value.

    >>> r, p = pearsonr(x, y)
    >>> r
    mpf('0.8645211772786436751458124677')
    >>> p
    mpf('0.02628844331049414042317641803')

    Compute a one-sided p-value.  The correlation coefficient is the
    same; only the p-value is different.

    >>> r, p = pearsonr(x, y, alternative='greater')
    >>> r
    mpf('0.8645211772786436751458124677')
    >>> p
    mpf('0.01314422165524707021158820901')
    """
    if alternative not in ['two-sided', 'less', 'greater']:
        raise ValueError("alternative must be 'two-sided', 'less', or "
                         "'greater'.")
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
    if alternative == 'two-sided':
        p = 2*mpmath.betainc(a, a, x2=0.5*(1-abs(r)))/mpmath.beta(a, a)
    elif alternative == 'less':
        p = mpmath.betainc(a, a, x2=0.5*(1+r))/mpmath.beta(a, a)
    else:
        # alternative == 'greater'
        p = mpmath.betainc(a, a, x2=0.5*(1-r))/mpmath.beta(a, a)

    return r, p


def pearsonr_ci(r, n, alpha, alternative='two-sided'):
    """
    Confidence interval of Pearson's correlation coefficient.

    This function uses Fisher's transformation to compute the confidence
    interval of Pearson's correlation coefficient.

    Examples
    --------
    Imports:

    >>> import mpmath
    >>> mpmath.mp.dps = 20
    >>> from mpsci.stats import pearsonr, pearsonr_ci

    Sample data:

    >>> a = [2, 4, 5, 7, 10, 11, 12, 15, 16, 20]
    >>> b = [2.53, 2.41, 3.60, 2.69, 3.19, 4.05, 3.71, 4.65, 4.33, 4.70]

    Compute the correlation coefficient:

    >>> r, p = pearsonr(a, b)
    >>> r
    mpf('0.893060379514729854846')
    >>> p
    mpf('0.00050197523992669206603645')

    Compute the 95% confidence interval for r:

    >>> rlo, rhi = pearsonr_ci(r, n=len(a), alpha=0.05)
    >>> rlo
    mpf('0.60185206817708369265664')
    >>> rhi
    mpf('0.97464778383702233502275')

    """
    if alternative not in ['two-sided', 'less', 'greater']:
        raise ValueError("alternative must be 'two-sided', 'less', or "
                         "'greater'.")

    with mpmath.mp.extradps(5):
        zr = mpmath.atanh(r)
        n = mpmath.mp.mpf(n)
        alpha = mpmath.mp.mpf(alpha)
        s = mpmath.sqrt(1/(n - 3))
        if alternative == 'two-sided':
            h = normal.invcdf(1 - alpha/2)
            zlo = zr - h*s
            zhi = zr + h*s
            rlo = mpmath.tanh(zlo)
            rhi = mpmath.tanh(zhi)
        elif alternative == 'less':
            h = normal.invcdf(1 - alpha)
            zhi = zr + h*s
            rhi = mpmath.tanh(zhi)
            rlo = -mpmath.mp.one
        else:
            # alternative == 'greater'
            h = normal.invcdf(1 - alpha)
            zlo = zr - h*s
            rlo = mpmath.tanh(zlo)
            rhi = mpmath.mp.one
        return rlo, rhi
