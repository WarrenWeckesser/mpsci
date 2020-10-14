
import mpmath


__all__ = ['logsumexp']


def logsumexp(logs, weights=None):
    """
    Compute the log of the sum of exponentials of the input sequence.

    Examples
    --------
    Imports and configuration:

    >>> import mpmath
    >>> mpmath.mp.dps = 25
    >>> from mpsci.fun import logsumexp

    Test data:

    >>> x  = [1, -2, 3, 0.5]

    >>> logsumexp(x)
    mpf('3.202253447679351758368594485')

    Compare that result to:

    >>> mpmath.log(mpmath.fsum([mpmath.exp(xi) for xi in x]))
    mpf('3.202253447679351758368594507')

    Weight the data with the weights [1, 2, 0, 2]:

    >>> w = [1, 2, 0, 2]

    >>> logsumexp(x, weights=w)
    mpf('1.83838776432614447252849022')

    Compare to:

    >>> mpmath.log(mpmath.fsum([wi*mpmath.exp(xi) for xi, wi in zip(x, w)]))
    mpf('1.838387764326144472528490234')
    """
    with mpmath.extradps(5):
        log_max = max(logs)
        exps = [mpmath.exp(t - log_max) for t in logs]
        if weights is None:
            result = mpmath.log(mpmath.fsum(exps)) + log_max
        else:
            weighted = [w*e for w, e in zip(weights, exps)]
            result = mpmath.log(mpmath.fsum(weighted)) + log_max
        return result
