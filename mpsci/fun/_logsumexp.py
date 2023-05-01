from mpmath import mp


__all__ = ['logsumexp']


def logsumexp(logs, weights=None):
    """
    Compute the log of the sum of exponentials of the input sequence.

    Examples
    --------
    Imports and configuration:

    >>> from mpmath import mp
    >>> mp.dps = 25
    >>> from mpsci.fun import logsumexp

    Test data:

    >>> x  = [1, -2, 3, 0.5]

    >>> logsumexp(x)
    mpf('3.202253447679351758368594485')

    Compare that result to:

    >>> mp.log(mp.fsum([mp.exp(xi) for xi in x]))
    mpf('3.202253447679351758368594507')

    Weight the data with the weights [1, 2, 0, 2]:

    >>> w = [1, 2, 0, 2]

    >>> logsumexp(x, weights=w)
    mpf('1.83838776432614447252849022')

    Compare to:

    >>> mp.log(mp.fsum([wi*mp.exp(xi) for xi, wi in zip(x, w)]))
    mpf('1.838387764326144472528490234')
    """
    with mp.extradps(5):
        log_max = max(logs)
        exps = [mp.exp(t - log_max) for t in logs]
        if weights is None:
            result = mp.log(mp.fsum(exps)) + log_max
        else:
            weighted = [w*e for w, e in zip(weights, exps)]
            result = mp.log(mp.fsum(weighted)) + log_max
        return result
