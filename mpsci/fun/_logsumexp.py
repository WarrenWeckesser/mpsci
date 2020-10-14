
import mpmath


__all__ = ['logsumexp']


def logsumexp(logs, weights=None):
    """
    Compute the log of the sum of exponentials of the input sequence.
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
