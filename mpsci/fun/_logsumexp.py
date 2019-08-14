
import mpmath


__all__ = ['logsumexp']


def logsumexp(logs):
    """
    Compute the log of the sum of exponentials of the input sequence.
    """
    with mpmath.extradps(5):
        log_max = max(logs)
        exps = [mpmath.exp(t - log_max) for t in logs]
        return mpmath.log(mpmath.fsum(exps)) + log_max
