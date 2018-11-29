import mpmath


def pmf(k, M, n, N):
    tot, good = M, n
    bad = tot - good
    pmf =  (mpmath.beta(good+1, 1) * mpmath.beta(bad+1,1) * mpmath.beta(tot-N+1, N+1) /
            (mpmath.beta(k+1, good-k+1) * mpmath.beta(N-k+1,bad-N+k+1) * mpmath.beta(tot+1, 1)))
    return pmf


def sf(k, M, n, N):
    sf = (mpmath.binomial(N, k+1) * mpmath.binomial(M-N, n - k - 1) / mpmath.binomial(M, n) *
          mpmath.hyp3f2(1, k + 1 - n, k + 1 - N, k + 2, M + k + 2 - n - N, 1))
    return sf


# XXX Call hypergeometric_pmf instead of duplicating the implementation.
# XXX Reconcile the API: "M, n, N" vs "good, bad, nsample"
def support(good, bad, nsample):
    tot = good + bad
    pmf = []
    expected = []
    support = range(max(0, nsample - bad), min(nsample + 1, good + 1))
    for k in support:
        # XXX beta(z, 1) can be simplified.
        pk = (mpmath.beta(good+1, 1)*mpmath.beta(bad+1, 1)*mpmath.beta(tot-nsample+1, nsample+1) /
              (mpmath.beta(k+1, good - k + 1)*mpmath.beta(nsample - k + 1, bad - nsample + k + 1)*mpmath.beta(tot+1, 1)))
        pmf.append(pk)
    return support, pmf

