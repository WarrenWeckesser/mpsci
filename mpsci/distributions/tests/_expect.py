from mpmath import mp


# To do:
# * Something like `expect` could be made part of the public API.
# * The integral-based versions of noncentral moment, entropy,
#   skewness and kurtosis could be added to the API of the distributions
#   for which there is no explicit formula.


def expect(dist, params, func, support=None, extradps=None):
    if not hasattr(dist, 'pdf'):
        raise ValueError('dist does not have a pdf function')
    if getattr(dist, '_multivariate', False):
        raise ValueError('`expect` for multivariate distributions '
                         'is not implemented')

    if extradps is None:
        extradps = mp.dps
    with mp.extradps(extradps):
        if support is None:
            if not hasattr(dist, 'support'):
                raise RuntimeError('support not available and not provided')

            support = dist.support(*params)

        def integrand(t):
            return (func(t) * dist.pdf(t, *params))

        return mp.quad(integrand, support)


def entropy_with_integral(dist, params, support=None, extradps=None):
    return expect(dist, params, lambda t: -dist.logpdf(t, *params),
                  support=support, extradps=extradps)


def check_entropy_with_integral(dist, params, support=None, extradps=None):
    entr = dist.entropy(*params)
    ex = entropy_with_integral(dist, params, support=support,
                               extradps=extradps)
    assert mp.almosteq(entr, ex), f"{entr} not almost equal to {ex}"


def noncentral_moment_with_integral(order, dist, params, extradps=None):
    order = mp.mpf(order)
    return expect(dist, params, lambda t: t**order, extradps=extradps)


def check_noncentral_moment_with_integral(order, dist, params, extradps=None):
    m = dist.noncentral_moment(order, *params)
    intgrl = noncentral_moment_with_integral(order, dist, params,
                                             extradps=extradps)
    assert mp.almosteq(m, intgrl)


def skewness_with_integral(dist, params):
    # Skewness is E(((x - mu)/sigma)**3); compute the expected
    # value with an integral.
    mu = dist.mean(*params)
    sigma = mp.sqrt(dist.var(*params))
    return expect(dist, params, lambda t: ((t - mu)/sigma)**3)


def check_skewness_with_integral(dist, params):
    sk = dist.skewness(*params)
    intgrl = skewness_with_integral(dist, params)
    assert mp.almosteq(sk, intgrl)


def kurtosis_with_integral(dist, params):
    # Excess kurtosis is E(((x - mu)/sigma)**4) - 3; compute the expected
    # value with an integral.
    mu = dist.mean(*params)
    sigma = mp.sqrt(dist.var(*params))
    k = expect(dist, params, lambda t: ((t - mu)/sigma)**4)
    return k - 3


def check_kurtosis_with_integral(dist, params):
    k = dist.kurtosis(*params)
    intgrl = kurtosis_with_integral(dist, params)
    assert mp.almosteq(k, intgrl), f"{k} not almost equal to {intgrl}"
