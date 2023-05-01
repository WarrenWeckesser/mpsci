from mpmath import mp
from ._basic import mean
from ..distributions import f


def anova_oneway(*args):
    """
    One-way analysis of variance (ANOVA) test.

    The number of replicates in each group may be different.

    Returns the F statistic and the p-value.

    Example
    -------
    >>> from mpmath import mp
    >>> mp.dps = 20
    >>> from mpsci.stats import anova_oneway

    This example is based on the article "One-way anova" from the
    **Handbook of Biological Statistics** by John H. McDonald
    (http://www.biostathandbook.com/onewayanova.html).
    Here are shell measurements from the mussel Mytilus trossulus from five
    locations: Tillamook, Oregon; Newport, Oregon; Petersburg, Alaska; Magadan,
    Russia; and Tvarminne, Finland.

    >>> tmook = [0.0571, 0.0813, 0.0831, 0.0976, 0.0817, 0.0859,
    ...          0.0735, 0.0659, 0.0923, 0.0836]
    >>> nport = [0.0873, 0.0662, 0.0672, 0.0819, 0.0749, 0.0649,
    ...          0.0835, 0.0725]
    >>> pburg = [0.0974, 0.1352, 0.0817, 0.1016, 0.0968, 0.1064,
    ...          0.105]
    >>> mdan = [0.1033, 0.0915, 0.0781, 0.0685, 0.0677, 0.0697,
    ...         0.0764, 0.0689]
    >>> tvar = [0.0703, 0.1026, 0.0956, 0.0973, 0.1039, 0.1045]
    >>> F, p = anova_oneway(tmook, nport, pburg, mdan, tvar)
    >>> F
    mpf('7.1210194716424442964706')
    >>> p
    mpf('0.00028122423145345577062353')

    """
    with mp.extradps(5):
        num_groups = len(args)
        groups = [[mp.mpf(x) for x in group] for group in args]
        n = 0
        grand_total = mp.zero
        for group in groups:
            n += len(group)
            grand_total += mp.fsum(group)
        grand_mean = grand_total / n

        v = mp.fsum(mp.fsum((g - grand_mean)**2 for g in group)
                    for group in groups)
        vb = mp.fsum(len(group)*(mean(group) - grand_mean)**2
                     for group in groups)
        vw = v - vb
        F = vb/(num_groups - 1) / (vw/(n - num_groups))
        dof_num = num_groups - 1
        dof_den = n - num_groups
        p = f.sf(F, dof_num, dof_den)
        return F, p
