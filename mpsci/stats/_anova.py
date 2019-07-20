
import mpmath
from ._basic import mean
from ..distributions import f


def anova_oneway(*args):
    """
    One-way analysis of variance (ANOVA) test.

    The number of replicates in each group may be different.

    Returns the F statistic and the p-value.
    """
    with mpmath.extradps(5):
        num_groups = len(args)
        groups = [[mpmath.mp.mpf(x) for x in group]
                  for group in args]
        n = 0
        grand_total = mpmath.mp.zero
        for group in groups:
            n += len(group)
            grand_total += mpmath.fsum(group)
        grand_mean = grand_total / n

        v = mpmath.fsum(mpmath.fsum((g - grand_mean)**2 for g in group)
                        for group in groups)
        vb = mpmath.fsum(len(group)*(mean(group) - grand_mean)**2
                         for group in groups)
        vw = v - vb
        F = vb/(num_groups - 1) / (vw/(n - num_groups))
        dof_num = num_groups - 1
        dof_den = n - num_groups
        p = f.sf(F, dof_num, dof_den)
        return F, p
