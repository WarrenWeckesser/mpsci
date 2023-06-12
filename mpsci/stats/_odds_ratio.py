
from dataclasses import dataclass
from mpmath import mp
from ..distributions import (fishers_noncentral_hypergeometric as fnch,
                             hypergeometric, normal)
from ._fisher_exact import fisher_exact


def _unpack_table(table):
    (a, b), (c, d) = table
    return a, b, c, d


def _unpack_table_to_mpf(table):
    a, b, c, d = _unpack_table(table)
    a = mp.mpf(a)
    b = mp.mpf(b)
    c = mp.mpf(c)
    d = mp.mpf(d)
    return a, b, c, d


def _row_or_column_zero(table):
    # Check if both values a row or column are zero.
    a, b, c, d = _unpack_table(table)
    return (a == b == 0) or (c == d == 0) or (a == c == 0) or (b == d == 0)


def _sample_odds_ratio(table):
    """
    Given a table [[a, b], [c, d]], compute a*d/(b*c).

    Return nan if the numerator and denominator are 0.
    Return inf if just the denominator is 0.
    """
    with mp.extradps(5):
        a, b, c, d = _unpack_table_to_mpf(table)
        if c > 0 and b > 0:
            oddsratio = a*d / (c*b)
        elif a == 0 or d == 0:
            oddsratio = mp.nan
        else:
            oddsratio = mp.inf
        return oddsratio


def _solve(func):
    """
    Solve func(nc) = 0.  func must be an increasing function.
    """
    with mp.extradps(5):
        # We could just as well call the variable `x` instead of `nc`, but we
        # always call this function with functions for which nc (the
        # noncentrality parameter) is the variable for which we are solving.
        nc = mp.one
        value = func(nc)
        if value == 0:
            return nc

        # Multiplicative factor by which to increase or decrease nc when
        # searching for a bracketing interval.
        factor = mp.mpf(2)
        # Find a bracketing interval.
        if value > 0:
            nc /= factor
            while func(nc) > 0:
                nc /= factor
            lo = nc
            hi = factor*nc
        else:
            nc *= factor
            while func(nc) < 0:
                nc *= factor
            lo = nc/factor
            hi = nc

        # lo and hi bracket the solution for nc.
        nc = mp.findroot(func, (lo, hi), solver='illinois')
        return nc


def _nc_hypergeom_mean_inverse(x, total, ngood, nsample):
    """
    For the given total, ngood, and nsample, find the noncentrality
    parameter of Fisher's noncentral hypergeometric distribution whose
    mean is x.
    """
    nc = _solve(lambda nc: fnch.mean(nc, total, ngood, nsample) - x)
    return nc


def _hypergeom_params_from_table(table):
    a, b, c, d = _unpack_table(table)
    x = a
    total = a + b + c + d
    ngood = a + b
    nsample = a + c
    return x, total, ngood, nsample


def _ci_upper(table, alpha):
    """
    Compute the upper end of the confidence interval.
    """
    if mp.isinf(_sample_odds_ratio(table)):
        return mp.inf

    x, total, ngood, nsample = _hypergeom_params_from_table(table)

    # fnch.cdf is a decreasing function of nc, so we negate
    # it in the lambda expression.
    nc = _solve(lambda nc: -fnch.cdf(x, nc, total, ngood, nsample) + alpha)
    return nc


def _ci_lower(table, alpha):
    """
    Compute the lower end of the confidence interval.
    """
    if _sample_odds_ratio(table) == 0:
        return mp.zero

    x, total, ngood, nsample = _hypergeom_params_from_table(table)

    nc = _solve(lambda nc: fnch.sf(x - 1, nc, total, ngood, nsample) - alpha)
    return nc


def _conditional_oddsratio(table):
    """
    Conditional MLE of the odds ratio for the 2x2 contingency table.
    """
    x, total, ngood, nsample = _hypergeom_params_from_table(table)
    sup, pmf = hypergeometric.support(total, ngood, nsample)
    sup = list(sup)

    # Check if x is at one of the extremes of the support.  If so, we know
    # the odds ratio is either 0 or inf.
    if x == sup[0]:
        # x is at the low end of the support.
        return mp.zero
    if x == sup[-1]:
        # x is at the high end of the support.
        return mp.inf

    nc = _nc_hypergeom_mean_inverse(x, total, ngood, nsample)
    return nc


def _conditional_oddsratio_ci(table, confidence_level,
                              alternative='two-sided'):
    """
    Conditional exact confidence interval for the odds ratio.

    This function implements the "exact confidence limits",
    as explained in section 2 of the paper:

        J. Cornfield (1956), A statistical problem arising from
        retrospective studies. In Neyman, J. (ed.), Proceedings of
        the Third Berkeley Symposium on Mathematical Statistics and
        Probability 4, pp. 135-148.

    That paper gives a concise summary of the formulas, but the
    primary reference for the conditional odds ratio with
    confidence intervals is

        R. A. Fisher (1935), The logic of inductive inference,
        Journal of the Royal Statistical Society, Vol. 98, No. 1,
        pp. 39-82.

    See "Example 1" starting on page 48.

    """
    with mp.extradps(5):
        confidence_level = mp.mpf(confidence_level)
        if alternative == 'two-sided':
            alpha = (mp.one - confidence_level)/2
            lower = _ci_lower(table, alpha)
            upper = _ci_upper(table, alpha)
        elif alternative == 'less':
            lower = mp.zero
            upper = _ci_upper(table, mp.one - confidence_level)
        else:
            # alternative == 'greater'
            lower = _ci_lower(table, mp.one - confidence_level)
            upper = mp.inf

        return lower, upper


def _sample_odds_ratio_ci(table, confidence_level,
                          alternative='two-sided'):
    confidence_level = mp.mpf(confidence_level)
    oddsratio = _sample_odds_ratio(table)
    log_or = mp.log(oddsratio)
    a, b, c, d = _unpack_table_to_mpf(table)
    se = mp.sqrt(sum(1/a + 1/b + 1/c + 1/d))
    if alternative == 'less':
        z = normal.invcdf(confidence_level)
        loglow = mp.ninf
        loghigh = log_or + z*se
    elif alternative == 'greater':
        z = normal.invcdf(confidence_level)
        loglow = log_or - z*se
        loghigh = mp.inf
    else:
        # alternative is 'two-sided'
        half = mp.mpf(0.5)
        z = normal.invcdf(half*confidence_level + half)
        loglow = log_or - z*se
        loghigh = log_or + z*se

    return mp.exp(loglow), mp.exp(loghigh)


@dataclass
class ConfidenceInterval:
    low: float
    high: float


@dataclass
class OddsRatioResult:
    """
    Result of `odds_ratio`.

    Attributes
    ----------
    table :
        The table that was passed to `odds_ratio`.
    kind : str
        The ``kind`` that was passed to `odds_ratio`. This will be
        either ``'conditional'`` or ``'sample'``.
    alternative : str
        The ``alternative`` that was passed to `odds_ratio`.  This will
        be ``'two-sided'``, ``'less'`` or ``'greater'``.
    odds_ratio : float
        The computed odds ratio.
    pvalue : float
        The p-value of the estimate of the odds ratio.

    Methods
    -------
    odds_ratio_ci
    """

    table: object
    kind: str
    alternative: str
    odds_ratio: float
    pvalue: float

    def odds_ratio_ci(self, confidence_level):
        """
        Confidence interval for the odds ratio.

        Parameters
        ----------
        confidence_level: float
            Desired confidence level for the confidence interval.
            The value must be given as a fraction between 0 and 1.

        Returns
        -------
        ci : ``ConfidenceInterval`` instance
            The confidence interval, represented as an object with
            attributes ``low`` and ``high``.

        Notes
        -----
        When ``kind`` is `'conditional'`, the limits of the confidence
        interval are the conditional "exact confidence limits" as defined in
        section 2 of Cornfield [2]_, and originally described by Fisher [1]_.
        The conditional odds ratio and confidence interval are also discussed
        in Section 4.1.2 of the text by Sahai and Khurshid [3]_.

        When ``kind`` is ``'sample'``, the confidence interval is computed
        under the assumption that the logarithm of the odds ratio is normally
        distributed with standard error given by::

            se = sqrt(1/a + 1/b + 1/c + 1/d)

        where ``a``, ``b``, ``c`` and ``d`` are the elements of the
        contingency table.  (See, for example, [3]_, section 3.1.3.2,
        or [4]_, section 2.3.3).

        References
        ----------
        .. [1] R. A. Fisher (1935), The logic of inductive inference,
               Journal of the Royal Statistical Society, Vol. 98, No. 1,
               pp. 39-82.
        .. [2] J. Cornfield (1956), A statistical problem arising from
               retrospective studies. In Neyman, J. (ed.), Proceedings of
               the Third Berkeley Symposium on Mathematical Statistics
               and Probability 4, pp. 135-148.
        .. [3] H. Sahai and A. Khurshid (1996), Statistics in Epidemiology:
               Methods, Techniques, and Applications, CRC Press LLC, Boca
               Raton, Florida.
        .. [4] Alan Agresti, An Introduction to Categorical Data Analyis
               (second edition), Wiley, Hoboken, NJ, USA (2007).
        """
        if self.kind == 'conditional':
            ci = self._conditional_odds_ratio_ci(confidence_level)
        else:
            ci = self._sample_odds_ratio_ci(confidence_level)
        return ci

    def _conditional_odds_ratio_ci(self, confidence_level):
        """
        Confidence interval for the conditional odds ratio.
        """
        if confidence_level < 0 or confidence_level > 1:
            raise ValueError('confidence_level must be between 0 and 1')

        if _row_or_column_zero(self.table):
            # If both values in a row or column are zero, the p-value is 1,
            # the odds ratio is NaN and the confidence interval is (0, inf).
            ci = (0, mp.inf)
        else:
            ci = _conditional_oddsratio_ci(self.table,
                                           confidence_level=confidence_level,
                                           alternative=self.alternative)
        return ConfidenceInterval(low=ci[0], high=ci[1])

    def _sample_odds_ratio_ci(self, confidence_level):
        """
        Confidence interval for the sample odds ratio.
        """
        if confidence_level < 0 or confidence_level > 1:
            raise ValueError('confidence_level must be between 0 and 1')

        if _row_or_column_zero(self.table):
            # If both values in a row or column are zero, the p-value is 1,
            # the odds ratio is NaN and the confidence interval is (0, inf).
            ci = (0, mp.inf)
        else:
            ci = _sample_odds_ratio_ci(self.table,
                                       confidence_level=confidence_level,
                                       alternative=self.alternative)
        return ConfidenceInterval(low=ci[0], high=ci[1])


def odds_ratio(table, kind='conditional', alternative='two-sided'):
    r"""
    Compute the odds ratio for a 2x2 contingency table.

    Parameters
    ----------
    table : array_like of ints
        A 2x2 contingency table.  Elements must be non-negative integers.
    kind : str, optional
        Which kind of odds ratio to compute, either the sample
        odds ratio (``kind='sample'``) or the conditional odds ratio
        (``kind='conditional'``).  Default is ``'conditional'``.
    alternative : {'two-sided', 'less', 'greater'}, optional
        Defines the alternative hypothesis.
        The following options are available (default is 'two-sided'):

        * 'two-sided'
        * 'less': one-sided
        * 'greater': one-sided

    Returns
    -------
    result : `OddsRatioResult` instance
        The returned object has two computed attributes:

        odds_ratio : mpmath.mpf
            * If `kind` is ``'sample'``, this is
              ``table[0, 0]*table[1, 1]/(table[0, 1]*table[1, 0])``.
              This is the prior odds ratio and not a posterior estimate.
            * If `kind` is ``'conditional'``, this is the conditional
              maximum likelihood estimate for the odds ratio. It is
              the noncentrality parameter of Fisher's noncentral
              hypergeometric distribution with the same hypergeometric
              parameters as `table` and whose mean is ``table[0, 0]``.
        pvalue : fractions.Fraction or mpmath.mpf
            The p-value associated with the computed odds ratio.

            * If `kind` is ``'sample'``, the p-value is based on the
              normal approximation to the distribution of the log of
              the sample odds ratio.
            * If `kind` is ``'conditional'``, the p-value is computed
              by :func:`mpsci.stats.fisher_exact`.

        The object also stores the input arguments `table`, `kind`
        and `alternative` as attributes.

        The object has the method `odds_ratio_ci` that computes
        the confidence interval of the odds ratio.

    References
    ----------
    .. [1] J. Cornfield (1956), A statistical problem arising from
           retrospective studies. In Neyman, J. (ed.), Proceedings of
           the Third Berkeley Symposium on Mathematical Statistics and
           Probability 4, pp. 135-148.
    .. [2] H. Sahai and A. Khurshid (1996), Statistics in Epidemiology:
           Methods, Techniques, and Applications, CRC Press LLC, Boca
           Raton, Florida.

    """
    if kind not in ['conditional', 'sample']:
        raise ValueError("kind must be 'conditional' or 'sample'.")
    if alternative not in ['two-sided', 'less', 'greater']:
        raise ValueError("alternative must be 'two-sided', 'less' or "
                         "'greater'.")

    if len(table) != 2 or (len(table[0]) != 2 or len(table[1]) != 2):
        raise ValueError("The input `table` must be shaped like a 2x2 array.")

    a, b, c, d = _unpack_table_to_mpf(table)
    if a < 0 or b < 0 or c < 0 or d < 0:
        raise ValueError("All values in `table` must be nonnegative.")

    if _row_or_column_zero(table):
        # If both values in a row or column are zero, the p-value is 1 and
        # the odds ratio is NaN.
        result = OddsRatioResult(table=table, kind=kind,
                                 alternative=alternative,
                                 odds_ratio=mp.nan, pvalue=1)
        return result

    if kind == 'sample':
        oddsratio = _sample_odds_ratio(table)
        log_or = mp.log(oddsratio)
        se = mp.sqrt(1/a + 1/b + 1/c + 1/d)
        if alternative == 'two-sided':
            pvalue = 2*mp.ncdf(-abs(log_or)/se)
        elif alternative == 'less':
            pvalue = mp.ncdf(log_or/se)
        else:
            pvalue = mp.ncdf(-log_or/se)
    else:
        # kind is 'conditional'
        oddsratio = _conditional_oddsratio(table)
        # We can use fisher_exact to compute the p-value.
        pvalue = fisher_exact(table, alternative=alternative)[1]

    result = OddsRatioResult(table=table, kind=kind, alternative=alternative,
                             odds_ratio=oddsratio, pvalue=pvalue)
    return result
