import math
from fractions import Fraction


__all__ = ['fisher_exact']


def _comb(n, k):
    """
    A fast way to calculate binomial coefficients by Andrew Dalke (contrib).
    """
    # This was taken from an answer on stackoverflow.
    if 0 <= k <= n:
        ntok = 1
        ktok = 1
        for t in range(1, min(k, n - k) + 1):
            ntok *= n
            ktok *= t
            n -= 1
        return ntok // ktok
    else:
        return 0


def _num_combs(a, row0sum, col0sum, n):
    m01 = row0sum - a
    m10 = col0sum - a
    m11 = n - (a + m01 + m10)
    num = _comb(a + m01, a) * _comb(m10 + m11, m10)
    return num


def fisher_exact(table, alternative='two-sided'):
    """
    Compute Fisher's exact test on the 2x2 table.

    This function uses python integers to compute the probability exactly.
    (Unlike most of the other functions in the ``mpsci`` library, this function
    does not use ``mpmath``.)

    Returns the sample odds ratio and the p-value.  Generally the return
    values will be instances of ``fractions.Fraction``, but the odds ratio
    will be ``math.nan`` if both ``a*d`` and ``b*c`` are zero, and it will
    be ``math.inf`` if ``b*c`` is 0 and ``a*d`` is not zero, where
    ``a = table[0][0]``, ``b = table[0][1]``, ``c = table[1][0]`` and
    ``d = table[1][1]``.

    *Warning:* The values in ``table`` should not be large!  The calculation
    can take a very long time and use a lot of memory if the values are large.
    (The precise definition of "large" will depend on the computer's speed
    and available memory.)

    Examples
    --------
    >>> from mpsci.stats import fisher_exact

    >>> table = [[8, 2], [1, 5]]
    >>> oddsratio, pvalue = fisher_exact(table)
    >>> oddsratio, float(oddsratio)
    (Fraction(20, 1), 20.0)
    >>> pvalue, float(pvalue)
    (Fraction(5, 143), 0.03496503496503497)

    An example with bigger values:

    >>> table = [[345, 455], [260, 345]]
    >>> oddsratio, pvalue = fisher_exact(table)
    >>> pvalue  # The following output is edited for this docstring.
    Fraction(5815594848535879587533843493536161337167965268243476365254048617
    5074871119589194456724424702642108585451894970002396727579577842267797518
    4929838662845360913926006683618646889965042763561174204677207235122556332
    9503454183984380902895184450793815543305546132593168918382926520280514271
    135809969597, 60789478542597481225083508544099336316985118448918449312948
    2297975307374215835510960358633704364890832917577706721466773385015807270
    0232888943268164264596951845578381273218929422021631512908332122679219965
    7632047453216230717808020513225358000202208441824610229474920736533616413
    69838473616509597)
    >>> float(pvalue)
    0.9566778639926435
    """

    a, b = int(table[0][0]), int(table[0][1])
    c, d = int(table[1][0]), int(table[1][1])

    # Sample odds ratio, *not* the MLE odds ratio!
    if b*c != 0:
        oddsratio = Fraction(a*d, b*c)
    else:
        if a*d != 0:
            oddsratio = math.inf
        else:
            oddsratio = math.nan

    row0sum = a + b
    row1sum = c + d
    col0sum = a + c

    n = row0sum + row1sum
    total_possible_tables = _comb(n, col0sum)

    a_max = min(row0sum, col0sum)
    a_min = max(0, (col0sum - row1sum))

    if alternative == "two-sided":
        tcount = _comb(row0sum, a) * _comb(row1sum, c)
        count = 0

        for m00 in range(a_min, a_max + 1):
            num = _num_combs(m00, row0sum, col0sum, n)
            if num > tcount:
                break
            count += num

        for m00 in range(a_max, m00, -1):
            num = _num_combs(m00, row0sum, col0sum, n)
            if num > tcount:
                break
            count += num

        p = Fraction(count, total_possible_tables)
        return oddsratio, p

    if alternative == "greater":
        start, stop = a, a_max
    elif alternative == "less":
        start, stop = a_min, a
    else:
        raise ValueError("invalid alternative %r" % (alternative,))

    count = 0
    for m00 in range(start, stop + 1):
        count += _num_combs(m00, row0sum, col0sum, n)

    p = Fraction(count, total_possible_tables)

    return oddsratio, p
