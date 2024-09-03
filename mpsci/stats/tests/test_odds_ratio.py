
from mpmath import mp
from mpsci.stats import odds_ratio


@mp.workdps(40)
def test_odds_ratio():
    # Test that the function call works, and that the
    # odds ratio is correct to 4 significant digits.
    # TODO: Create a test with higher precision required.
    table = [[10, 11], [37, 8]]
    result = odds_ratio(table)
    # Reference value computed with scipy.stats.contingency.odds_ratio.
    assert round(result.odds_ratio, 4) == 0.2024
