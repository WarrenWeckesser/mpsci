"""
``stats``
---------

Most of these functions were implemented to test their counterparts
in `scipy.stats`.

"""

from ._basic import *
from ._goftests import *
from ._pearsonr import pearsonr, pearsonr_ci
from ._anova import anova_oneway
from ._fisher_exact import fisher_exact
