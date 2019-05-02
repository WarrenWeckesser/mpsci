"""
``distributions``
-----------------

A few probability distributions.

Note that these implementations do not necessarily use the same parametrization
as the corresponding implementations in `scipy.stats`.

"""

from . import exponweib
from . import gamma
from . import genextreme
from . import geninvgauss
from . import ncx2
from . import normal
from . import rice

from . import hypergeometric
from . import fishers_noncentral_hypergeometric
