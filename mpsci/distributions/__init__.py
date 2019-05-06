"""
``distributions``
-----------------

A few probability distributions.

Note that these implementations do not necessarily use the same parametrization
as the corresponding implementations in `scipy.stats`.

"""

from .continuous import cosine
from .continuous import exponweib
from .continuous import gamma
from .continuous import genextreme
from .continuous import geninvgauss
from .continuous import ncx2
from .continuous import normal
from .continuous import rice

from .discrete import hypergeometric
from .discrete import fishers_noncentral_hypergeometric
