
import pytest
from mpsci.distributions._weibull_common import _validate_params


def test_validate_params_k_bad():
    with pytest.raises(ValueError, match='k must be positive'):
        _validate_params(-3.5, 0, 1)


def test_validate_params_scale_bad():
    with pytest.raises(ValueError, match='scale must be positive'):
        _validate_params(3.5, 0, -1.25)
