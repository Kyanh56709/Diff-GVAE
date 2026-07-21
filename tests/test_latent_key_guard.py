import pytest
from utils.latent_extraction import (
    _RECOMMENDED_LATENT_KEYS,
    _validate_ddpm_latent_key,
)


def test_concat_mu_is_always_allowed():
    _validate_ddpm_latent_key("concat_mu", _RECOMMENDED_LATENT_KEYS, allow_non_concat_mu=False)


def test_unknown_key_raises():
    with pytest.raises(ValueError, match="Unknown latent_key"):
        _validate_ddpm_latent_key("bogus", _RECOMMENDED_LATENT_KEYS, allow_non_concat_mu=False)


def test_non_concat_key_requires_optin():
    with pytest.raises(ValueError, match="opt in"):
        _validate_ddpm_latent_key("fused_cls_mu", _RECOMMENDED_LATENT_KEYS, allow_non_concat_mu=False)


def test_non_concat_key_allowed_with_optin():
    _validate_ddpm_latent_key("fused_cls_mu", _RECOMMENDED_LATENT_KEYS, allow_non_concat_mu=True)
