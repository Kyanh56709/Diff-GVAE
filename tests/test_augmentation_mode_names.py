"""Semantic naming of augmentation modes.

Label convention: binary_label 1 = NON-responder, 0 = responder (confirmed
2026-08-06).  The legacy mode name ``responder_only`` actually generates
class 1 (NON-responders); the explicit replacement name is
``nonresponder_only``, with ``responder_only`` kept as a deprecated alias.
"""

import numpy as np
import pytest

from training.latent_ddpm_augmentation import (
    _classes_for_augmentation,
    _normalize_augmentation_modes,
)

IMBALANCED = np.array([0] * 62 + [1] * 185, dtype=np.int64)


def test_nonresponder_only_returns_class_1():
    assert _classes_for_augmentation(IMBALANCED, "nonresponder_only") == (1,)


def test_responder_only_deprecated_alias_warns():
    with pytest.warns(DeprecationWarning, match="nonresponder_only"):
        assert _classes_for_augmentation(IMBALANCED, "responder_only") == (1,)


def test_both_classes_returns_present_classes():
    assert _classes_for_augmentation(IMBALANCED, "both_classes") == (0, 1)


def test_minority_only_returns_minority_class():
    # class 0 has 62 samples vs 185 for class 1 -> minority is class 0
    assert _classes_for_augmentation(IMBALANCED, "minority_only") == (0,)


def test_normalize_accepts_nonresponder_only():
    assert "nonresponder_only" in _normalize_augmentation_modes(
        ["nonresponder_only"]
    )


def test_normalize_still_accepts_deprecated_responder_only():
    # alias must still pass validation so the DeprecationWarning can surface
    assert "responder_only" in _normalize_augmentation_modes(["responder_only"])
