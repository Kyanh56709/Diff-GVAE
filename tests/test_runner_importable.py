"""Regression test: outputs/gvae runners must be importable on Python 3.9.

The conditional DDPM augmentation runner used PEP 604 union syntax
(``int | None``) in function signatures without ``from __future__ import
annotations``.  On Python < 3.10 this crashes at import time with
``TypeError: unsupported operand type(s) for |: 'type' and 'NoneType'``.
"""


def test_conditional_ddpm_augmentation_runner_importable():
    import outputs.gvae.train_conditional_ddpm_augmentation_runner  # noqa: F401


def test_gvae_runner_importable():
    import outputs.gvae.train_gvae_runner  # noqa: F401
