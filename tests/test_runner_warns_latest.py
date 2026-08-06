"""The augmentation runner must warn when auto-selecting the latest run.

When ``--gvae-run-id`` is omitted and ``latest_gvae_run_id()`` picks the
run, a WARNING line must be printed to stderr so the implicit choice is
visible in logs.
"""

import os

from outputs.gvae.train_conditional_ddpm_augmentation_runner import (
    _resolve_gvae_run_id,
    latest_gvae_run_id,
)

WARNING_PREFIX = (
    "WARNING: no --gvae-run-id given; using latest run by directory mtime: "
)


def _make_checkpoint_root(tmp_path, run_ids):
    root = tmp_path / 'checkpoints'
    for i, run_id in enumerate(run_ids):
        run_dir = root / run_id
        run_dir.mkdir(parents=True)
        # deterministic, strictly increasing mtimes
        os.utime(run_dir, (1_600_000_000 + i, 1_600_000_000 + i))
    return root


def test_no_warning_when_run_id_explicit(tmp_path, capsys):
    root = _make_checkpoint_root(tmp_path, ['run_aaa', 'run_bbb'])
    result = _resolve_gvae_run_id('run_aaa', root)
    captured = capsys.readouterr()
    assert result == 'run_aaa'
    assert captured.err == ''


def test_warns_on_stderr_when_run_id_omitted(tmp_path, capsys):
    root = _make_checkpoint_root(tmp_path, ['run_aaa', 'run_bbb'])
    result = _resolve_gvae_run_id(None, root)
    captured = capsys.readouterr()
    assert result == latest_gvae_run_id(root) == 'run_bbb'
    assert WARNING_PREFIX + 'run_bbb' in captured.err
    # the warning must not pollute stdout (used for machine-readable output)
    assert 'WARNING' not in captured.out


def test_warned_run_matches_latest_by_mtime(tmp_path, capsys):
    root = _make_checkpoint_root(tmp_path, ['run_old', 'run_mid', 'run_new'])
    result = _resolve_gvae_run_id(None, root)
    capsys.readouterr()
    assert result == 'run_new'
