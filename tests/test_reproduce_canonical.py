"""H2 — the one-command canonical reproduction driver (dry-run contract)."""
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _dry(steps):
    out = subprocess.run(
        [sys.executable, str(ROOT / "reproduce_canonical.py"), "--dry-run", "--steps", steps],
        capture_output=True, text=True, cwd=ROOT)
    assert out.returncode == 0, out.stderr
    return out.stdout


def test_dry_run_lists_all_canonical_steps():
    s = _dry("graph,validate,oof,gvae-latent,ddpm")
    assert "build_ln_pc_ihc_g.py" in s and "--drop-radiology-artifacts both" in s
    assert "data_validation.py" in s
    assert s.count("run_oof.py") == 5  # seeds 42-46
    assert "run_bestparam_ranked_r32.py" in s
    assert "run_ddpm_r32.sh" in s
    assert "REPRODUCE_DONE (dry-run)" in s


def test_steps_subset_excludes_others():
    s = _dry("graph")
    assert "build_ln_pc_ihc_g.py" in s
    assert "run_oof.py" not in s and "run_ddpm_r32.sh" not in s


def test_ddpm_without_gvae_latent_is_rejected():
    out = subprocess.run(
        [sys.executable, str(ROOT / "reproduce_canonical.py"), "--dry-run", "--steps", "ddpm"],
        capture_output=True, text=True, cwd=ROOT)
    assert out.returncode != 0
    assert "gvae-latent" in (out.stderr + out.stdout)


def test_out_of_order_input_is_normalized_to_canonical_order():
    s = _dry("ddpm,gvae-latent")
    assert s.index("run_bestparam_ranked_r32.py") < s.index("run_ddpm_r32.sh")


def test_unknown_step_is_rejected():
    out = subprocess.run(
        [sys.executable, str(ROOT / "reproduce_canonical.py"), "--dry-run", "--steps", "bogus"],
        capture_output=True, text=True, cwd=ROOT)
    assert out.returncode != 0
    assert "unknown step" in (out.stderr + out.stdout)
