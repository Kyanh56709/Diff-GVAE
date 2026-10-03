"""a3/a4 research scripts: default run is the historical A2 run, --run-id
overrides it, missing dirs fail loudly, ambiguous rank_1 dirs fail loudly."""
import importlib.util
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "research/2026-08-06-project-review-audit/scripts"
HIST = "conditional_latent_ddpm_from_gvae_bestparam_ranked_20260807_095233_20260810_153849"


def _load(name):
    spec = importlib.util.spec_from_file_location(name, SCRIPTS / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(params=["a3_filter_quantile_retune", "a4_tstr_control"])
def script(request):
    return _load(request.param)


def test_default_run_is_historical(script, monkeypatch, tmp_path):
    (tmp_path / "outputs/conditional_latent_ddpm" / HIST).mkdir(parents=True)
    monkeypatch.chdir(tmp_path)
    assert script.parse_args([]) == Path("outputs/conditional_latent_ddpm") / HIST


def test_run_id_override(script, monkeypatch, tmp_path):
    (tmp_path / "outputs/conditional_latent_ddpm/new_run").mkdir(parents=True)
    monkeypatch.chdir(tmp_path)
    assert script.parse_args(["--run-id", "new_run"]) == Path("outputs/conditional_latent_ddpm/new_run")


def test_missing_run_dir_exits(script, monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    with pytest.raises(SystemExit):
        script.parse_args(["--run-id", "does_not_exist"])


def test_rank1_dir_unique(script, tmp_path):
    (tmp_path / "rank_1_a").mkdir()
    (tmp_path / "rank_2_b").mkdir()
    assert script.rank1_dir(tmp_path).name == "rank_1_a"


def test_rank1_dir_ambiguous_fails(script, tmp_path):
    (tmp_path / "rank_1_a").mkdir()
    (tmp_path / "rank_1_b").mkdir()
    with pytest.raises(AssertionError):
        script.rank1_dir(tmp_path)
