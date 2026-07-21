import numpy as np
from sklearn.preprocessing import StandardScaler as RealScaler

import training.latent_ddpm_augmentation as mod


def _make_spy(store):
    class SpyScaler(RealScaler):
        def fit_transform(self, X, y=None, **kw):
            arr = np.asarray(X)
            store["n_rows"] = int(arr.shape[0])
            store["mean_abs_max"] = float(np.abs(arr.mean(axis=0)).max())
            return super().fit_transform(X, y, **kw)

    return SpyScaler


def _split():
    rng = np.random.default_rng(0)
    x_train = rng.standard_normal((8, 5)).astype("float32")            # mean ~0
    x_val = (rng.standard_normal((4, 5)) + 100.0).astype("float32")    # mean ~100
    y_train = np.array([0, 1, 0, 1, 0, 1, 0, 1])
    y_val = np.array([0, 1, 0, 1])
    return x_train, y_train, x_val, y_val


def test_downstream_scaler_fits_train_only(monkeypatch):
    captured = {}
    monkeypatch.setattr(mod, "StandardScaler", _make_spy(captured))
    x_train, y_train, x_val, y_val = _split()
    mod.train_downstream_classifier(x_train, y_train, x_val, y_val)
    assert captured["n_rows"] == 8          # only the 8 TRAIN rows were fit
    assert captured["mean_abs_max"] < 5.0   # train mean ~0, not the val ~100 mean


def test_spy_assertions_actually_detect_leakage():
    # Sanity check on the discriminator itself: a deliberately LEAKY fit (train+val)
    # trips both assertions. Defined locally — the tracked module is never mutated.
    captured = {}
    Spy = _make_spy(captured)
    x_train, _yt, x_val, _yv = _split()

    def leaky_fit(xt, xv):
        Spy().fit_transform(np.vstack([xt, xv]))  # the bug we are guarding against

    leaky_fit(x_train, x_val)
    assert captured["n_rows"] == 12             # a leaky fit sees all 12 rows
    assert captured["mean_abs_max"] > 5.0       # and the mean is pulled toward val's ~100
