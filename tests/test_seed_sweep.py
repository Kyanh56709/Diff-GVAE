from review_fixes_2026_07.seed_sweep import aggregate_seed_results, run_seed_sweep


def test_aggregate_only_mean_keys():
    summaries = [
        {"mean_roc_auc": 0.6, "std_roc_auc": 0.10},
        {"mean_roc_auc": 0.8, "std_roc_auc": 0.10},
    ]
    agg = aggregate_seed_results(summaries)
    assert set(agg) == {"mean_roc_auc"}          # std_* is not aggregated
    assert agg["mean_roc_auc"]["mean"] == 0.7
    assert agg["mean_roc_auc"]["n_seeds"] == 2
    assert agg["mean_roc_auc"]["values"] == [0.6, 0.8]


def test_run_seed_sweep_varies_seed_and_uses_train_fn():
    def fake_train(data, mc, tc):
        return {"mean_roc_auc": 0.5 + 0.05 * tc["random_seed"]}, None, None

    res = run_seed_sweep(None, {}, {"random_seed": 0}, seeds=(0, 2), train_fn=fake_train)
    assert [r["seed"] for r in res["per_seed"]] == [0, 2]
    assert res["aggregate"]["mean_roc_auc"]["n_seeds"] == 2
