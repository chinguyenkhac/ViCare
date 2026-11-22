import numpy as np

from vicare_safe.confidence import ConfidenceModel, features, level_of, pick_oos_threshold, reliability_by_level
from vicare_safe.metrics import auroc, bootstrap_ci, ece, mcnemar_p, retrieval_summary


def test_logistic_learns_separable_signal():
    rng = np.random.default_rng(0)
    x_pos = rng.normal(2, 0.5, (100, 8)); x_neg = rng.normal(-2, 0.5, (100, 8))
    X = np.vstack([x_pos, x_neg]); y = np.r_[np.ones(100), np.zeros(100)]
    m = ConfidenceModel().fit(X, y)
    p = m.predict_proba(X)
    assert p[:100].mean() > 0.9 and p[100:].mean() < 0.1


def test_features_shape():
    hits = [{"score": 0.9, "dense_rank": 1, "bm25_rank": 2, "dense_cos": 0.88}, {"score": 0.3}]
    f = features(hits, "bé bị sốt", restored=True)
    assert len(f) == 8 and f[3] == 1.0 and f[5] == 1.0


def test_levels():
    assert level_of(0.8) == "Cao" and level_of(0.5) == "Trung bình" and level_of(0.1) == "Thấp"
    rows = reliability_by_level([0.9, 0.8, 0.1], [1, 1, 0])
    assert rows[0]["n"] == 2 and rows[0]["actual_acc"] == 1.0


def test_threshold_and_metrics():
    assert pick_oos_threshold([0.9, 0.95], [0.1, 0.2], 1.0) > 0.2
    assert retrieval_summary([1, 3, None, 11])["R@5"] == 0.5
    assert auroc([0.9, 0.8], [0.1, 0.2]) == 1.0
    assert ece([1.0, 0.0], [1, 0]) == 0.0
    p, a_only, b_only = mcnemar_p([1, 1, 0, 0], [1, 0, 1, 1])
    assert (a_only, b_only) == (1, 2) and 0 < p <= 1
    lo, hi = bootstrap_ci([1, 0, 1, 1])
    assert 0 <= lo <= hi <= 1
