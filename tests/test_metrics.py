"""
test_metrics.py
===============
Testes unitários para cálculo de K-S, AUC e Gini.
"""

import numpy as np
import pytest

from src.metrics import calculate_ks, evaluate_model_performance


def test_calculate_ks():
    # Simula separação clara entre bons e maus
    np.random.seed(42)
    scores_goods = np.random.normal(loc=680, scale=60, size=1000)
    scores_bads = np.random.normal(loc=450, scale=70, size=300)

    scores = np.clip(np.concatenate([scores_goods, scores_bads]), 300, 850)
    y_true = np.concatenate([np.zeros(1000), np.ones(300)])

    res = calculate_ks(scores, y_true)
    assert res.ks_stat > 30.0, f"KS esperado > 30%, obtido {res.ks_stat}"
    assert 300 <= res.cutoff_at_max_ks <= 850


def test_evaluate_model_performance():
    np.random.seed(42)
    scores = np.random.randint(300, 850, size=500)
    # Probabilidade de default inversamente proporcional ao score
    prob_bad = np.clip(1.0 - (scores - 300) / 550.0, 0.01, 0.99)
    y_true = np.random.binomial(n=1, p=prob_bad)

    metrics = evaluate_model_performance(scores, prob_bad, y_true)
    assert 0.5 <= metrics.auc <= 1.0
    assert -100.0 <= metrics.gini <= 100.0
    assert pytest.approx(metrics.gini, abs=0.5) == (2.0 * metrics.auc - 1.0) * 100.0
